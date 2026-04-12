import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

from models.rewriter import MemeRewriter


logger = logging.getLogger(__name__)


class ExplanationProxy(nn.Module):
    """
    Neural network to predict BART encoder hidden states from CLIP embeddings.

    Input: concatenated CLIP image and text embeddings [B, 1536]
    Output: predicted BART encoder hidden state [B, bart_hidden_size]
    """

    def __init__(self, bart_hidden_size: int = 1024):
        """
        Initialize ExplanationProxy network.

        Args:
            bart_hidden_size: Hidden size of the target BART model
                             (1024 for bart-large, 768 for bart-base)
        """
        super().__init__()

        self.bart_hidden_size = bart_hidden_size

        # Layer 1: Linear + LayerNorm + GELU
        self.layer1 = nn.Linear(1536, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.act1 = nn.GELU()

        # Layer 2: Linear + LayerNorm + GELU
        self.layer2 = nn.Linear(1024, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.act2 = nn.GELU()

        # Layer 3: Linear to output
        self.layer3 = nn.Linear(1024, bart_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 1536]

        Returns:
            Output tensor [B, bart_hidden_size]
        """
        # Layer 1
        x = self.layer1(x)
        x = self.ln1(x)
        x = self.act1(x)

        # Layer 2
        x = self.layer2(x)
        x = self.ln2(x)
        x = self.act2(x)

        # Layer 3
        x = self.layer3(x)

        return x


class ExplanationProxyTrainer:
    """Trainer for the ExplanationProxy network."""

    def __init__(
        self,
        rewriter: MemeRewriter,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize ExplanationProxyTrainer.

        Args:
            rewriter: MemeRewriter instance for getting BART hidden states
            clip_model_name: HuggingFace model identifier for CLIP
            device: Device to run on ('cuda', 'cpu'). Auto-detected if None.
        """
        self.rewriter = rewriter
        self.clip_model_name = clip_model_name
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP (must use clip-vit-large-patch14: 768-dim embeds → 1536 concat)
        logger.info(f"Loading CLIP model {clip_model_name}...")
        self.clip_model = CLIPModel.from_pretrained(
            clip_model_name, cache_dir=cache_dir
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            clip_model_name, cache_dir=cache_dir
        )
        self.clip_model.eval()

        # Initialize proxy network
        bart_hidden_size = rewriter.hidden_size or 1024
        self.proxy = ExplanationProxy(
            bart_hidden_size=bart_hidden_size
        ).to(self.device)

        logger.info(
            f"ExplanationProxyTrainer initialized on device: {self.device}"
        )

    def extract_clip_features(
        self,
        images: List,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Extract and concatenate CLIP image + text embeddings.

        Args:
            images: List of PIL Images or image paths
            texts: List of text strings

        Returns:
            Tensor of shape [N, 1536] with concatenated embeddings
        """
        # Handle image paths
        processed_images = []
        for img in images:
            if isinstance(img, str):
                from PIL import Image

                img = Image.open(img).convert("RGB")
            processed_images.append(img)

        # Process with CLIP
        inputs = self.clip_processor(
            images=processed_images,
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # clip-vit-large-patch14 → 768-dim each → 1536 concatenated
            image_embeds = outputs.image_embeds  # [N, 768]
            text_embeds = outputs.text_embeds    # [N, 768]

        # Concatenate → [N, 1536]  (matches ExplanationProxy input dim)
        combined = torch.cat(
            [image_embeds, text_embeds], dim=1
        )  # [N, 1536]

        return combined

    def extract_bart_targets(
        self,
        texts: List[str],
        target_groups: List[Optional[str]],
        attack_types: List[Optional[str]],
        implicit_meanings: List[Optional[str]],
    ) -> torch.Tensor:
        """
        Extract BART encoder hidden states as training targets.

        Args:
            texts: List of original texts
            target_groups: List of target groups
            attack_types: List of attack types
            implicit_meanings: List of implicit meanings

        Returns:
            Tensor of shape [N, bart_hidden_size] with encoder hidden states
        """
        if self.rewriter.model is None:
            self.rewriter.load_model()

        hidden_states = []

        for text, tg, at, im in zip(
            texts, target_groups, attack_types, implicit_meanings
        ):
            h = self.rewriter.get_encoder_hidden_state(
                text,
                target_group=tg,
                attack_type=at,
                implicit_meaning=im,
                mode="full",
            )
            hidden_states.append(h)

        # Stack along batch dimension: [N, 1, hidden_size] -> [N, hidden_size]
        targets = torch.cat(hidden_states, dim=0)

        return targets

    def train(
        self,
        images: List,
        texts: List[str],
        target_groups: List[Optional[str]],
        attack_types: List[Optional[str]],
        implicit_meanings: List[Optional[str]],
        val_images: Optional[List] = None,
        val_texts: Optional[List[str]] = None,
        val_target_groups: Optional[List[Optional[str]]] = None,
        val_attack_types: Optional[List[Optional[str]]] = None,
        val_implicit_meanings: Optional[List[Optional[str]]] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the ExplanationProxy network.

        Args:
            images: List of training images
            texts: List of training texts
            target_groups: List of training target groups
            attack_types: List of training attack types
            implicit_meanings: List of training implicit meanings
            val_images: List of validation images (optional)
            val_texts: List of validation texts (optional)
            val_target_groups: List of validation target groups (optional)
            val_attack_types: List of validation attack types (optional)
            val_implicit_meanings: List of validation implicit meanings (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for Adam optimizer
            save_dir: Directory to save best checkpoint

        Returns:
            Dictionary with training history (loss curves)
        """
        logger.info("Extracting training features...")
        train_features = self.extract_clip_features(images, texts)
        train_targets = self.extract_bart_targets(
            texts, target_groups, attack_types, implicit_meanings
        )

        has_val = all(
            x is not None
            for x in [
                val_images,
                val_texts,
                val_target_groups,
                val_attack_types,
                val_implicit_meanings,
            ]
        )

        if has_val:
            logger.info("Extracting validation features...")
            val_features = self.extract_clip_features(val_images, val_texts)
            val_targets = self.extract_bart_targets(
                val_texts, val_target_groups, val_attack_types, val_implicit_meanings
            )

        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(
            train_features, train_targets
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Setup optimizer and loss
        optimizer = Adam(self.proxy.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # Training loop
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        best_val_loss = float("inf")

        self.proxy.train()

        for epoch in range(num_epochs):
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions = self.proxy(batch_features)
                loss = loss_fn(predictions, batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            if has_val:
                val_loss = 0.0
                self.proxy.eval()
                with torch.no_grad():
                    val_features = val_features.to(self.device)
                    val_targets = val_targets.to(self.device)
                    val_predictions = self.proxy(val_features)
                    val_loss = loss_fn(val_predictions, val_targets).item()

                history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )

                # Save best checkpoint
                if save_dir and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = Path(save_dir) / "best_proxy.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.proxy.state_dict(), save_path)
                    logger.info(f"Saved best checkpoint to {save_path}")

                self.proxy.train()
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.6f}"
                )

        return history

    def evaluate(
        self,
        images: List,
        texts: List[str],
        target_groups: List[Optional[str]],
        attack_types: List[Optional[str]],
        implicit_meanings: List[Optional[str]],
    ) -> Dict[str, float]:
        """
        Evaluate the ExplanationProxy on a dataset.

        Args:
            images: List of evaluation images
            texts: List of evaluation texts
            target_groups: List of target groups
            attack_types: List of attack types
            implicit_meanings: List of implicit meanings

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Extracting evaluation features...")
        features = self.extract_clip_features(images, texts)
        targets = self.extract_bart_targets(
            texts, target_groups, attack_types, implicit_meanings
        )

        # Evaluation
        loss_fn = nn.MSELoss()
        self.proxy.eval()

        with torch.no_grad():
            features = features.to(self.device)
            targets = targets.to(self.device)
            predictions = self.proxy(features)
            mse_loss = loss_fn(predictions, targets).item()

        results = {
            "mse_loss": mse_loss,
            "num_samples": len(texts),
            "model_name": self.rewriter.model_name,
            "clip_model": self.clip_model_name,
        }

        logger.info(f"Evaluation results: {results}")

        return results

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load proxy network weights from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading proxy checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.proxy.load_state_dict(state_dict)
