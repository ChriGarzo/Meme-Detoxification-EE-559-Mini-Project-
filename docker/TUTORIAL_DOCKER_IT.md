# Tutorial: Creare e pubblicare l'immagine Docker sul cluster EPFL

Questo tutorial spiega come costruire l'immagine Docker del progetto e caricarla sul registry del cluster RCP di EPFL. **Ogni membro del gruppo deve farlo una volta sola dal proprio computer.**

---

## Cosa ti serve

- Docker Desktop installato sul tuo computer (scaricabile da https://www.docker.com/products/docker-desktop)
- Le credenziali GASPAR (username e password EPFL)
- Il repository del progetto clonato in locale

---

## Passo 1 — Crea il progetto su Harbor (il registry di EPFL)

Harbor è il registro Docker del cluster. Devi creare un progetto pubblico sotto il tuo nome prima di poter caricare l'immagine.

1. Vai su **https://registry.rcp.epfl.ch** e accedi con le tue credenziali GASPAR.
2. Clicca su **"New Project"** (o **"Nuovo Progetto"**).
3. Inserisci come nome: `ee-559-<tuo_username_gaspar>` (es. `ee-559-garzone`).
4. Imposta l'accesso a **Public** (obbligatorio — il cluster deve poter scaricare l'immagine senza autenticazione).
5. Clicca **OK** per creare il progetto.

> ⚠️ Il nome del progetto deve essere esattamente `ee-559-<tuo_username_gaspar>`, altrimenti gli script RunAI non troveranno l'immagine.

---

## Passo 2 — Apri il terminale nella cartella del progetto

Apri il terminale (o PowerShell su Windows) e spostati nella cartella `hateful_meme_rewriting/`:

```
cd percorso/alla/cartella/hateful_meme_rewriting
```

Tutti i comandi dei passi successivi vanno eseguiti da questa cartella.

---

## Passo 3 — Costruisci l'immagine Docker

Questo comando crea l'immagine a partire dal `Dockerfile` nella cartella `docker/`.

**Su Linux o Mac:**
```bash
docker build --platform linux/amd64 -t registry.rcp.epfl.ch/ee-559-<tuo_username>/hmr:v0.1 docker/
```

**Su Windows (PowerShell):**
```powershell
docker build --platform linux/amd64 -t registry.rcp.epfl.ch/ee-559-<tuo_username>/hmr:v0.1 docker/
```

Sostituisci `<tuo_username>` con il tuo username GASPAR (es. `garzone`).

> Il flag `--platform linux/amd64` è necessario perché il cluster usa Linux a 64 bit. Su Mac con chip Apple Silicon (M1/M2/M3) senza questo flag l'immagine verrebbe costruita per ARM e non funzionerebbe sul cluster.

La costruzione richiede **10–20 minuti** la prima volta perché scarica e installa tutti i pacchetti Python. Le volte successive sarà molto più veloce grazie alla cache di Docker.

Al termine vedrai una riga simile a:
```
=> => writing image sha256:abc123...   0.0s
=> => naming to registry.rcp.epfl.ch/ee-559-<tuo_username>/hmr:v0.1
```

---

## Passo 4 — Accedi al registry

Prima di poter caricare l'immagine, devi autenticarti con le tue credenziali GASPAR:

```bash
docker login registry.rcp.epfl.ch
```

Inserisci username e password GASPAR quando richiesto. Vedrai:
```
Login Succeeded
```

---

## Passo 5 — Carica l'immagine sul registry

```bash
docker push registry.rcp.epfl.ch/ee-559-<tuo_username>/hmr:v0.1
```

Il caricamento richiede tempo perché l'immagine è grande (diversi GB). **Non chiudere il terminale e non mettere il computer in standby** durante l'operazione.

Se la connessione si interrompe a metà (errore tipo `use of closed network connection`), rilancia semplicemente lo stesso comando: Docker riprende dal punto in cui si era fermato, senza ricaricare i layer già caricati.

Al termine vedrai qualcosa come:
```
v0.1: digest: sha256:abc123... size: 1234
```

---

## Passo 6 — Verifica (opzionale)

Per verificare che l'immagine sia stata caricata correttamente:

1. Vai su **https://registry.rcp.epfl.ch**
2. Apri il progetto `ee-559-<tuo_username>`
3. Clicca su **Repositories** → dovresti vedere `hmr` con il tag `v0.1`

---

## Riepilogo dei comandi

```bash
# 1. Costruisci l'immagine (dalla cartella hateful_meme_rewriting/)
docker build --platform linux/amd64 -t registry.rcp.epfl.ch/ee-559-<tuo_username>/hmr:v0.1 docker/

# 2. Accedi al registry
docker login registry.rcp.epfl.ch

# 3. Carica l'immagine
docker push registry.rcp.epfl.ch/ee-559-<tuo_username>/hmr:v0.1
```

---

## Cosa succede dopo

Una volta caricata l'immagine, puoi eseguire qualsiasi stage della pipeline sul cluster con gli script nella cartella `scripts/`. Esempio per lo Stage 0:

```bash
# Prima accedi al cluster via SSH
ssh <tuo_username>@jumphost.rcp.epfl.ch

# Poi ottieni il tuo UID numerico
id -u

# Poi esegui lo script (sostituisci 123456 con il tuo UID)
bash scripts/runai_stage0_filter.sh 123456 harmeme
```

Gli script usano automaticamente il tuo `$USER` per trovare la tua immagine Docker — non devi modificare nulla.

Per l'ordine completo degli stage da eseguire, consulta il `README.md` nella root del progetto.
