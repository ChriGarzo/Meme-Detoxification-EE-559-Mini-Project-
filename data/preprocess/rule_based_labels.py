"""
Rule-based keyword matchers for assigning target_group and attack_type labels
to meme text when ground-truth labels are unavailable or incomplete.
"""

# Define keyword→label mappings
# target_group keywords
TARGET_GROUP_KEYWORDS = {
    "race_ethnicity": ["race", "racial", "black", "white", "asian", "african", "latino",
                       "hispanic", "ethnic", "negro", "colored", "ape", "monkey"],
    "nationality": ["immigrant", "illegal", "foreigner", "border", "alien", "refugee",
                    "deport", "migrant", "mexican", "chinese", "indian", "arab"],
    "religion": ["muslim", "islam", "christian", "jewish", "jew", "hindu", "mosque",
                 "church", "allah", "quran", "bible", "rabbi", "imam"],
    "gender": ["woman", "women", "girl", "female", "feminist", "bitch", "slut",
               "whore", "misogyn", "patriarchy", "housewife", "kitchen"],
    "sexual_orientation": ["gay", "lesbian", "homosexual", "lgbtq", "queer", "fag",
                          "homo", "trans", "bisexual", "pride"],
    "disability": ["retard", "disabled", "cripple", "handicap", "autis", "mental",
                   "wheelchair", "deaf", "blind", "dumb"],
}

# attack_type keywords
ATTACK_TYPE_KEYWORDS = {
    "contempt": ["disgust", "pathetic", "worthless", "garbage", "trash", "scum", "filth"],
    "mocking": ["joke", "funny", "laugh", "stupid", "idiot", "dumb", "moron", "clown"],
    "inferiority": ["inferior", "lesser", "beneath", "below", "subhuman", "primitive",
                    "backward", "uncivilized", "low iq"],
    "slurs": ["nigger", "kike", "spic", "chink", "fag", "dyke", "wetback",
              "raghead", "gook", "beaner", "coon", "cracker"],
    "exclusion": ["ban", "exclude", "remove", "kick out", "get out", "go back",
                  "not welcome", "don't belong", "deport"],
    "dehumanizing": ["animal", "beast", "vermin", "pest", "cockroach", "rat",
                     "parasite", "disease", "plague", "infestation", "breed"],
    "inciting_violence": ["kill", "shoot", "hang", "burn", "attack", "destroy",
                         "beat", "punch", "stab", "bomb", "execute", "genocide"],
}


def match_target_group(text: str) -> str:
    """Match target_group from text using keyword matching. Returns None if no match."""
    text_lower = text.lower()
    scores = {}
    for group, keywords in TARGET_GROUP_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[group] = count
    if scores:
        return max(scores, key=scores.get)
    return None


def match_attack_type(text: str) -> str:
    """Match attack_type from text using keyword matching. Returns None if no match."""
    text_lower = text.lower()
    scores = {}
    for atype, keywords in ATTACK_TYPE_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[atype] = count
    if scores:
        return max(scores, key=scores.get)
    return None


def assign_labels(text: str, existing_target_group=None, existing_attack_type=None):
    """
    Assign target_group and attack_type, using existing labels if available,
    falling back to keyword matching.
    """
    tg = existing_target_group if existing_target_group else match_target_group(text)
    at = existing_attack_type if existing_attack_type else match_attack_type(text)
    return tg, at
