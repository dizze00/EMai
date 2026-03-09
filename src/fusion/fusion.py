import json
from pathlib import Path

# Path relative to this file's folder (so it works no matter where you run from)
_rules_path = Path(__file__).resolve().parent / "rules.json"
with open(_rules_path, "r", encoding="utf8") as f:
    RULES = json.load(f)

def fuse(text, emotion):
    text_lower = text.lower()

    for rule in RULES:
        if emotion != rule["emotion"]:
            continue

        for phrase in rule["contains"]:
            if phrase in text_lower:
                return {
                    "text": text,
                    "emotion": emotion,
                    "meaning": rule["meaning"]
                }

    return {
        "text": text,
        "emotion": emotion,
        "meaning": "Neutral interpretation"
    }