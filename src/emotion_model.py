import numpy as np

def classify_emotion(features):
    pitch = features["pitch_mean"]
    pitch_var = features["pitch_std"]
    energy = features["energy_mean"]
    energy_var = features["energy_std"]
    centroid = features["centroid_mean"]
    zcr = features["zcr_mean"]

    # Score each emotion; pick the strongest (avoids "first match wins")
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    scores = {}

    # Angry: high energy, high noisiness (zcr), often higher pitch variance
    angry = (clamp(energy, 0, 0.25) / 0.2) * 0.5 + (clamp(zcr, 0, 0.15) / 0.12) * 0.5
    scores["angry"] = angry if energy > 0.12 and zcr > 0.06 else 0

    # Sad: low energy, low pitch, low variance (flat)
    sad = (1 - clamp(energy, 0.04, 0.15) / 0.15) * 0.5 + (1 - clamp(pitch, 80, 180) / 180) * 0.3 + (1 - clamp(pitch_var, 0, 25) / 25) * 0.2
    scores["sad"] = sad if energy < 0.12 and pitch < 200 else 0

    # Happy: higher pitch, decent energy, higher variance
    happy = (clamp(pitch, 160, 320) / 320) * 0.4 + (clamp(energy, 0.06, 0.2) / 0.2) * 0.3 + (clamp(pitch_var, 15, 50) / 50) * 0.3
    scores["happy"] = happy if pitch > 170 and energy > 0.08 else 0

    # Excited: high energy + high pitch variance (animated)
    excited = (clamp(energy, 0.1, 0.25) / 0.2) * 0.5 + (clamp(pitch_var, 15, 60) / 50) * 0.5
    scores["excited"] = excited if energy > 0.12 and pitch_var > 18 else 0

    # Calm: low energy, low variance, mid pitch
    calm = (1 - clamp(energy, 0.04, 0.14) / 0.14) * 0.5 + (1 - clamp(pitch_var, 0, 20) / 20) * 0.5
    scores["calm"] = calm if energy < 0.12 and pitch_var < 18 else 0

    best = max(scores, key=scores.get)
    return best if scores[best] >= 0.35 else "neutral"


if __name__ == "__main__":
    # Example usage
    example = {
        "pitch_mean": 180,
        "pitch_std": 25,
        "energy_mean": 0.12,
        "energy_std": 0.03,
        "centroid_mean": 2500,
        "zcr_mean": 0.07,
    }
    print(classify_emotion(example))