import librosa
import numpy as np

def extract_prosody(path):
    # Load audio
    y, sr = librosa.load(path, sr=16000)

    # FAST pitch estimation (YIN)
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )
    # Use only voiced frames (typical speech 80–400 Hz); ignore silence/unvoiced
    voiced = (f0 >= 80) & (f0 <= 400)
    f0_voiced = f0[voiced] if np.any(voiced) else f0
    pitch_mean = float(np.mean(f0_voiced))
    pitch_std = float(np.std(f0_voiced))

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    # Zero-crossing rate (noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "centroid_mean": centroid_mean,
        "zcr_mean": zcr_mean,
    }