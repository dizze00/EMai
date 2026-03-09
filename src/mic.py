import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

def record(seconds=3, filename="mic.wav"):
    sr = 16000
    print("Recording...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    # 16-bit PCM (whisper.cpp expects 16 kHz mono 16-bit WAV)
    wav.write(filename, sr, (audio * 32767).clip(-32768, 32767).astype(np.int16))
    print("Saved:", filename)
    return filename