"""Acoustic perturbation of synthesised sessions (same operations as make_schedule.py --perturb).

Every source clip gets a random band limitation (2nd-order Butterworth band-pass, high-pass 50-300 Hz,
low-pass 3.4-7.9 kHz), every unit (clip or chunk) a gain jitter of +-3 dB, and the whole session a continuous
noise floor (low-passed and white noise mixture) 25-40 dB below the RMS of its speech, so that digital silence
and abrupt channel changes no longer mark a clip boundary. A separate numpy generator is used, so that the
selection, order, gaps and gains of the sessions are identical with and without perturbation.
"""
import numpy as np
from scipy.signal import butter, sosfilt

SR = 16000


def band(x, rng):
    hp, lp = rng.uniform(50, 300), rng.uniform(3400, 7900)
    return sosfilt(butter(2, [hp, lp], btype="band", fs=SR, output="sos"), x).astype(np.float32)


def jitter(rng):
    return float(10 ** (rng.uniform(-3, 3) / 20))


def noise_floor(wav, rng):
    w = rng.standard_normal(len(wav) + 4096).astype(np.float32)
    x = 0.6 * sosfilt(butter(1, 0.02, btype="low", output="sos"), w) + 0.08 * w
    x = (x[4096:] / (np.std(x[4096:]) + 1e-8)).astype(np.float32)
    act = np.abs(wav) > 1e-4
    rms = np.sqrt(np.mean(wav[act] ** 2) + 1e-12) if act.any() else 0.1
    return (wav + x * rms * 10 ** (-rng.uniform(25, 40) / 20)).astype(np.float32)
