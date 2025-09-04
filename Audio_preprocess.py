def extract_loudest_call_energy(signal: np.ndarray, sr: int,
                                frame_ms=200.0, hop_ms=100.0,
                                ratio=0.10, min_duration=0.1):
    """
    clip the '1s of maximum energy window'。
    """
    N = len(signal)
    win_len = int(round(1.0 * sr))
    if N == 0 or win_len <= 0:
        return signal
    if N <= win_len:
        return signal
    power = signal.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(power)))
    # energies[i] = sum(power[i : i+win_len])
    energies = csum[win_len:] - csum[:-win_len]

    start = int(np.argmax(energies))
    end = start + win_len
    return signal[start:end]


def to_fixed_timesteps(clip: np.ndarray, target: int = 1000,
                       pad_mode: str = "constant") -> np.ndarray:
    """resampling to the target sample"""
    N = len(clip)
    if N == target:
        return clip.astype(np.float32)

    if N > target:
        g = math.gcd(N, target)
        up   = target // g
        down = N // g
        clip = resample_poly(clip, up, down)
        if len(clip) != target:                      
            clip = clip[:target] if len(clip) > target \
                   else np.pad(clip, (0, target-len(clip)))
    else:
        clip = np.pad(clip, (0, target - N), mode=pad_mode)

    return clip.astype(np.float32)

def normalize_robust(x, target_peak=0.05, peak_pct=99.5):
    ref = np.percentile(np.abs(x), peak_pct)  # 99.5% amplitude
    scale = target_peak / (ref + 1e-8)
    y = x * scale
    return np.clip(y, -target_peak, target_peak)

def append_silence_with_fade(x, n_silence=600, fade=32):
    tail = np.zeros(n_silence, dtype=x.dtype)
    if fade > 0:
        ramp = np.linspace(1.0, 0.0, fade)
        tail[:fade] = x[-1] * ramp
    return np.concatenate([x, tail])

def remove_dc(x):
    return x - np.mean(x)
