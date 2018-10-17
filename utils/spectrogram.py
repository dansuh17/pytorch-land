import librosa
import numpy as np
import scipy


def split_spectrogram(spec, chunk_size_in_frames: int):
    """
    Splits spectrogram into chunks that have sizes of
    'chunk_size' in the time axis.

    Args:
        spec (np.ndarray): spectrogram represented as 2D array
        chunk_size_in_frames (int): number of frames to split into

    Returns:
        split_spec = list of np.ndarray's representing chunks having ``chunk_size`` sizes
    """
    time_length = spec.shape[1]
    num_chunks = time_length // chunk_size_in_frames
    spec = spec[:, :num_chunks * chunk_size_in_frames]
    return np.hsplit(spec, num_chunks)


def denormalize_db_spectrogram(spec, top_db=80.0, mean_db=50.0):
    """
    Denormalizes normalized spectrum into db-spectrum.

    Args:
        spec: normalized spectrum
        top_db (float): maximum db level
        mean_db (float): mean db level

    Returns:
        denormalized db spectrum
    """
    return spec * top_db + mean_db


def normalize_db_spectrogram(db_spec, top_db=80.0):
    """
    Normalizes db spectrum with zero mean and unit variance.
    In normal cases, librosa generates db spectrum having 80dB as maximum.
    Thus, specifying the `top_db` will normalize assuming the
    spectrum lies within such range.

    Args:
        db_spec: db spectrum
        top_db (float): maximum decibel range

    Returns:
        normalized db spectrum
    """
    mean = np.mean(db_spec)
    return (db_spec - mean) / top_db


def recover_spectrogram(power_spec, original_phase):
    """
    Recover spectrogram from power spectrogram.
    In order to sufficiently recover the original spectrogram,
    phase information is required.

    Args:
        power_spec (np.ndarray): power spectrogram
        original_phase (np.ndarray): phase information
            - usually extracted from original spectorgram

    Returns:
        recovered_spec (np.ndarray): recovered spectrogram
    """
    assert power_spec.shape == original_phase.shape
    # take the abs since power_spec may have negative values
    abs_power_spec = np.abs(power_spec)
    return np.multiply(np.sqrt(abs_power_spec), np.exp(1j * original_phase))


def recover_audio_from_mel_spectrogram(mel_spec, original_spec,
                                       n_fft: int, fs: int, n_mels: int):
    """
    Recover and create audio from mel-spectrogram.
    In order to sufficiently recover the audio, original spectrogram as well
    as some information about the original audio is required.

    Args:
        mel_spec (np.ndarray): mel-spectrogram
        original_spec (np.ndarray): original spectrogram
        n_fft (int): number of FFT bins
        fs (int): sampling rate of original audio
        n_mels (int): number of mel-spectrogram bins

    Returns:
        ys (np.ndarray): recovered audio signal
    """
    mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels)
    # power spectrum recovered
    recovered_power_spec = np.dot(np.matrix(mel_basis).I, mel_spec)
    recovered_spec = recover_spectrogram(recovered_power_spec, np.angle(original_spec))
    # return reconstructed audio
    _, ys = scipy.signal.istft(recovered_spec, fs=fs, n_fft=n_fft)
    return ys
