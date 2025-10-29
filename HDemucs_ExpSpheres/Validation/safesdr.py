import numpy as np
import warnings

def safe_db(num, den, inf_to_nan=False):
    """ Compute dB rations avoiding warnings.
    Use inf_to_nan to convert any +inf or -inf to nan so they're also excluding
    when using functions such as np.nanmean or np.nanmedian.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = 10 * np.log10(num / den)
        if inf_to_nan:
            output[np.isinf(output)] = np.nan
    return output


def windowed_energy(signal, sample_rate):
    signal = np.pad(signal[:, :], ((0, 0), (0, sample_rate - signal.shape[-1] % sample_rate)))
    signal = np.stack(np.split(signal, signal.shape[-1] // sample_rate, axis=-1), axis=1)
    return np.mean(signal ** 2, axis=-1)
    

def compute_track_sdr(estimates, references, sample_rate):
    distortion_e = windowed_energy(estimates - references, sample_rate)
    references_e = windowed_energy(references, sample_rate)

    silence_th_dB = 20  # Threshold from the instrument peak to consider a frame as silent
    references_e[references_e < references_e.max(1, keepdims=True) * 10**(-silence_th_dB/10)] = 0

    sdr_frames = safe_db(references_e, distortion_e, inf_to_nan=True)
    return np.nanmedian(sdr_frames, axis=-1)
