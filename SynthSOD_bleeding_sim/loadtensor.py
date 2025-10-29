import torch
import pathlib
import soundfile as sf

from collections.abc import Callable


def load_into_tensor(func: Callable) -> Callable:
    '''
    Decorator that enables a function returning a NumPy array (e.g., from `soundfile.read`)
    to seamlessly work with a preallocated PyTorch tensor as output.

    If a PyTorch tensor is passed to `out=`, it is converted to a NumPy array internally,
    and the wrapped function writes directly into it. The modified tensor is returned.

    If no `out` is provided, the function's output is converted to a PyTorch tensor.

    Parameters
    ----------
    - func : Callable
        The function to wrap, typically one that loads audio into a NumPy array.

    Returns
    -------
    - Callable
        A wrapped function that returns a PyTorch tensor and optionally writes into
        a provided tensor in-place.
    '''
    def wrapper(*args,**kwargs):
        out_tensor = kwargs.get('out',None)
        if out_tensor is not None:
            kwargs['out'] = out_tensor.numpy()
            func(*args,**kwargs)
             
            return out_tensor
        else:
            wave = func(*args,**kwargs)
            
            return torch.from_numpy(wave)

    return wrapper


@load_into_tensor
def load_audio_chunk(
    filepath: pathlib.Path,
    start: int,
    stop: int,
    fill_value: float = 0,
    out: torch.Tensor = None,
    dtype: str = 'float32'):
    '''
    Loads a chunk of audio from a file between the given start and stop frames.

    Optionally writes the data into a provided PyTorch tensor for in-place loading.

    Parameters
    ----------
    - filepath : pathlib.Path
        Path to the audio file (e.g., WAV, FLAC).
    - start : int
        The starting frame index.
    - stop : int
        The stopping frame index (exclusive).
    - fill_value : float, optional
        Value used to pad if the requested segment goes beyond the file duration.
    - out : torch.Tensor, optional
        Optional preallocated tensor to write the audio into.
        Must be a 1D float tensor on CPU.
    - dtype : str, optional
        The NumPy dtype to use when reading the audio (default is 'float32').

    Returns
    -------
    - torch.Tensor
        A tensor containing the loaded audio, or the same tensor passed via `out`.
    '''
    wave,_ = sf.read(
        filepath,
        start=start,
        stop=stop,
        fill_value=fill_value,
        out=out,
        dtype=dtype)

    return wave