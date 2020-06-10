import numpy as np

from .filters import center_freqs, get_coefficients


def cochleagram(signal, samplerate, time_resolution, freq_resolution, bottom_end):
    """
    Creates a matrix through time of the gammatone envelope response to an input signal.
    :param np.array signal: The input signal.
    :param int samplerate: The rate at which your signal was sampled.
    :param int time_resolution: The width (In seconds) of each time-segment
    :param int freq_resolution: The number of central frequencies to test the cochlear response of the signal against.
    :param int bottom_end: The lowest frequency to test for.
    :return:
    """
    colwidth = int(time_resolution * samplerate)

    gammatones = get_coefficients(signal, samplerate, center_freqs(samplerate, freq_resolution, bottom_end))

    ncols = gammatones.shape[1] // colwidth

    y = np.empty((freq_resolution, ncols))

    for col in range(ncols):
        seg = gammatones[:, col*colwidth + np.arange(colwidth)]
        y[:, col] = np.sqrt(seg.mean(1))

    return y
