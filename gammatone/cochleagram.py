import numpy as np

from .filters import center_freqs, get_coefficients


def cochleagram(signal, samplerate, time_resolution, freq_resolution, bottom_end):
    colwidth = int(time_resolution * samplerate)

    gammatones = get_coefficients(signal, samplerate, center_freqs(samplerate, freq_resolution, bottom_end))

    ncols = gammatones.shape[1] // colwidth

    y = np.empty((freq_resolution, ncols))

    for col in range(ncols):
        seg = gammatones[:, col*colwidth + np.arange(colwidth)]
        y[:, col] = np.sqrt(seg.mean(1))

    return y
