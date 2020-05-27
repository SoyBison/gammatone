import numpy as np
import pyopencl.array as cla
import pyopencl as cl
import pyopencl.clmath as clm

DEFAULT_FILTER_NUM = 100
DEFAULT_LOW_FREQ = 100
DEFAULT_HIGH_FREQ = 44100 // 4

EARQ = 9.26449
MIN_BW = 24.7
CORR = EARQ * MIN_BW
ORDER = 1

ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)


def erb_space(low_freq=DEFAULT_LOW_FREQ, high_freq=DEFAULT_HIGH_FREQ, num=DEFAULT_FILTER_NUM, __test=False, q=q):
    """
    Calculates a single point on an ERB scale in the defined frequency range, at the level defined.

    :param num: Defines the r
    :param low_freq: Defines the lower bound of the frequency range.
    :param high_freq: Defines the upper bound of the frequency range.
    :return: The ERB space between ``low_freq`` and ``high_freq`` at resolution defined by ``num``. If ``__test`` is
    ```False```, returns an opencl array, ```True``` returns a numpy array.
    """
    fracs = cla.arange(q, 1, num+1, dtype=np.float32) / num

    erb = - CORR + clm.exp(fracs * (np.log(low_freq + CORR) - np.log(high_freq + CORR))) * (high_freq + CORR)
    return erb


def centre_freqs(fs, num_freqs, cutoff, q=q):
    """
    Calculates the center frequencies from a sampling frequency, low end, and the number of filters you need.
    A wrapper for :func: `erg_space`
    :param fs: ``fs`` / 2 is passed to the high end of :func: `erg_space`
    :param num_freqs: is passed to the :param: ``num`` of :func: `erg_space`
    :param cutoff: is passed to the low end of :func: `erg_space`
    :return: The erb space between ``cutoff`` and ``fs / 2`` with resolution ``num_freqs``
    """
    return erb_space(cutoff, fs // 2, num_freqs, q=q)


def make_erb_filters(fs, centre_fs, width=1.0):
    t = 1 / fs

    erb = width * ((centre_fs / EARQ) ** ORDER + MIN_BW ** ORDER) ** (1 / ORDER)

    b = 1.019 * 2 * np.pi * erb
    arg = 2 * centre_fs * np.pi * t
    vec = clm.exp(np.complex(0, 2) * arg)
    btex = clm.exp(b * t)

    a0 = t
    a2 = 0
    b0 = 1
    b1 = -2 * clm.cos(arg) / btex
    b2 = clm.exp(-2 * b * t)

    rp = np.sqrt(3 + 2 ** 1.5)
    rn = np.sqrt(3 - 2 ** 1.5)

    common = -t / btex

    cosarg = clm.cos(arg)
    sinarg = clm.sin(arg)

    k1 = cosarg + rp * sinarg
    k2 = cosarg - rp * sinarg
    k3 = cosarg + rn * sinarg
    k4 = cosarg - rn * sinarg

    a11 = common * k1
    a12 = common * k2
    a13 = common * k3
    a14 = common * k4

    gain_arg = clm.exp(np.complex(0, 1) * arg - b * t)

    gain = (vec - gain_arg * k1) * \
           (vec - gain_arg * k2) * \
           (vec - gain_arg * k3) * \
           (vec - gain_arg * k4) * \
           (t * btex / (-1 / btex + 1 + vec * (1 - btex))) ** 4

    allfilts = cla.zeros_like(centre_fs) + 1

    fcoefs = [a0 * allfilts, a11, a12, a13, a14, a2 * allfilts, b0 * allfilts, b1, b2, np.abs(gain.get())]

    return np.column_stack([x.get() if type(x) == cla.Array else x for x in fcoefs])
