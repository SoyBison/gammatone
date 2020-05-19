import numpy as np
import pyopencl.array as cla
import pyopencl as cl

DEFAULT_FILTER_NUM = 100
DEFAULT_LOW_FREQ = 100
DEFAULT_HIGH_FREQ = 44100 // 4

ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)

erbprg = cl.Program(ctx, """
    __kernel void erb(__global double *x, int hi, int lo, const float corr)
    {
        int gid = get_global_id(0);
        x[gid] = - corr + exp(x[gid] * (log(lo + corr) - log(hi + corr))) * (hi + corr);
    }
    """).build()


def erb_space(low_freq=DEFAULT_LOW_FREQ, high_freq=DEFAULT_HIGH_FREQ, num=DEFAULT_FILTER_NUM, __test=False):
    """
    Calculates a single point on an ERB scale in the defined frequency range, at the level defined.

    :param num: Defines the r
    :param low_freq: Defines the lower bound of the frequency range.
    :param high_freq: Defines the upper bound of the frequency range.
    :return: The ERB space between ``low_freq`` and ``high_freq`` at resolution defined by ``num``. If ``__test`` is
    ```False```, returns an opencl array, ```True``` returns a numpy array.
    """
    ear_q = 9.26449
    min_bw = 24.7
    corr = ear_q * min_bw

    fracs = np.arange(1, num + 1) / num

    # Memory operations

    frac_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=fracs.nbytes)
    erb = erbprg.erb
    erb.set_scalar_arg_dtypes([None, np.int32, np.int32, np.float32])
    cl.enqueue_copy(q, frac_g, fracs)

    # Run

    erb(q, fracs.shape, None, frac_g, high_freq, low_freq, corr)

    if __test:
        frac_r = np.empty_like(fracs)
        cl.enqueue_copy(q, frac_r, frac_g)
        return frac_r

    return frac_g


def centre_freqs(fs, num_freqs, cutoff):
    """
    Calculates the center frequencies from a sampling frequency, low end, and the number of filters you need.
    A wrapper for :func: `erg_space`
    :param fs: ``fs`` / 2 is passed to the high end of :func: `erg_space`
    :param num_freqs: is passed to the :param: ``num`` of :func: `erg_space`
    :param cutoff: is passed to the low end of :func: `erg_space`
    :return: The erb space between ``cutoff`` and ``fs / 2`` with resolution ``num_freqs``
    """
    return erb_space(cutoff, fs / 2, num_freqs)



