import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)

with open('gtkernel.c', 'r') as f:
    gtkernel = f.read()

GTFILT = cl.Program(ctx, gtkernel).build()


def gammatone_filter(wave, fs, cf, queue=q, context=ctx):
    """
    A wrapper for the pyopencl function that's in ``gtkernel.c``. The cl interface is kinda a pain, so this is to make
    it act more like a python function, since it shouldn't ever need any more memory operations than what it has.
    :param np.Array wave: The input wave, should be mono, if you need to do stereo, do the channels seperately.
    :param int fs: sampling frequency in Hz, if you don't know, it's probably 44100, but it's the second output from
    `soundfile.open`
    :param int cf: center frequency of the filter (Hz), if you don't know, 5512.5 is a good guess.
    :param cl.CommandQueue queue: The command queue for pyopencl. Only necessary if you *need* to access the buffer
    afterwards, which you probably don't.
    :param cl.Context context: Similar rules to ``queue``.
    :return np.Array: a signal representing the basilar membrane displacement, which can be segmented, and those
    segments averaged within to make a cepstrogram.
    """
    func = GTFILT.gammatone
    func.set_scalar_arg_dtypes([None, None, np.int32, np.float64, np.int32, np.int32])
    wave_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=wave.nbytes)
    cl.enqueue_copy(queue, wave_g, wave)
    gt_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=wave.nbytes)
    gt = np.empty_like(wave)
    func(queue, wave.shape, None, wave_g, gt_g, fs, cf, len(wave))
    cl.enqueue_copy(queue, gt, gt_g)
    return gt


