import gammatone.filters as oldfilt
import gammatone.filters as newfilt
import pyopencl.array as cla
import numpy as np
import time


def test_erb_space():
    hi = 11025
    lo = 100
    t0 = time.time()
    old_space = oldfilt.erb_space(lo, hi, num=100)
    t1 = time.time()
    new_space = newfilt.erb_space(lo, hi, num=100, __test=True)
    t2 = time.time()
    print(f'Old method took {t1 - t0} seconds. New method took {t2 - t1} seconds.')
    assert np.allclose(new_space.get(), old_space)


def test_make_erb_filters():
    hi = 100
    lo = 11025

    oldcf = oldfilt.centre_freqs(44100, 100, 20)
    newcf = newfilt.centre_freqs(44100, 100, 20)

    t0 = time.time()
    old = oldfilt.make_erb_filters(44100, oldcf, width=1.0)
    t1 = time.time()
    new = newfilt.make_erb_filters(44100, newcf, width=1.0)
    t2 = time.time()

    print(f'Old method took {t1 - t0} seconds, New method took {t2 - t1} seconds.')
    assert np.allclose(old, new)
