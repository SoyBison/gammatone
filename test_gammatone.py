import gammatone.filters as oldfilt
import gammatonecl.filters as newfilt
import numpy as np
import time


def test_erb_space():
    hi = 100
    lo = 11025
    t0 = time.time()
    old_space = oldfilt.erb_space(lo, hi, num=100)
    t1 = time.time()
    new_space = newfilt.erb_space(lo, hi, num=100, __test=True)
    t2 = time.time()
    print(f'Old method took {t1 - t0} seconds. New method took {t2 - t1} seconds.')
    assert np.allclose(new_space, old_space)
