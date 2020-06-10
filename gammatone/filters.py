import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cla
import pyopencl.clmath as clm
from pyopencl.elementwise import ElementwiseKernel
import numpy as np
import math
import time
import soundfile
from matplotlib import pyplot as plt
from tqdm import tqdm
from pyopencl.scan import GenericScanKernel

ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)
M_PI = 3.14159265358979323846
BW_CORRECTION = 1.0190
EAR_CORRECTION = 4.37e-3
GM_PARAMETER = 24.7


q_maker = ElementwiseKernel(ctx,
                            "double *qcos, double *qsin, double *t, double tpt, double cf",
                            "qcos[i] = cos(tpt * cf * t[i]); qsin[i] = - sin(tpt * cf * t[i])"
                            )

p0 = GenericScanKernel(ctx, np.float64,
                       arguments='double *x, double *y, double d',
                       neutral='0',
                       input_expr='x[i]',
                       scan_expr='a*(-.8)*d + b',
                       output_statement="y[i] = item"
                       )


u0 = ElementwiseKernel(ctx,
                       'double *p0, double *p1, double *p2, double *y, double d',
                       'y[i] = p0[i] + d*p1[i] + d*d*p2[i]')

bm_maker = ElementwiseKernel(ctx,
                             'double *ur, double *ui, double *y, double g',
                             'y[i] = sqrt(ur[i] * ur[i] + ui[i] * ui[i]) * g')

shift = ElementwiseKernel(ctx,
                          "double *x",
                          "x[i] = x[i+1]")


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def erb(x):
    return GM_PARAMETER * (EAR_CORRECTION * x + 1.0)


def get_coefficients(signal, fs, cfs):
    samp_g = cla.to_device(q, signal)
    tpt = np.float64((M_PI + M_PI) / fs)
    ts_g = cla.arange(q, 0, len(signal), dtype=np.float64)
    coefficients = []
    for cf in cfs:
        # Calculating the parameters for the given center frequencies
        tptbw = tpt * erb(cf) * BW_CORRECTION
        decay = np.exp(-tptbw)
        gain = np.float64(tptbw)
        # Setting up memory for everything.
        qcos = cla.empty_like(ts_g)
        qsin = cla.empty_like(ts_g)
        bm_g = cla.empty_like(ts_g)
        p0r_g = cla.empty_like(ts_g)
        p0i_g = cla.empty_like(ts_g)
        ur_g = cla.empty_like(ts_g)
        ui_g = cla.empty_like(ts_g)
        # Preparing the imaginary/real cyclical effect
        q_maker(qcos, qsin, ts_g, tpt, cf)
        cosx = samp_g * qcos
        sinx = samp_g * qsin
        # Performing the filtering operation
        p0(cosx, p0r_g, decay)
        p0(sinx, p0i_g, decay)
        # Preparing the memory to calculate basilar membrane displacement.
        p1r_g = p0r_g.copy(q)
        shift(p1r_g)
        p2r_g = p1r_g.copy(q)
        shift(p2r_g)
        p1i_g = p0i_g.copy(q)
        shift(p1i_g)
        p2i_g = p1i_g.copy(q)
        shift(p2i_g)
        # Calculating Basilar Membrane displacement
        u0(p0r_g, p1r_g, p2r_g, ur_g, decay)
        u0(p0i_g, p1i_g, p2i_g, ui_g, decay)
        bm_maker(ur_g, ui_g, bm_g, gain)
        # Append to the list
        cl_x = bm_g.get()
        # cl_x = normalize(cl_x)
        coefficients.append(cl_x)

    return np.row_stack(coefficients)


def erb_space(top, bottom, number):
    frac_space = np.arange(1, number + 1) / number
    points = -1 / EAR_CORRECTION + np.exp(frac_space * (np.log(bottom + 1 / EAR_CORRECTION) -
                                                        np.log(top + 1 / EAR_CORRECTION))
                                          ) * (top + 1 / EAR_CORRECTION)
    return points


def center_freqs(samplerate, number, bottom):
    return erb_space(samplerate / 2, bottom, number)
