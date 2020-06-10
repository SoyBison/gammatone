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

q_maker = ElementwiseKernel(ctx,
                            "double *qcos, double *qsin, double *t, double tpt, double cf",
                            "qcos[i] = cos(tpt * cf * t[i]); qsin[i] = - sin(tpt * cf * t[i])"
                            )

p0 = GenericScanKernel(ctx, np.float64,
                       arguments='double *x, double *y, double d',
                       neutral='0',
                       input_expr='x[i]',
                       scan_expr='b*d',
                       output_statement="""y[i] = item - d*d*prev_item"""
                       )

p1 = GenericScanKernel(ctx, np.float64,
                       arguments='double *x, double d',
                       neutral='0',
                       input_expr='x[i]',
                       scan_expr='b*d*d*d',
                       output_statement="x[i]= item - d*d*d*d*prev_item")

u0 = ElementwiseKernel(ctx,
                       'double *p0, double *p1, double *p2, double *y, double d',
                       'y[i] = p0[i] + 4.0 * d * p1[i] + d*d*p2[i]')

bm_maker = ElementwiseKernel(ctx,
                             'double *ur, double *ui, double *qcos, double *qsin, double *y, double g',
                             'y[i] = (ur[i] * qcos[i] + ui[i] * qsin[i]) * g')

shift = ElementwiseKernel(ctx,
                          "double *x",
                          "x[i] = x[i+1]")


def erb(x):
    return 24.7 * (4.37e-3 * x + 1.0)


def get_coefficients(signal, fs, cfs):
    samp_g = cla.to_device(q, signal)
    tpt = (M_PI + M_PI) / fs
    ts_g = cla.arange(q, 0, len(signal))
    coefficients = []
    if type(cfs) != list:
        cfs = [cfs]
    for cf in cfs:
        # Calculating the parameters for the given center frequencies
        tptbw = tpt * erb ( cf ) * BW_CORRECTION
        decay = np.exp(-tptbw)
        gain = np.float64((tptbw ** 4) / 3)
        # Setting up memory for everything.
        qcos = cla.empty_like(ts_g)
        qsin = cla.empty_like(ts_g)
        bm_g = cla.empty_like(ts_g)
        p0r_g = cla.empty_like(ts_g)
        p0i_g = cla.empty_like(ts_g)
        ur_g = cla.empty_like(ts_g)
        ui_g = cla.empty_like(ts_g)
        # Preparing the imaginary/real cyclical effects
        q_maker(qcos, qsin, ts_g, tpt, cf)
        cosx = samp_g * qcos
        sinx = samp_g * qsin
        # Performing the filtering operation
        p0(cosx, p0r_g, decay)
        p0(sinx, p0i_g, decay)
        p1(p0r_g, decay)
        p1(p0i_g, decay)
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
        bm_maker(ur_g, ui_g, qcos, qsin, bm_g, gain)
        # Append to the list
        cl_x = (bm_g**2).get()
        coefficients.append(cl_x)

    return np.row_stack(coefficients)

