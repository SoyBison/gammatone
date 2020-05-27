#define erb(x)              (24.7 * (4.37e-3 * (x) + 1.0));
#define VERY_SMALL_NUMBER   1e-200
#define BW_CORR             1.0190
#ifndef PI
#define PI                  3.14159265358979323846
#define myMax(x,y)     ( ( x ) > ( y ) ? ( x ) : ( y ) )
#define myMod(x,y)     ( ( x ) - ( y ) * floor ( ( x ) / ( y ) ) )

__kernel void gammatone(__global double *x, __global double *bm, int fs, double cf, int nsamples){
    /* setup */
    int gid = get_global_id(0);
    double p0r, p1r, p2r, p3r, p4r, p0i, p1i, p2i, p3i, p4i;
    double a1, a2, a3, a4, a5, u0r, u0i;
    double qcos, qsin, oldcs, coscf, sincf, oldphase, dp, dps;


    oldphase = 0.0;
    double tpt = (2 * PI) / fs;
    double tptbw = tpt * erb(cf) * BW_CORR;
    double a = exp(-tptbw);

    gain = (tptbw*tptbw*tptbw*tptbw) / 3;

    a1 = 4.0*a; a2 = -6.0 * a * a; a3 = 4.0*a*a*a; a4 = -a*a*a*a; a5 = a*a
    p0r = 0.0; p1r = 0.0; p2r = 0.0; p3r = 0.0; p4r = 0.0;
    p0i = 0.0; p1i = 0.0; p2i = 0.0; p3i = 0.0; p4i = 0.0;

    coscf = cos(tpt * cf);
    sincf = sin(tpt * cf);

    qcos = 1; qsin = 0;
    /* Filter part 1 and shift to DC */
    p0r = qcos*x[gid] + a1*p1r + a2*p2r + a3*p2r + a3*p3r + a4*p4r;
    p0i = qsin*x[gid] + a1*p1i + a2*p2i + a3*p2i + a3*p3i + a4*p4i;

    }