#ifndef RKDP8_CKS_CU
#define RKDP8_CKS_CU

#include "../geodesic_base.cu"
#include "../cks_metric.cu"
#include "../disk.cu"

extern "C" __global__
void trace_rkdp8_cks(const RenderParams *pp, unsigned char *output, const float *skymap) {
    const RenderParams &p = *pp;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int)p.width, H = (int)p.height;
    if (ix >= W || iy >= H) return;

    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    if (!initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta)) return;

    int idx = (iy * W + ix) * 3;
    output[idx + 0] = (unsigned char)255;
    output[idx + 1] = (unsigned char)0;
    output[idx + 2] = (unsigned char)0;
}

extern "C" __global__
void ray_trace_rkdp8_cks(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
}

#endif
