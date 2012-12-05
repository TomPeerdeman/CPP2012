#ifndef WAVE_KERNEL_H
#define WAVE_KERNEL_H

double *computeWaveCuda(int i_max, int t_max, int tpb, double *hOld, double *hCur, double *hNext);

#endif
