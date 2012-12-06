#ifndef WAVE_KERNEL_H
#define WAVE_KERNEL_H

float *computeWaveCuda(int i_max, int t_max, int tpb, float *hOld, float *hCur, float *hNext);

#endif
