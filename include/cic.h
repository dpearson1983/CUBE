#ifndef _CIC_H_
#define _CIC_H_

#include <fftw3.h>
#include <vector_types.h>

void CICbinningCorrection(fftw_complex *delta, int3 N, double3 L, std::vector<double> &kx,
                          std::vector<double> &ky, std::vector<double> &kz);

void getCICInfo(double3 pos, const int3 &N, const double3 &L, 
                std::vector<size_t> &indices, std::vector<double> &weights);

#endif
