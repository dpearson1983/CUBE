#ifndef _CIC_H_
#define _CIC_H_

#include <fftw3.h>
#include "tpods.h"

void CICbinningCorrection(fftw_complex *delta, vec3<int> N, vec3<double> L, std::vector<double> &kx,
                          std::vector<double> &ky, std::vector<double> &kz);

void getCICInfo(vec3<double> pos, const vec3<int> &N, const vec3<double> &L, 
                std::vector<size_t> &indices, std::vector<double> &weights);

#endif
