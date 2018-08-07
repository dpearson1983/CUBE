#ifndef _POWER_H_
#define _POWER_H_

#include <vector>
#include <cmath>

void binFrequencies(fftw_complex *delta, std::vector<double> &P, std::vector<int> &N_k,
                    vec3<int> N, std::vector<double> &kx, std::vector<double> &ky, 
                    std::vector<double> &kz, double delta_k, double k_min, double k_max, 
                    double SN);

void normalizePower(std::vector<double> &P, std::vector<int> &N_k, double nbsqwsq);

#endif
