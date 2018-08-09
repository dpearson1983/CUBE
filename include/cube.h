#ifndef _CUBE_H_
#define _CUBE_H_

#include <vector>
#include <vector_types.h>
#include <fftw3.h>

int3 getSmallGridDimensions(double k_min, double k_max, double3 delta_k);

void getSmallCube(std::vector<double3> &cube, int3 N_grid, fftw_complex *dk, int3 N, 
                  double k_min, double k_max, std::vector<double> &kx, std::vector<double> &ky, 
                  std::vector<double> &kz, double3 delta_k, std::vector<int4> &kvec);

#endif
