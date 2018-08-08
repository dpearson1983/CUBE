#ifndef _CUBE_H_
#define _CUBE_H_

#include <vector>
#include <vector_types.h>
#include <fftw3.h>

int4 getSmallGridDimensions(double k_min, double k_max, double3 delta_k);

void getSmallCube(std::vector<double3> &cube, fftw_complex *dk, int4 N_grid, std::vector<double> &kx,
                  std::vector<double> &ky, std::vector<double> &kz, int3 N);

#endif
