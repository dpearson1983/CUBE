#include <vector>
#include <cmath>
#include <vector_types.h>
#include <fftw3.h>
#include <omp.h>
#include "../include/cube.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

// While it should be unnecessary, doing the division, adding 0.5 and then rounding down ensures
// that this function returns the correct result. Without it, there seems to be some truncation
// error that creeps in as the frequencies are multiples of PI (even when PI is rounded to the 
// precision of a double, the problem still occurs). Occasionally, k/delta_k does not evaluate to
// an exact integer which, without the additional steps, would cause the index returned to be one
// off from the correct value.
int kMatch(double k, std::vector<double> &kb, double delta_k) {
    int index = int(floor(k/delta_k + 0.5));
    if (index < 0) index += kb.size();
    return index;
}

int3 getSmallGridDimensions(double k_min, double k_max, double3 delta_k) {
    int3 N_grid;
    N_grid.x = 2*k_max/delta_k.x + 1 - (int(2*k_max/delta_k.x) % 2);
    N_grid.y = 2*k_max/delta_k.y + 1 - (int(2*k_max/delta_k.y) % 2);
    N_grid.z = 2*k_max/delta_k.z + 1 - (int(2*k_max/delta_k.z) % 2);
    return N_grid;
}

void getSmallCube(std::vector<double3> &cube, int3 N_grid, fftw_complex *dk, int3 N, 
                  double k_min, double k_max, std::vector<double> &kx, std::vector<double> &ky, 
                  std::vector<double> &kz, double3 delta_k, std::vector<int4> &kvec) {
    if (cube.size() != N_grid.x*N_grid.y*N_grid.z) {
        cube.resize(N_grid.x*N_grid.y*N_grid.z);
    }
    
    for (int i = 0; i < N_grid.x; ++i) {
        double k_x = (i - (N_grid.x/2 - 1))*delta_k.x;
        int i2 = kMatch(k_x, kx, delta_k.x);
        for (int j = 0; j < N_grid.y; ++j) {
            double k_y = (j - (N_grid.y/2 - 1))*delta_k.y;
            int j2 = kMatch(k_y, ky, delta_k.y);
            for (int k = 0; k < N_grid.z; ++k) {
                double k_z = (k - (N_grid.z/2 - 1))*delta_k.z;
                int k2 = kMatch(k_z, kz, delta_k.z);
                int index1 = k + N_grid.z*(j + N_grid.y*i);
                double k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                
                if (k2 > N.z/2) {
                    i2 = kMatch(-k_x, kx, delta_k.x);
                    j2 = kMatch(-k_y, ky, delta_k.y);
                    k2 = kMatch(-k_z, kz, delta_k.z);
                    int index2 = k2 + (N.z/2 + 1)*(j2 + N.y*i2);
                    cube[index1].x = dk[index2][0];
                    cube[index1].y = -dk[index2][1];
                    cube[index1].z = k_mag;
                } else {
                    int index2 = k2 + (N.z/2 + 1)*(j2 + N.y*i2);
                    cube[index1].x = dk[index2][0];
                    cube[index1].y = dk[index2][1];
                    cube[index1].z = k_mag;
                }
                
                if (k_mag >= k_min && k_mag < k_max) {
                    int4 temp = {i - N_grid.x/2, j - N_grid.y/2, k - N_grid.z/2, index1};
                    kvec.push_back(temp);
                }
            }
        }
    }
}

void getSmallCube(std::vector<double3> &cube, int3 N_grid, fftw_complex *dk, int3 N, 
                  double k_min, double k_max, std::vector<double> &kx, std::vector<double> &ky, 
                  std::vector<double> &kz, double3 delta_k) {
    if (cube.size() != N_grid.x*N_grid.y*N_grid.z) {
        cube.resize(N_grid.x*N_grid.y*N_grid.z);
    }
    
    for (int i = 0; i < N_grid.x; ++i) {
        double k_x = (i - (N_grid.x/2 - 1))*delta_k.x;
        int i2 = kMatch(k_x, kx, delta_k.x);
        for (int j = 0; j < N_grid.y; ++j) {
            double k_y = (j - (N_grid.y/2 - 1))*delta_k.y;
            int j2 = kMatch(k_y, ky, delta_k.y);
            for (int k = 0; k < N_grid.z; ++k) {
                double k_z = (k - (N_grid.z/2 - 1))*delta_k.z;
                int k2 = kMatch(k_z, kz, delta_k.z);
                int index1 = k + N_grid.z*(j + N_grid.y*i);
                double k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                
                if (k2 > N.z/2) {
                    i2 = kMatch(-k_x, kx, delta_k.x);
                    j2 = kMatch(-k_y, ky, delta_k.y);
                    k2 = kMatch(-k_z, kz, delta_k.z);
                    int index2 = k2 + (N.z/2 + 1)*(j2 + N.y*i2);
                    cube[index1].x = dk[index2][0];
                    cube[index1].y = -dk[index2][1];
                    cube[index1].z = k_mag;
                } else {
                    int index2 = k2 + (N.z/2 + 1)*(j2 + N.y*i2);
                    cube[index1].x = dk[index2][0];
                    cube[index1].y = dk[index2][1];
                    cube[index1].z = k_mag;
                }
            }
        }
    }
}
