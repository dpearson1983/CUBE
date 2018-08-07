#include <sstream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "../include/tpods.h"

double sinc(double x) {
    if (x != 0) {
        return sin(x)/x;
    } else {
        return 1.0;
    }
}

double CICgridCorrection(double kx, double ky, double kz, vec3<int> N, vec3<double> L) {
    double prodSinc = 1.0/(sinc(0.5*kx*L.x/N.x)*sinc(0.5*ky*L.y/N.y)*sinc(0.5*kz*L.z/N.z));
    return prodSinc*prodSinc;
}

void CICbinningCorrection(fftw_complex *delta, vec3<int> N, vec3<double> L, std::vector<double> &kx,
                          std::vector<double> &ky, std::vector<double> &kz) {
#pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k <= N.z/2; ++k) {
                int index = k + (N.z/2 + 1)*(j + N.y*i);
                delta[index][0] *= CICgridCorrection(kx[i], ky[j], kz[k], N, L);
                delta[index][1] *= CICgridCorrection(kx[i], ky[j], kz[k], N, L);
            }
        }
    }
}

void getCICInfo(vec3<double> pos, const vec3<int> &N, const vec3<double> &L, 
                std::vector<size_t> &indices, std::vector<double> &weights) {
    vec3<double> del_r = {L.x/N.x, L.y/N.y, L.z/N.z};
    vec3<int> ngp = {int(pos.x/del_r.x), int(pos.y/del_r.y), int(pos.z/del_r.z)};
    vec3<double> r_ngp = {(ngp.x + 0.5)*del_r.x, (ngp.y + 0.5)*del_r.y, (ngp.z + 0.5)*del_r.z};
    vec3<double> dr = {pos.x - r_ngp.x, pos.y - r_ngp.y, pos.z - r_ngp.z};
    vec3<int> shift;
    if (dr.x != 0) shift.x = int(dr.x/fabs(dr.x));
    else shift.x = 0;
    if (dr.y != 0) shift.y = int(dr.y/fabs(dr.y));
    else shift.y = 0; 
    if (dr.z != 0) shift.z = int(dr.z/fabs(dr.z));
    else shift.z = 0;
    
    dr.x = fabs(dr.x);
    dr.y = fabs(dr.y);
    dr.z = fabs(dr.z);
    
    double dV = del_r.x*del_r.y*del_r.z;
    
    if (ngp.x + shift.x < -1 || ngp.x + shift.x > N.x || shift.x < -1 || shift.x > 1) {
        std::stringstream errMessage;
        errMessage << "x index seems funny: " << ngp.x << ", " << shift.x << "\n";
        errMessage << pos.x << ", " << r_ngp.x << ", " << dr.x;
        throw std::runtime_error(errMessage.str());
    }
    
    if (ngp.y + shift.y < -1 || ngp.y + shift.y > N.y || shift.y < -1 || shift.y > 1) {
        std::stringstream errMessage;
        errMessage << "y index seems funny: " << ngp.y << ", " << shift.y << "\n";
        errMessage << pos.y << ", " << r_ngp.y << ", " << dr.y;
        throw std::runtime_error(errMessage.str());
    }
    
    if (ngp.z + shift.z < -1 || ngp.z + shift.z > N.z || shift.z < -1 || shift.z > 1) {
        std::stringstream errMessage;
        errMessage << "z index seems funny: " << ngp.z << ", " << shift.z << "\n";
        errMessage << pos.z << ", " << r_ngp.z << ", " << dr.z;
        throw std::runtime_error(errMessage.str());
    }
    
    if (ngp.x + shift.x == -1) shift.x = N.x - 1;
    if (ngp.y + shift.y == -1) shift.y = N.y - 1;
    if (ngp.z + shift.z == -1) shift.z = N.z - 1;
    
    if (ngp.x + shift.x == N.x) shift.x = 1 - N.x;
    if (ngp.y + shift.y == N.y) shift.y = 1 - N.y;
    if (ngp.z + shift.z == N.z) shift.z = 1 - N.z;
    
    indices.push_back(size_t(ngp.z + N.z*(ngp.y + N.y*ngp.x)));
    indices.push_back(size_t(ngp.z + N.z*(ngp.y + N.y*(ngp.x + shift.x))));                 // Shift: x
    indices.push_back(size_t(ngp.z + N.z*((ngp.y + shift.y) + N.y*ngp.x)));                 // Shift: y
    indices.push_back(size_t((ngp.z + shift.z) + N.z*(ngp.y + N.y*ngp.x)));                 // Shift: z
    indices.push_back(size_t(ngp.z + N.z*((ngp.y + shift.y) + N.y*(ngp.x + shift.x))));     // Shift: x, y
    indices.push_back(size_t((ngp.z + shift.z) + N.z*(ngp.y + N.y*(ngp.x + shift.x))));     // Shift: x, z
    indices.push_back(size_t((ngp.z + shift.z) + N.z*((ngp.y + shift.y) + N.y*ngp.x)));     // Shift: y, z
    indices.push_back(size_t((ngp.z + shift.z) + N.z*((ngp.y + shift.y) + N.y*(ngp.x + shift.x)))); // Shift: x, y, z
    
    weights.push_back(((del_r.x - dr.x)*(del_r.y - dr.y)*(del_r.z - dr.z))/dV);
    weights.push_back((dr.x*(del_r.y - dr.y)*(del_r.z - dr.z))/dV);
    weights.push_back(((del_r.x - dr.x)*dr.y*(del_r.z - dr.z))/dV);
    weights.push_back(((del_r.x - dr.x)*(del_r.y - dr.y)*dr.z)/dV);
    weights.push_back((dr.x*dr.y*(del_r.z - dr.z))/dV);
    weights.push_back((dr.x*(del_r.y - dr.y)*dr.z)/dV);
    weights.push_back(((del_r.x - dr.x)*dr.y*dr.z)/dV);
    weights.push_back((dr.x*dr.y*dr.z)/dV);
}
