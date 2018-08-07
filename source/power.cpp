#include <vector>
#include <cmath>
#include <fftw3.h>
#include "../include/tpods.h"
#include "../include/power.h"

void binFrequencies(fftw_complex *delta, std::vector<double> &P, std::vector<int> &N_k,
                    vec3<int> N, std::vector<double> &kx, std::vector<double> &ky, 
                    std::vector<double> &kz, double delta_k, double k_min, double k_max, 
                    double SN) {
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k <= N.z/2; ++k) {
                double k_mag = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                
                if (k_mag >= k_min && k_mag < k_max) {
                    int index = k + (N.z/2 + 1)*(j + N.y*i);
                    int bin = int((k_mag - k_min)/delta_k);
                    P[bin] += delta[index][0]*delta[index][0] + delta[index][1]*delta[index][1] - SN;
                    N_k[bin]++;
                }
            }
        }
    }
}

void normalizePower(std::vector<double> &P, std::vector<int> &N_k, double nbsqwsq) {
    for (int i = 0; i < P.size(); ++i) {
        if (N_k[i] != 0) P[i] /= nbsqwsq*N_k[i];
    }
}
                    
