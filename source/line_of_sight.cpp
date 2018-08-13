#include <vector>
#include <fftw3.h>
#include <omp.h>
#include <vector_types.h>
#include "../include/transformers.h"
#include "../include/line_of_sight.h"

const std::string wisdom_file = "fftWisdom.dat";

void get_A0(std::vector<double> &dr, std::vector<double> &A_0, int3 N) {
    if (A_0.size() != N.x*N.y*2*(N.z/2 + 1)) {
        A_0.resize(N.x*N.y*2*(N.z/2 + 1));
    }
    generate_wisdom_fipr2c(A_0, N, wisdom_file, omp_get_max_threads()); // TODO: Add this function to transformers
    
    #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k < N.z; ++k) {
                int index1 = k + N.z*(j + N.y*i);
                int index2 = k + 2*(N.z/2 + 1)*(j + N.y*i);
                
                A_0[index2] = dr[index1];
            }
        }
    }
    
    fip_r2c(A_0, N, wisdom_file, omp_get_max_threads());
}

void sum_Bs(fftw_complex *A_2, fftw_complex *Bxx, fftw_complex *Byy, fftw_complex *Bzz, 
            fftw_complex *Bxy, fftw_complex *Bxz, fftw_complex *Byz, int3 N, double3 L) {
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
    
    #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k <= N.z/2; ++k) {
                int index = k + (N.z/2 + 1)*(j + N.y*i);
                double k_magsq = kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k];
                
                A_2[index][0] = (kx[i]*kx[i]*Bxx[index][0] + ky[j]*ky[j]*Byy[index][0] + 
                                 kz[k]*kz[k]*Bzz[index][0] + 2.0*kx[i]*ky[j]*Bxy[index][0] +
                                 2.0*kx[i]*kz[k]*Bxz[index][0] + 2.0*ky[j]*kz[k]*Byz[index][0])/k_magsq;
                A_2[index][1] = (kx[i]*kx[i]*Bxx[index][1] + ky[j]*ky[j]*Byy[index][1] + 
                                 kz[k]*kz[k]*Bzz[index][1] + 2.0*kx[i]*ky[j]*Bxy[index][1] +
                                 2.0*kx[i]*kz[k]*Bxz[index][1] + 2.0*ky[j]*kz[k]*Byz[index][1])/k_magsq;
            }
        }
    }
}

void get_A2(std::vector<double> &dr, std::vector<double> &A_2, int3 N, double3 L, double3 r_min) {
    if (A_2.size() != N.x*N.y*2*(N.z/2 + 1)) {
        A_2.resize(N.x*N.y*2*(N.z/2 + 1));
    }
    std::vector<double> Bxx(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Byy(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Bzz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Bxy(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Bxz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Byz(N.x*N.y*2*(N.z/2 + 1));
    generate_wisdom_fipr2c(Bxx, N, wisdom_file, omp_get_max_threads());
    
    double3 del_r = {L.x/N.x, L.y/N.y, L.z/N.z};
    
    #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        double r_x = r_min.x + (i + 0.5)*del_r.x;
        for (int j = 0; j < N.y; ++j) {
            double r_y = r_min.y + (j + 0.5)*del_r.y;
            for (int k = 0; k < N.z; ++k) {
                double r_z = r_min.z + (k + 0.5)*del_r.z;
                double r_magsq = r_x*r_x + r_y*r_y+ r_z*r_z;
                int index1 = k + N.z*(j + N.y*i);
                int index2 = k + 2*(N.z/2 + 1)*(j + N.y*i);
                
                Bxx[index2] = (r_x*r_x*dr[index1])/r_magsq;
                Byy[index2] = (r_y*r_y*dr[index1])/r_magsq;
                Bzz[index2] = (r_z*r_z*dr[index1])/r_magsq;
                Bxy[index2] = (r_x*r_y*dr[index1])/r_magsq;
                Bxz[index2] = (r_x*r_z*dr[index1])/r_magsq;
                Byz[index2] = (r_y*r_z*dr[index1])/r_magsq;
            }
        }
    }
    
    fip_r2c(Bxx, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Byy, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Bzz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Bxy, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Bxz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Byz, N, wisdom_file, omp_get_max_threads());
    
    sum_Bs((fftw_complex *)A_2.data(), (fftw_complex *)Bxx.data(), (fftw_complex *)Byy.data(),
           (fftw_complex *)Bzz.data(), (fftw_complex *)Bxy.data(), (fftw_complex *)Bxz.data(),
           (fftw_complex *)Byz.data(), N, L);
}

void sum_Cs(fftw_complex *A_4, fftw_complex *Cxxx, fftw_complex *Cyyy, fftw_complex *Czzz, 
            fftw_complex *Cxxy, fftw_complex *Cxxz, fftw_complex *Cyyx, fftw_complex *Cyyz, 
            fftw_complex *Czzx, fftw_complex *Czzy, fftw_complex *Cxyy, fftw_complex *Cxzz, 
            fftw_complex *Cyzz, fftw_complex *Cxyz, fftw_complex *Cyxz, fftw_complex *Czxy, 
            int3 N, double3 L) {
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
    
    #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k <= N.z/2; ++k) {
                int index = k + (N.z/2 + 1)*(j + N.y*i);
                double k_magsq = kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k];
                A_4[index][0] = kx[i]*kx[i]*kx[i]*kx[i]*Cxxx[index][0];
                A_4[index][0] += ky[j]*ky[j]*ky[j]*ky[j]*Cyyy[index][0];
                A_4[index][0] += kz[k]*kz[k]*kz[k]*kz[k]*Czzz[index][0];
                A_4[index][0] += 4.0*kx[i]*kx[i]*kx[i]*ky[j]*Cxxy[index][0];
                A_4[index][0] += 4.0*kx[i]*kx[i]*kx[i]*kz[k]*Cxxz[index][0];
                A_4[index][0] += 4.0*ky[j]*ky[j]*ky[j]*kx[i]*Cyyx[index][0];
                A_4[index][0] += 4.0*ky[j]*ky[j]*ky[j]*kz[k]*Cyyz[index][0];
                A_4[index][0] += 4.0*kz[k]*kz[k]*kz[k]*kx[i]*Czzx[index][0];
                A_4[index][0] += 4.0*kz[k]*kz[k]*kz[k]*ky[j]*Czzy[index][0];
                A_4[index][0] += 6.0*kx[i]*kx[i]*ky[j]*ky[j]*Cxyy[index][0];
                A_4[index][0] += 6.0*kx[i]*kx[i]*kz[k]*kz[k]*Cxzz[index][0];
                A_4[index][0] += 6.0*ky[j]*ky[j]*kz[k]*kz[k]*Cyzz[index][0];
                A_4[index][0] += 12.0*kx[i]*ky[j]*kz[k]*kx[i]*Cxyz[index][0];
                A_4[index][0] += 12.0*kx[i]*ky[j]*kz[k]*ky[j]*Cyxz[index][0];
                A_4[index][0] += 12.0*kx[i]*ky[j]*kz[k]*kz[k]*Czxy[index][0];
                
                A_4[index][1] = kx[i]*kx[i]*kx[i]*kx[i]*Cxxx[index][1];
                A_4[index][1] += ky[j]*ky[j]*ky[j]*ky[j]*Cyyy[index][1];
                A_4[index][1] += kz[k]*kz[k]*kz[k]*kz[k]*Czzz[index][1];
                A_4[index][1] += 4.0*kx[i]*kx[i]*kx[i]*ky[j]*Cxxy[index][1];
                A_4[index][1] += 4.0*kx[i]*kx[i]*kx[i]*kz[k]*Cxxz[index][1];
                A_4[index][1] += 4.0*ky[j]*ky[j]*ky[j]*kx[i]*Cyyx[index][1];
                A_4[index][1] += 4.0*ky[j]*ky[j]*ky[j]*kz[k]*Cyyz[index][1];
                A_4[index][1] += 4.0*kz[k]*kz[k]*kz[k]*kx[i]*Czzx[index][1];
                A_4[index][1] += 4.0*kz[k]*kz[k]*kz[k]*ky[j]*Czzy[index][1];
                A_4[index][1] += 6.0*kx[i]*kx[i]*ky[j]*ky[j]*Cxyy[index][1];
                A_4[index][1] += 6.0*kx[i]*kx[i]*kz[k]*kz[k]*Cxzz[index][1];
                A_4[index][1] += 6.0*ky[j]*ky[j]*kz[k]*kz[k]*Cyzz[index][1];
                A_4[index][1] += 12.0*kx[i]*ky[j]*kz[k]*kx[i]*Cxyz[index][1];
                A_4[index][1] += 12.0*kx[i]*ky[j]*kz[k]*ky[j]*Cyxz[index][1];
                A_4[index][1] += 12.0*kx[i]*ky[j]*kz[k]*kz[k]*Czxy[index][1];
                
                A_4[index][0] /= k_magsq*k_magsq;
                A_4[index][1] /= k_magsq*k_magsq;
            }
        }
    }
}

void get_A4(std::vector<double> &dr, std::vector<double> &A_4, int3 N, double3 L, double3 r_min) {
    if (A_4.size() != N.x*N.y*2*(N.z/2 + 1)) {
        A_4.resize(N.x*N.y*2*(N.z/2 + 1));
    }
    std::vector<double> Cxxx(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cyyy(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Czzz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cxxy(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cxxz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cyyx(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cyyz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Czzx(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Czzy(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cxyy(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cxzz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cyzz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cxyz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Cyxz(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> Czxy(N.x*N.y*2*(N.z/2 + 1));
    generate_wisdom_fipr2c(Cxxx, N, wisdom_file, omp_get_max_threads());
    
    double3 del_r = {L.x/N.x, L.y/N.y, L.z/N.z};
    
    #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        double r_x = r_min.x + (i + 0.5)*del_r.x;
        for (int j = 0; j < N.y; ++j) {
            double r_y = r_min.y + (j + 0.5)*del_r.y;
            for (int k = 0; k < N.z; ++k) {
                double r_z = r_min.z + (k + 0.5)*del_r.z;
                double r_magsq = r_x*r_x + r_y*r_y+ r_z*r_z;
                int index1 = k + N.z*(j + N.y*i);
                int index2 = k + 2*(N.z/2 + 1)*(j + N.y*j);
                
                Cxxx[index2] = (r_x*r_x*r_x*r_x*dr[index1])/(r_magsq*r_magsq);
                Cyyy[index2] = (r_y*r_y*r_y*r_y*dr[index1])/(r_magsq*r_magsq);
                Czzz[index2] = (r_z*r_z*r_z*r_z*dr[index1])/(r_magsq*r_magsq);
                Cxxy[index2] = (r_x*r_x*r_x*r_y*dr[index1])/(r_magsq*r_magsq);
                Cxxz[index2] = (r_x*r_x*r_x*r_z*dr[index1])/(r_magsq*r_magsq);
                Cyyx[index2] = (r_y*r_y*r_y*r_x*dr[index1])/(r_magsq*r_magsq);
                Cyyz[index2] = (r_y*r_y*r_y*r_z*dr[index1])/(r_magsq*r_magsq);
                Czzx[index2] = (r_z*r_z*r_z*r_x*dr[index1])/(r_magsq*r_magsq);
                Czzy[index2] = (r_z*r_z*r_z*r_y*dr[index1])/(r_magsq*r_magsq);
                Cxyy[index2] = (r_x*r_x*r_y*r_y*dr[index1])/(r_magsq*r_magsq);
                Cxzz[index2] = (r_x*r_x*r_z*r_z*dr[index1])/(r_magsq*r_magsq);
                Cyzz[index2] = (r_y*r_y*r_z*r_z*dr[index1])/(r_magsq*r_magsq);
                Cxyz[index2] = (r_x*r_x*r_y*r_z*dr[index1])/(r_magsq*r_magsq);
                Cyxz[index2] = (r_y*r_y*r_x*r_z*dr[index1])/(r_magsq*r_magsq);
                Czxy[index2] = (r_z*r_z*r_x*r_y*dr[index1])/(r_magsq*r_magsq);
            }
        }
    }
    
    fip_r2c(Cxxx, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cyyy, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Czzz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cxxy, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cxxz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cyyx, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cyyz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Czzx, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Czzy, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cxyy, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cxzz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cyzz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cxyz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Cyxz, N, wisdom_file, omp_get_max_threads());
    fip_r2c(Czxy, N, wisdom_file, omp_get_max_threads());
    
    sum_Cs((fftw_complex *)A_4.data(), (fftw_complex *)Cxxx.data(), (fftw_complex *)Cyyy.data(), 
           (fftw_complex *)Czzz.data(), (fftw_complex *)Cxxy.data(), (fftw_complex *)Cxxz.data(), 
           (fftw_complex *)Cyyx.data(), (fftw_complex *)Cyyz.data(), (fftw_complex *)Czzx.data(), 
           (fftw_complex *)Czzy.data(), (fftw_complex *)Cxyy.data(), (fftw_complex *)Cxzz.data(), 
           (fftw_complex *)Cyzz.data(), (fftw_complex *)Cxyz.data(), (fftw_complex *)Cyxz.data(), 
           (fftw_complex *)Czxy.data(), N, L);
}
