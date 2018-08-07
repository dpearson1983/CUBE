#ifndef _TRANSFORMERS_H_
#define _TRANSFORMERS_H_

#include <vector>
#include <string>
#include <fftw3.h>
#include <omp.h>
#include "tpods.h"

std::vector<double> fft_freq(int N, double L);

double fundamental_frequency(int N, double L);

void foop_r2c(std::vector<double> &dr, std::vector<fftw_complex> &dk, vec3<int> N, 
                       std::string wisdom_file, int nthreads = omp_get_max_threads());

void boop_c2r(std::vector<double> &dr, std::vector<fftw_complex> &dk, vec3<int> N,
                        std::string wisdom_file, int nthreads = omp_get_max_threads());

void fip_r2c(std::vector<double> &delta, vec3<int> N, std::string wisdom_file,
                   int nthreads = omp_get_max_threads());

void bip_c2r(std::vector<double> &delta, vec3<int> N, std::string wisdom_file,
                   int nthreads = omp_get_max_threads());

void foop_c2c(std::vector<fftw_complex> &dr, std::vector<fftw_complex> &dk, vec3<int> N,
             std::string wisdom_file, int nthreads = omp_get_max_threads());

void boop_c2c(std::vector<fftw_complex> &dr, std::vector<fftw_complex> &dk, vec3<int> N,
             std::string wisdom_file, int nthreads = omp_get_max_threads());

void fip_c2c(std::vector<fftw_complex> &delta, vec3<int> N,
             std::string wisdom_file, int nthreads = omp_get_max_threads());

void bip_c2c(std::vector<fftw_complex> &delta, vec3<int> N,
             std::string wisdom_file, int nthreads = omp_get_max_threads());

#endif
