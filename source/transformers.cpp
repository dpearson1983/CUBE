#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <fftw3.h>
#include <omp.h>
#include <vector_types.h>
#include "../include/transformers.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

std::vector<double> fft_freq(int N, double L) {
    std::vector<double> k;
    k.reserve(N);
    double dk = (2.0*PI)/L;
    for (int i = 0; i <= N/2; ++i)
        k.push_back(i*dk);
    for (int i = N/2 + 1; i < N; ++i)
        k.push_back((i - N)*dk);
    return k;
}

double fundamental_frequency(int N, double L) {
    return (2.0*PI)/L;
}

void generate_wisdom_foopr2c(std::vector<double> &dr, std::vector<fftw_complex> &dk, int3 N, 
                       std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    size_t N_com = N.x*N.y*(N.z/2 + 1);
    if (dr.size() == N_tot && dk.size() == N_com) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dr2dk = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, dr.data(), dk.data(), FFTW_MEASURE);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_destroy_plan(dr2dk);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
}

void generate_wisdom_boopc2r(std::vector<double> &dr, std::vector<fftw_complex> &dk, int3 N, 
                       std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    size_t N_com = N.x*N.y*(N.z/2 + 1);
    if (dr.size() == N_tot && dk.size() == N_com) {
    fftw_init_threads();
    fftw_import_wisdom_from_filename(wisdom_file.c_str());
    fftw_plan_with_nthreads(nthreads);
    fftw_plan dk2dr = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, dk.data(), dr.data(), FFTW_MEASURE);
    fftw_export_wisdom_to_filename(wisdom_file.c_str());
    
    fftw_destroy_plan(dk2dr);
    fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void generate_wisdom_fipr2c(std::vector<double> &delta, int3 N, std::string wisdom_file,
                   int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*2*(N.z/2 + 1);
    if (delta.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dr2dk = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, delta.data(),
                                               (fftw_complex *) delta.data(), FFTW_MEASURE);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_destroy_plan(dr2dk);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void generate_wisdom_bipc2r(std::vector<double> &delta, int3 N, std::string wisdom_file,
                   int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*2*(N.z/2 + 1);
    if (delta.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dk2dr = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, (fftw_complex *) delta.data(),
                                               delta.data(), FFTW_MEASURE);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_destroy_plan(dk2dr);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void generate_wisdom_foopc2c(std::vector<fftw_complex> &dr, std::vector<fftw_complex> &dk, int3 N,
             std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    if (dr.size() == N_tot && dk.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dr2dk = fftw_plan_dft_3d(N.x, N.y, N.z, dr.data(), dk.data(), FFTW_FORWARD, 
                                           FFTW_MEASURE);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_destroy_plan(dr2dk);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
}

void generate_wisdom_boopc2c(std::vector<fftw_complex> &dr, std::vector<fftw_complex> &dk, int3 N,
             std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    if (dr.size() == N_tot && dk.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dk2dr = fftw_plan_dft_3d(N.x, N.y, N.z, dk.data(), dr.data(), FFTW_BACKWARD, 
                                           FFTW_MEASURE);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_destroy_plan(dk2dr);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
}

void generate_wisdom_fipc2c(std::vector<fftw_complex> &delta, int3 N,
             std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    if (delta.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dr2dk = fftw_plan_dft_3d(N.x, N.y, N.z, delta.data(), delta.data(), FFTW_FORWARD, 
                                           FFTW_MEASURE);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_destroy_plan(dr2dk);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
}

void generate_wisdom_bipc2c(std::vector<fftw_complex> &delta, int3 N,
             std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    if (delta.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dk2dr = fftw_plan_dft_3d(N.x, N.y, N.z, delta.data(), delta.data(), FFTW_BACKWARD, 
                                           FFTW_MEASURE);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_destroy_plan(dk2dr);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
} 

void foop_r2c(std::vector<double> &dr, std::vector<fftw_complex> &dk, int3 N, 
                       std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    size_t N_com = N.x*N.y*(N.z/2 + 1);
    if (dr.size() == N_tot && dk.size() == N_com) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dr2dk = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, dr.data(), dk.data(), FFTW_WISDOM_ONLY);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_execute(dr2dk);
        
        fftw_destroy_plan(dr2dk);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
}

void boop_c2r(std::vector<double> &dr, std::vector<fftw_complex> &dk, int3 N,
                        std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    size_t N_com = N.x*N.y*(N.z/2 + 1);
    if (dr.size() == N_tot && dk.size() == N_com) {
    fftw_init_threads();
    fftw_import_wisdom_from_filename(wisdom_file.c_str());
    fftw_plan_with_nthreads(nthreads);
    fftw_plan dk2dr = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, dk.data(), dr.data(), FFTW_WISDOM_ONLY);
    fftw_export_wisdom_to_filename(wisdom_file.c_str());
    
    fftw_execute(dk2dr);
    
    fftw_destroy_plan(dk2dr);
    fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void fip_r2c(std::vector<double> &delta, int3 N, std::string wisdom_file,
                   int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*2*(N.z/2 + 1);
    if (delta.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dr2dk = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, delta.data(),
                                               (fftw_complex *) delta.data(), FFTW_WISDOM_ONLY);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_execute(dr2dk);
        
        fftw_destroy_plan(dr2dk);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void bip_c2r(std::vector<double> &delta, int3 N, std::string wisdom_file,
                   int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*2*(N.z/2 + 1);
    if (delta.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dk2dr = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, (fftw_complex *) delta.data(),
                                               delta.data(), FFTW_WISDOM_ONLY);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_execute(dk2dr);
        
        fftw_destroy_plan(dk2dr);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void foop_c2c(std::vector<fftw_complex> &dr, std::vector<fftw_complex> &dk, int3 N,
             std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    if (dr.size() == N_tot && dk.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dr2dk = fftw_plan_dft_3d(N.x, N.y, N.z, dr.data(), dk.data(), FFTW_FORWARD, 
                                           FFTW_WISDOM_ONLY);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_execute(dr2dk);
        
        fftw_destroy_plan(dr2dk);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
}

void boop_c2c(std::vector<fftw_complex> &dr, std::vector<fftw_complex> &dk, int3 N,
             std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    if (dr.size() == N_tot && dk.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dk2dr = fftw_plan_dft_3d(N.x, N.y, N.z, dk.data(), dr.data(), FFTW_BACKWARD, 
                                           FFTW_WISDOM_ONLY);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_execute(dk2dr);
        
        fftw_destroy_plan(dk2dr);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
}

void fip_c2c(std::vector<fftw_complex> &delta, int3 N,
             std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    if (delta.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dr2dk = fftw_plan_dft_3d(N.x, N.y, N.z, delta.data(), delta.data(), FFTW_FORWARD, 
                                           FFTW_WISDOM_ONLY);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_execute(dr2dk);
        
        fftw_destroy_plan(dr2dk);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
}

void bip_c2c(std::vector<fftw_complex> &delta, int3 N,
             std::string wisdom_file, int nthreads = omp_get_max_threads()) {
    size_t N_tot = N.x*N.y*N.z;
    if (delta.size() == N_tot) {
        fftw_init_threads();
        fftw_import_wisdom_from_filename(wisdom_file.c_str());
        fftw_plan_with_nthreads(nthreads);
        fftw_plan dk2dr = fftw_plan_dft_3d(N.x, N.y, N.z, delta.data(), delta.data(), FFTW_BACKWARD, 
                                           FFTW_WISDOM_ONLY);
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        
        fftw_execute(dk2dr);
        
        fftw_destroy_plan(dk2dr);
        fftw_cleanup_threads();
    } else {
        std::stringstream err_msg;
        err_msg << "Incorrect array sizes.\n";
        throw std::runtime_error(err_msg.str());
    }        
} 
