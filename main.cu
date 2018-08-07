#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "include/bispec.h"
#include "include/transformers.h"
#include "include/power.h"
#include "include/harppi.h"

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    std::cout << "Initializing..." << std::endl;
    parameters p(argv[1]);
    p.print();
    
    // Setup the cosmology class object with values needed to get comoving distances.
    // NOTE: The distances returned will be in h^-1 Mpc, so the value of H_0 is not actually used.
    cosmology cosmo(p.getd("H_0"), p.getd("Omega_M"), p.getd("Omega_L"));
    
    // Storage for values
    vec3<double> gal_pk_nbw = {0.0, 0.0, 0.0};
    vec3<double> gal_bk_nbw = {0.0, 0.0, 0.0};
    vec3<double> ran_pk_nbw = {0.0, 0.0, 0.0};
    vec3<double> ran_bk_nbw = {0.0, 0.0, 0.0};
    
    // Both r_min and L will be set automatically when reading in randoms
    vec3<double> r_min;
    vec3<double> L;
    
    vec3<int> N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    
    std::cout << "Setting file type variables..." << std::endl;
    FileType dataFileType, ranFileType;
    setFileType(p.gets("dataFileType"), dataFileType);
    setFileType(p.gets("ranFileType"), ranFileType);
    
    std::vector<double> delta(N.x*N.y*2.0*(N.z/2 + 1));
    double alpha;
    
    std::cout << "Reading in data and randoms files..." << std::endl;
    // Since the N's can be large values, individual arrays for the FFTs will be quite large. Instead
    // of reusing a fixed number of arrays, by using braced enclosed sections, variables declared
    // within the braces will go out of scope, freeing the associated memory. Here, given how the
    // backend code works, there are two temporary arrays to store the galaxy field and the randoms
    // field.
    {
        std::vector<double> ran(N.x*N.y*N.z);
        std::vector<double> gal(N.x*N.y*N.z);
        
        std::cout << "   Getting randoms..." << std::endl;
        readFile(p.gets("randomsFile"), ran, N, L, r_min, cosmo, ran_pk_nbw, ran_bk_nbw, 
                 p.getd("z_min"), p.getd("z_max"), ranFileType);
        std::cout << "   Getting galaxies..." << std::endl;
        readFile(p.gets("dataFile"), gal, N, L, r_min, cosmo, gal_pk_nbw, gal_bk_nbw, p.getd("z_min"),
                 p.getd("z_max"), dataFileType);
        
        alpha = gal_pk_nbw.x/ran_pk_nbw.x;
        
        std::cout << "   Computing overdensity..." << std::endl;
        #pragma omp parallel for
        for (size_t i = 0; i < N.x; ++i) {
            for (size_t j = 0; j < N.y; ++j) {
                for (size_t k = 0; k < N.z; ++k) {
                    int index1 = k + N.z*(j + N.y*i);
                    int index2 = k + 2*(N.z/2 + 1)*(j + N.y*i);
                    
                    delta[index2] = gal[index1] - alpha*ran[index1];
                }
            }
        }
    }
    std::cout << "Done!" << std::endl;
    
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
}
