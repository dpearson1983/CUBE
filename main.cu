#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "include/bispec.h"
#include "include/gpuerrchk.h"
#include "include/transformers.h"
#include "include/power.h"
#include "include/cube.h"
#include "include/cic.h"
#include "include/line_of_sight.h"
#include "include/file_io.h"
#include "include/harppi.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

int main(int argc, char *argv[]) {
    std::cout << "Initializing..." << std::endl;
    parameters p(argv[1]);
    p.print();
    
    // Setup the cosmology class object with values needed to get comoving distances.
    // NOTE: The distances returned will be in h^-1 Mpc, so the value of H_0 is not actually used.
    cosmology cosmo(p.getd("H_0"), p.getd("Omega_M"), p.getd("Omega_L"));
    
    // Storage for values
    double3 gal_pk_nbw = {0.0, 0.0, 0.0};
    double3 gal_bk_nbw = {0.0, 0.0, 0.0};
    double3 ran_pk_nbw = {0.0, 0.0, 0.0};
    double3 ran_bk_nbw = {0.0, 0.0, 0.0};
    
    // Both r_min and L will be set automatically when reading in randoms
    double3 r_min;
    double3 L;
    
    int3 N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    
    std::cout << "Setting file type variables..." << std::endl;
    FileType dataFileType, ranFileType;
    setFileType(p.gets("dataFileType"), dataFileType);
    setFileType(p.gets("ranFileType"), ranFileType);
    
    double alpha;
    std::vector<double> delta(N.x*N.y*N.z);
    
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
        for (size_t i = 0; i < gal.size(); ++i) {
            delta[i] = gal[i] - alpha*ran[i];
        }
    }
    std::cout << "Done!" << std::endl;
    
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
    
    std::vector<double> A_0(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> A_2(N.x*N.y*2*(N.z/2 + 1));
    
    get_A0(delta, A_0, N);
    get_A2(delta, A_2, N, L, r_min);
    CICbinningCorrection((fftw_complex *)A_0.data(), N, L, kx, ky, kz);
    CICbinningCorrection((fftw_complex *)A_2.data(), N, L, kx, ky, kz);
    
    // TODO: Setup up small cubes, call GPU functions, output result.
    
    double3 delta_k = {(2.0*PI)/L.x, (2.0*PI)/L.y, (2.0*PI)/L.z};
    int3 N_grid = getSmallGridDimensions(p.getd("k_min"), p.getd("k_max"), delta_k);
    std::cout << N_grid.x << "x" << N_grid.y << "x" << N_grid.z << std::endl;
    
    std::vector<double3> a_0(N_grid.x*N_grid.y*N_grid.z);
    std::vector<double3> a_2(N_grid.x*N_grid.y*N_grid.z);
    std::vector<int4> kvec;
    
    getSmallCube(a_0, N_grid, (fftw_complex *)A_0.data(), N, p.getd("k_min"), p.getd("k_max"), kx, ky,
                 kz, delta_k, kvec);
    getSmallCube(a_2, N_grid, (fftw_complex *)A_2.data(), N, p.getd("k_min"), p.getd("k_max"), kx, ky,
                 kz, delta_k);
    
    std::cout << a_0[42].x << ", " << a_2[42].x << std::endl;
    
    std::ofstream fout("kvec.dat");
    for (size_t i = 0; i < kvec.size(); ++i) {
        fout << kvec[i].x << " " << kvec[i].y << " " << kvec[i].z << " " << kvec[i].w << "\n";
    }
    fout.close();
    
    std::vector<double3> ks;
    int numBispecBins = getNumBispecBins(p.getd("k_min"), p.getd("k_max"), p.getd("Delta_k"), ks);
    std::cout << "Number of bispectrum bins: " << numBispecBins << std::endl;
    std::vector<unsigned int> N_tri(numBispecBins + 1);
    std::vector<double> B_0(numBispecBins);
    std::vector<double> B_2(numBispecBins);
    unsigned int *dN_tri;
    double *dB_0, *dB_2;
    double3 *da_0, *da_2;
    int4 *dkvec;
    
    // Allocate GPU memory
    gpuErrchk(cudaMalloc((void **)&dN_tri, N_tri.size()*sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **)&dB_0, numBispecBins*sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dB_2, numBispecBins*sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&da_0, N_grid.x*N_grid.y*N_grid.z*sizeof(double3)));
    gpuErrchk(cudaMalloc((void **)&da_2, N_grid.x*N_grid.y*N_grid.z*sizeof(double3)));
    gpuErrchk(cudaMalloc((void **)&dkvec, kvec.size()*sizeof(int4)));
    
    // Copy data to the GPU, this initializes dN_tri, dB_0 and dB_2 to zero
    gpuErrchk(cudaMemcpy(dN_tri, N_tri.data(), N_tri.size()*sizeof(unsigned int), 
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dB_0, B_0.data(), numBispecBins*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dB_2, B_2.data(), numBispecBins*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(da_0, a_0.data(), N_grid.x*N_grid.y*N_grid.z*sizeof(double3), 
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(da_2, a_2.data(), N_grid.x*N_grid.y*N_grid.z*sizeof(double3), 
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dkvec, kvec.data(), kvec.size()*sizeof(int4), cudaMemcpyHostToDevice));
    
    int numBlocks1D = kvec.size()/32 + 1;
    dim3 num_threads(32,32);
    dim3 num_blocks(numBlocks1D,numBlocks1D/2 + 1);
    double2 k_lim = {p.getd("k_min"), p.getd("k_max")};
    std::cout << num_blocks.x << " " << num_blocks.y << std::endl;
    
    calcN_tri<<<num_blocks,num_threads>>>(da_0, dkvec, dN_tri, N_grid, kvec.size(), p.getd("Delta_k"),
                                          numBispecBins, k_lim);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    calcB_02<<<num_blocks,num_threads>>>(da_0, da_2, dkvec, dB_0, dB_2, N_grid, kvec.size(), 
                                         p.getd("Delta_k"), numBispecBins, k_lim);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    int normBlocks, normThreads;
    if (numBispecBins <= 1024) {
        normBlocks = 1;
        normThreads = numBispecBins;
    } else {
        normBlocks = numBispecBins/1024 + 1;
        normThreads = 1024;
    }
    normB_l<<<normBlocks,normThreads>>>(dB_0, dN_tri, gal_bk_nbw.z, numBispecBins);
    normB_l<<<normBlocks,normThreads>>>(dB_2, dN_tri, gal_bk_nbw.z, numBispecBins);
    
    gpuErrchk(cudaMemcpy(B_0.data(), dB_0, numBispecBins*sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(B_2.data(), dB_2, numBispecBins*sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(N_tri.data(), dN_tri, N_tri.size()*sizeof(unsigned int), 
                         cudaMemcpyDeviceToHost));
    
    std::cout << N_tri[numBispecBins] << std::endl;
    
    writeBispectrumFile(p.gets("outFile"), B_0, B_2, N_tri, ks);
    
    gpuErrchk(cudaFree(dB_0));
    gpuErrchk(cudaFree(dB_2));
    gpuErrchk(cudaFree(da_0));
    gpuErrchk(cudaFree(da_2));
    gpuErrchk(cudaFree(dkvec));
    gpuErrchk(cudaFree(dN_tri));
    
    return 0;
}
