#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <vector_types.h>

__constant__ int3 d_kBins[691]; // 8292 bytes


// Define atomicAdd for older NVIDIA GPUs
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ > 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Returns the real part of the product of three complex numbers.
__device__ double realPart(double3 k1, double3 k2, double3 k3) {
    return k1.x*k2.x*k3.x - k1.x*k2.y*k3.y - k1.y*k2.x*k3.y - k1.y*k2.y*k3.x;
}

__device__ void swapIfGreater(double &x, double &y) {
    if (x > y) {
        double temp = x;
        x = y;
        y = temp;
    }
}

// This function finds the index for a given triplet of k's based on their magnitudes. Unfortunately,
// given the triangle and the k1 <= k2 <= k3 conditions, a direct calculation of the index is not
// very straightforward. For now, this function works, though its not very efficient. Once all of the
// core algorithms are working, this will be explored further to find a better algorithm. In testing,
// execution of this function seems to be on the order of a few microseconds on average.
__device__ int getBin(double k1, double k2, double k3, double binWidth, int N, double k_min, 
                      double k_max) {
    swapIfGreater(k1, k2);
    swapIfGreater(k1, k3);
    swapIfGreater(k2, k3);
    
    int i_k1 = (k1 - k_min)/binWidth;
    int j_k2 = (k2 - k_min)/binWidth;
    int k_k3 = (k3 - k_min)/binWidth;
    
    for (int i = 0; i < N; ++i) {
        if (i_k1 == d_kBins[i].x && j_k2 == d_kBins[i].y && k_k3 == d_kBins[i].z) {
            return i;
        }
    }
    return -1;
}

__device__ int3 get_indices(int index, int3 N) {
    int3 unravel;
    unravel.z = index % N.z;
    unravel.y = ((index - unravel.z)/N.z) % N.y;
    unravel.x = ((index - unravel.z)/N.z - unravel.y)/N.y;
    return unravel;
}

std::vector<int3> setBins(double binWidth, int N, double k_min, double k_max) {
    std::vector<int3> kBins;
    for (int i = 0; i < N; ++i) {
        double k_1 = k_min + (i + 0.5)*binWidth;
        for (int j = i; j < N; ++j) {
            double k_2 = k_min + (j + 0.5)*binWidth;
            for (int k = j; k < N; ++k) {
                double k_3 = k_min + (k + 0.5)*binWidth;
                if (k_3 <= k_1 + k_2 && k_3 < k_max) {
                    int3 temp = {i, j, k};
                    kBins.push_back(temp);
                }
            }
        }
    }
    return kBins;
}
    

/* This is the GPU function to calculate the bispectrum. It will be called with a 2D grid of 2D
 * thread blocks. The y dimension corresponds to k_1 and the x dimension corresponds to k_2,
 * up until the point that we need to bin the value at which point the vectors will be ordered from
 * smallest to largest to match the order associated with the bins.
 * 
 * Since there is a lot going on, and the data structures don't necessarily make obvious what is
 * being stored in each array, the following notes are provided:
 * 
 * 1. double3 *A_0 - The double3 data structure is provided by vector_types.h with members x, y, 
 *                    and z. This particular array stores the Fourier transformed overdensity
 *                    field with the real part at member x, and the imaginary part at member y.
 *                    For convenience, the last data member stores the magnitude of the k vector
 *                    at that grid point.
 * 2. int4 *k_vec - The int4 data structure is provided by vector_type.h with members x, y, z, and w.
 *                  This particular array stores the integer multiples of the fundamental frequencies
 *                  to define each k vector in the Fourier transformed overdensity grid. The x, y and
 *                  z members correspond to the x, y and z components of the vectors. The w member
 *                  stores the flattened array index for the location in A_0 to negate look-up time.
 * 3. double *Bk - This data structure simply stores the calculated bispectrum.
 * 4. int4 N_grid - This contains the dimensions of A_0 and the total number of elements in the
 *                  x, y, z and w members, respectively. This is used for bounds checking after
 *                  calculating k_3.
 * 5. int N_kvec - This is the number of k vectors stored in k_vec. It is used for bounds checking
 *                 each thread as there will usually be more total threads than total k vectors.
 * 6. double binWidth - This is the width of the bispectrum bins, used to locate which bin each k
 *                      vector belongs in, which in turn determines which Bk element to add the 
 *                      result to.
 * 7. int numBins - The number of bispectrum bins, which in this current iteration must be 691, or
 *                  a k_min = 0.04 and a k_max = 0.168 and a binWidth of 0.008. Eventually the code
 *                  will be made more flexible by making the shared data structure be defined with
 *                  the extern flag.
 * 8. double2 k_lim - The double2 data structure is provided by vector_types.h with members x and y.
 *                    This particular variable contains k_min in the x member and k_max in the y
 *                    member. 
 */
__global__ void calcB0(double3 *A_0, int4 *kvec, double *Bk, int3 N_grid,
                       int N, double binWidth, int numBins, double2 k_lim) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    int xShift = N_grid.x/2;
    int yShift = N_grid.y/2;
    int zShift = N_grid.z/2;
    
    __shared__ double Bk_local[4096];
    for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i) {
        Bk_local[i] = 0;
    }
    __syncthreads();
    
    if (tid < N) {
        int4 k_1 = kvec[tid];
        k_1.x *= -1;
        k_1.y *= -1;
        k_1.z *= -1;
        double3 dk_1 = A_0[k_1.w];
        int ik1 = (dk_1.z - k_lim.x)/binWidth;
        for (int i = tid; i < N; ++i) {
            int4 k_2 = kvec[i];
            double3 dk_2 = A_0[k_2.w];
            int4 k_3 = {k_1.x - k_2.x, k_1.y - k_2.y, k_1.z - k_2.z, 0};
            int i3, j3, k3;
            i3 = k_3.x + xShift;
            j3 = k_3.y + yShift;
            k3 = k_3.z + zShift;
            if (i3 >= 0 && j3 >= 0 && k3 >= 0 && i3 < N_grid.x && j3 < N_grid.y && k3 < N_grid.z) {
                k_3.w = k3 + N_grid.z*(j3 + N_grid.y*i3);
                double3 dk_3 = A_0[k_3.w];
                if (dk_3.z < k_lim.y && dk_3.z >= k_lim.x) {
                    double val = realPart(dk_1, dk_2, dk_3);
                    int ik2 = (dk_2.z - k_lim.x)/binWidth;
                    int ik3 = (dk_3.z - k_lim.x)/binWidth;
                    int bin = ik3 + numBins*(ik2 + numBins*ik1);
                    atomicAdd(&Bk_local[bin], val);
                }
            }
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i) {
        atomicAdd(&Bk[i], Bk_local[i]);
    }
}

__global__ void calcB2(double3 *A_0, double3 *A_2, int4 *kvec, double *Bk, int3 N_grid,
                       int N, double binWidth, int numBins, double2 k_lim) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    int xShift = N_grid.x/2;
    int yShift = N_grid.y/2;
    int zShift = N_grid.z/2;
    
    __shared__ double Bk_local[4096];
    for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i) {
        Bk_local[i] = 0;
    }
    __syncthreads();
    
    if (tid < N) {
        int4 k_1 = kvec[tid];
        k_1.x *= -1;
        k_1.y *= -1;
        k_1.z *= -1;
        double3 dk_1 = A_2[k_1.w];
        int ik1 = (dk_1.z - k_lim.x)/binWidth;
        for (int i = tid; i < N; ++i) {
            int4 k_2 = kvec[i];
            double3 dk_2 = A_0[k_2.w];
            int4 k_3 = {k_1.x - k_2.x, k_1.y - k_2.y, k_1.z - k_2.z, 0};
            int i3, j3, k3;
            i3 = k_3.x + xShift;
            j3 = k_3.y + yShift;
            k3 = k_3.z + zShift;
            if (i3 >= 0 && j3 >= 0 && k3 >= 0 && i3 < N_grid.x && j3 < N_grid.y && k3 < N_grid.z) {
                k_3.w = k3 + N_grid.z*(j3 + N_grid.y*i3);
                double3 dk_3 = A_0[k_3.w];
                if (dk_3.z < k_lim.y && dk_3.z >= k_lim.x) {
                    double val = realPart(dk_1, dk_2, dk_3);
                    int ik2 = (dk_2.z - k_lim.x)/binWidth;
                    int ik3 = (dk_3.z - k_lim.x)/binWidth;
                    int bin = ik3 + numBins*(ik2 + numBins*ik1);
                    atomicAdd(&Bk_local[bin], val);
                }
            }
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i) {
        atomicAdd(&Bk[i], Bk_local[i]);
    }
}

__global__ void calcNtri(double3 *A_0, int4 *k, unsigned int *N_tri, int3 N_grid, int N, 
                         float binWidth, int numBins, double2 k_lim) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    int xShift = N_grid.x/2;
    int yShift = N_grid.y/2;
    int zShift = N_grid.z/2;
    
    __shared__ unsigned int Ntri_local[4096];
    for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i) {
        Ntri_local[i] = 0;
    }
    __syncthreads();
    
    if (tid < N) {
        int4 k_1 = k[tid];
        double3 dk_1 = A_0[k_1.w];
        int ik1 = (dk_1.z - k_lim.x)/binWidth;
        for (int i = tid; i < N; ++i) {
            int4 k_2 = k[i];
            double3 dk_2 = A_0[k_2.w];
            int4 k_3 = {-k_1.x - k_2.x, -k_1.y - k_2.y, -k_1.z - k_2.z, 0};
            int i3 = k_3.x + xShift;
            int j3 = k_3.y + yShift;
            int k3 = k_3.z + zShift;
            if (i3 >= 0 && j3 >= 0 && k3 >= 0 && i3 < N_grid.x && j3 < N_grid.y && k3 < N_grid.z) {
                k_3.w = k3 + N_grid.z*(j3 + N_grid.y*i3);
                double3 dk_3 = A_0[k_3.w];
                if (dk_3.z < k_lim.y && dk_3.z >= k_lim.x) {
                    int ik2 = (dk_2.z - k_lim.x)/binWidth;
                    int ik3 = (dk_3.z - k_lim.x)/binWidth;
                    int bin = ik3 + numBins*(ik2 + numBins*ik1);
                    atomicAdd(&Ntri_local[bin], 1);
                }
            }
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i) {
        atomicAdd(&N_tri[i], Ntri_local[i]);
    }
}

__global__ void reduceGrid(double *Small, double *Large, int3 N, int N_small,
                           double2 k_lim, double Delta_k) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < N.x*N.y*N.z) {
        int3 indices = get_indices(tid, N);
        double k1 = k_lim.x + (indices.x + 0.5)*Delta_k;
        double k2 = k_lim.x + (indices.y + 0.5)*Delta_k;
        double k3 = k_lim.x + (indices.z + 0.5)*Delta_k;
        int bin = getBin(k1, k2, k3, Delta_k, N_small, k_lim.x, k_lim.y);
        atomicAdd(&Small[bin], Large[tid]);
    }
}

__global__ void reduceGrid(unsigned int *Small, unsigned int *Large, int3 N, int N_small,
                           double2 k_lim, double Delta_k) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < N.x*N.y*N.z) {
        int3 indices = get_indices(tid, N);
        double k1 = k_lim.x + (indices.x + 0.5)*Delta_k;
        double k2 = k_lim.x + (indices.y + 0.5)*Delta_k;
        double k3 = k_lim.x + (indices.z + 0.5)*Delta_k;
        int bin = getBin(k1, k2, k3, Delta_k, N_small, k_lim.x, k_lim.y);
        atomicAdd(&Small[bin], Large[tid]);
    }
}

__global__ void normB_l(double *B_l, unsigned int *N_tri, double norm, int numBins) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < numBins && N_tri[tid] > 0) {
        B_l[tid] /= (norm*N_tri[tid]);
    }
}

int getNumBispecBins(double k_min, double k_max, double binWidth, std::vector<double3> &ks) {
    int totBins = 0;
    int N = (k_max - k_min)/binWidth;
    
    for (int i = 0; i < N; ++i) {
        double k_1 = k_min + (i + 0.5)*binWidth;
        for (int j = i; j < N; ++j) {
            double k_2 = k_min + (j + 0.5)*binWidth;
            for (int k = j; k < N; ++k) {
                double k_3 = k_min + (k + 0.5)*binWidth;
                if (k_3 <= k_1 + k_2 && k_3 <= k_max) {
                    totBins++;
                    double3 kt = {k_1, k_2, k_3};
                    ks.push_back(kt);
                }
            }
        }
    }
    
    return totBins;
}

void writeBispectrumFile(std::string file, std::vector<double> &B_0, std::vector<double> &B_2,
                         std::vector<unsigned int> &N_tri, std::vector<double3> &ks) {
    std::ofstream fout(file);
    fout.precision(15); // TODO: Change to use <limits>
    for (int i = 0; i < B_0.size(); ++i) {
        fout << ks[i].x << " " << ks[i].y << " " << ks[i].z << " " << B_0[i] << " ";
        fout << B_2[i] << " " << N_tri[i] << "\n";
    }
    fout.close();
}

#endif
