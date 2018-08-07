#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <vector_types.h>

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
__device__ double realPart(double3 k1, double3, k2, double3 k3) {
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
    
    int index = 0;
    for (int i = 0; i < N; ++i) {
        double k_1 = k_min + (i + 0.5)*binWidth;
        for (int j = i; j < N; ++j) {
            double k_2 = k_min + (j + 0.5)*binWidth;
            for (int k = j; k < N; ++k) {
                double k_3 = k_min + (k + 0.5)*binWidth;
                if (k_3 <= k_1 + k_2 && k_3 <= k_max) {
                    if (k_1 == k1 && k_2 == k2 && k_3 == k3) {
                        return index;
                    } else {
                        index++;
                    }
                }
            }
        }
    }
    return -1;
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
__global__ void calcB_0(double3 *A_0, int4 *k_vec, double *B_0, int4 N_grid, int N_kvec, 
                        double binWidth, int numBins, double2 k_lim) {
    int k_j = threadIdx.x + blockIdx.x*blockDim.x;
    int k_i = threadIdx.y + blockIdx.y*blockDim.y;
    // TODO: Look into how this stacking will affect the dimensionality of the 2D grid of blocks.
    if (k_j < k_i) {
        k_i = N_kvec - k_i;
        k_j = N_kvec - 1 - k_j;
    }
    
    // TODO: Change to a variable that is passed to the function instead of calculating each time.
    //       This is likely a fairly minor optimization, but is an optimization, nonetheless.
    int xShift = N_grid.x/2;
    int yShift = N_grid.y/2;
    int zShift = N_grid.z/2;
    
    if (k_j < N_kvec && k_i < N_kvec) {
        int4 k_k = {-k_i.x - k_j.x, -k_i.y - k_j.y, -k_i.z - k_j.z, 0};
        int3 i = {k_k.x + xShift, k_k.y + yShift, k_k.z + zShift};
        if (i.x >= 0 && i.y >= 0, && i.z >=0 && i.x < N_grid.x && i.y < N_grid.y && i.z < N_grid.z) {
            k_k.w = i.z + N_grid.z*(i.y + N_grid.y*i.x);
            double val = realPart(A_0[k_i.w], A_0[k_j.w], A_0[k_k.w]);
            // TODO: Add error checking for bin number, e.g. if bin = -1 handle error
            int bin = getBin(A_0[k_i.w].z, A_0[k_j.w].z, A_0[k_k.w].z, binWidth, numBins, k_lim.x);
            atomicAdd(&B_0[bin], val);
        }
    }
}

__global__ void calcB_02(double3 *A_0, double3 *A_2, int4 *k_vec, double *B_0, double *B_2, 
                         int4 N_grid, int N_kvec, double binWidth, int numBins, double2 k_lim) {
    int k_j = threadIdx.x + blockIdx.x*blockDim.x;
    int k_i = threadIdx.y + blockIdx.y*blockDim.y;
    if (k_j < k_i) {
        k_i = N_kvec - k_i;
        k_j = N_kvec - 1 - k_j;
    }
    
    if (k_2 < N_kvec && k_1 < N_kvec) {
        int4 k_k = {-k_i.x - k_j.x, -k_i.y - k_j.y, -k_i.z - k_j.z, 0};
        int3 i = {k_k.x + xShift, k_k.y + yShift, k_k.z + zShift};
        if (i.x >= 0 && i.y >= 0, && i.z >=0 && i.x < N_grid.x && i.y < N_grid.y && i.z < N_grid.z) {
            k_k.w = i.z + N_grid.z*(i.y + N_grid.y*i.x);
            double B0 = realPart(A_0[k_i.w], A_0[k_j.w], A_0[k_k.w]);
            double B2 = realPart(A_2[k_i.w], A_0[k_j.w], A_0[k_k.w]);
            int bin = getBin(A_0[k_i.w].z, A_0[k_j.w].z, A_0[k_k.w].z, binWidth, numBins, k_lim.x);
            atomicAdd(&B_0[bin], B0);
            atomicAdd(&B_2[bin], B2);
        }
    }
}

__global__ void calcN_tri(double3 *A_0, int4 *k_vec, unsigned int *N_tri, int4 N_grid, int N_kvec,
                          double binWidth, int numBins, double2 k_lim) {
    int k_j = threadIdx.x + blockIdx.x*blockDim.x;
    int k_i = threadIdx.y + blockIdx.y*blockDim.y;
    // TODO: Look into how this stacking will affect the dimensionality of the 2D grid of blocks.
    if (k_j < k_i) {
        k_i = N_kvec - k_i;
        k_j = N_kvec - 1 - k_j;
    }
    
    // TODO: Change to a variable that is passed to the function instead of calculating each time.
    //       This is likely a fairly minor optimization, but is an optimization, nonetheless.
    int xShift = N_grid.x/2;
    int yShift = N_grid.y/2;
    int zShift = N_grid.z/2;
    
    if (k_j < N_kvec && k_i < N_kvec) {
        int4 k_k = {-k_i.x - k_j.x, -k_i.y - k_j.y, -k_i.z - k_j.z, 0};
        int3 i = {k_k.x + xShift, k_k.y + yShift, k_k.z + zShift};
        if (i.x >= 0 && i.y >= 0, && i.z >=0 && i.x < N_grid.x && i.y < N_grid.y && i.z < N_grid.z) {
            k_k.w = i.z + N_grid.z*(i.y + N_grid.y*i.x);
            int bin = getBin(A_0[k_i.w].z, A_0[k_j.w].z, A_0[k_k.w].z, binWidth, numBins, k_lim.x);
            atomicAdd(&N_tri[bin], 1);
        }
    }
}

__global__ normB_l(double *B_l, unsigned int *N_tri, double norm, int numBins) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < numBins && N_tri[tid] > 0) {
        B_l[tid] /= (norm*N_tri[tid]);
    }
}

#endif
