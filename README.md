# CUBE - CUDA Bispectrum Estimation
This software will calculate the galaxy bispectrum monopole and quadrupole using NVIDIA GPUs to 
speed up the *O*(*N*<sup>2</sup>) calculation. This is based on the code that was used in this [paper](https://arxiv.org/abs/1712.04970). However, this version *should* be more efficient by having a better work
load balance between the threads.

**NOTE:** This code is still in developement and has not yet been tested. In fact, there are still large chunks of code that needs to be written before this software can run.

## Structure of the thread blocks
When calculating the galaxy bispectrum, it is first necessary to calculate overdensity fields, and 
then perform a Fourier transform. This gives you 3D field of frequency vectors. The first step after
performing the Fourier transform is to then cut down the size of the cube by removing points where
|**_k_**| > *k*<sub>max</sub>. This makes it easier to fit in what is often limited GPU memory.

From that cut down 3D field,
a 1D list of vectors whose magnitude are *k*<sub>min</sub> < |**_k_**| < *k*<sub>max</sub> is created
which stores the integer multiples of the fundamental frequencies of each component as well as the 
flattened index in the cut down 3D field. This allows look up times for **_k_**<sub>1</sub> and 
**_k_**<sub>2</sub> to become negligible.
The first version of this software (never publicly released) used a single thread for each 
**_k_**<sub>1</sub> vector, then ran through all possible combinations of **_k_**<sub>2</sub> vectors for that **_k_**<sub>1</sub> vector. After using a vector as **_k_**<sub>1</sub>, we don't need to
consider it again, and we simply need to start **_k_**<sub>2</sub> at the point in the 1D list that
corresponds to the current **_k_**<sub>1</sub>. Because of this, the first version of the code had
some threads doing a lot of work, and others doing almost nothing. For this version, the threads are
organized in a 2D grid of dimensions *N*<sub>k</sub>x(*N*<sub>k</sub>/2 + 1). The *x* dimension runs
over **_k_**<sub>2</sub> and the *y* over **_k_**<sub>1</sub>. The reason the second dimension is 
roughly halved is because of not wanting to double count triangles. When we get to the second row of
our threads in the 2D grid, the very first thread would repeat the calculation of the second thread
of the first row. Instead, the code is setup to calculate the value from what would be the very last
thread of the last row in a square array. The first two threads of the third row calculate the values
that would have been in the last two threads of the second to last row, and so on.

This grid of threads is, of course, sub-divided into thread blocks, also arranged in 2D. By "tucking
up" the non-repeating part of the bottom half of the 2D grid, we can keep the vast majority of thread
blocks full of active threads, and each threads work load should be almost identical meaning no individual thread should be idle for too long.

## To Do List:
1. ~~Add in parts of BESTFITS that can be re-used in this code~~
2. ~~Create code to cut out the small cubes~~
3. ~~Implement Bianchi et al. 2015 method of encoded line-of-sight information to get A_2 for the quadrupole calculation.~~
4. Structure main.cu
