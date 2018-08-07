# CUBE - CUDA Bispectrum Estimation
This software will calculate the galaxy bispectrum monopole and quadrupole using NVIDIA GPUs to 
speed up the O(N^2) calculation. This is based on the code that was used in this [paper]{https://arxiv.org/abs/1712.04970}. However, this version *should* be more efficient by having a better work
load balance between the threads.

## Structure of the thread blocks
When calculating the galaxy bispectrum, it is first necessary to calculate overdensity fields, and 
then perform a Fourier transform. This gives you 3D field of frequency vectors. From that 3D field,
a 1D list of vectors whose magnitude are *k*<sub>min</sub> < |**_k_**| < *k*<sub>max</sub>
The first version of this software (never publicly released) used a single thread for each 
k<sub>1</sub> vector, then ran through all possible combinations of k<sub>2</sub> vectors for that 
k<sub>1</sub> vector. 
