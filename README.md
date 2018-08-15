# CUBE - CUDA Bispectrum Estimation
This software will calculate the galaxy bispectrum monopole and quadrupole using NVIDIA GPUs to 
speed up the *O*(*N*<sup>2</sup>) calculation. This is based on the code that was used in this [paper](https://arxiv.org/abs/1712.04970).

**NOTE:** This code is still in development and has not yet been tested. In fact, there are still large chunks of code that needs to be written before this software can run.

## To Do List:
1. ~~Add in parts of BESTFITS that can be re-used in this code~~
2. ~~Create code to cut out the small cubes~~
3. ~~Implement Bianchi et al. 2015 method of encoded line-of-sight information to get A_2 for the quadrupole calculation.~~
4. ~~Structure main.cu~~
5. Implement bispectrum monopole and quadrupole shot-noise estimator
6. ~~Remove tpods.h dependence and replace with vector_types.h~~
