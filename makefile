VXX = nvcc
CXX = cuda-g++
CXXFLAGS = -march=native -mtune=native -O3
LIBGSL = -lgsl -lgslcblas -lm
LIBFFTW = -lfftw3 -lfftw3_omp -fopenmp
LIBFITS = -lCCfits -lcfitsio
VXXFLAGS = -arch=sm_52 -ccbin=cuda-g++ --compiler-options "$(LIBGSL) $(LIBFFTW) $(LIBFITS) $(CXXFLAGS)"

build: cic cosmology cube file_io galaxy harppi line_of_sight power transformers main.cu
	$(VXX) $(VXXFLAGS) -O3 -o $(HOME)/bin/cube main.cu obj/*.o
	
cic: source/cic.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(LIBFFTW) $(CXXFLAGS) -c -o obj/cic.o source/cic.cpp
	
cosmology: source/cosmology.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(CXXFLAGS) -c -o obj/cosmology.o source/cosmology.cpp
	
cube: source/cube.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/cube.o source/cube.cpp
	
file_io: source/file_io.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(LIBFFTW) $(LIBFITS) $(CXXFLAGS) -c -o obj/file_io.o source/file_io.cpp
	
galaxy: source/galaxy.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(LIBFFTW) $(CXXFLAGS) -c -o obj/galaxy.o source/galaxy.cpp

harppi: source/harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -c -o obj/harppi.o source/harppi.cpp
	
line_of_sight: source/line_of_sight.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/line_of_sight.o source/line_of_sight.cpp
	
power: source/power.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/power.o source/power.cpp
	
transformers: source/transformers.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/transformers.o source/transformers.cpp
