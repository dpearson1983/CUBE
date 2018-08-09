#ifndef _FILE_IO_H_
#define _FILE_IO_H_

#include <string>
#include <vector>
#include <vector_types.h>
#include "cosmology.h"

// 
enum FileType{
    unsupported,
    dr12,
    patchy,
    dr12_ran,
    patchy_ran
};

void setFileType(std::string typeString, FileType &type);

void readFile(std::string file, std::vector<double> &delta, int3 N, double3 &L, 
              double3 &r_min, cosmology &cosmo, double3 &pk_nbw, double3 &bk_nbw,
              double z_min, double z_max, FileType type);

void writeBispectrumFile(std::string file, std::vector<double3> &ks, std::vector<double> &B);

void writeShellFile(std::string file, std::vector<double> &shell, int3 N);

void writePowerSpectrumFile(std::string file, std::vector<double> &ks, std::vector<double> &P);

std::string filename(std::string base, int digits, int num, std::string ext);

#endif
