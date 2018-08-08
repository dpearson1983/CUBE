#ifndef _LINE_OF_SIGHT_H_
#define _LINE_OF_SIGHT_H_

#include <vector>
#include <vector_types.h>

void get_A0(std::vector<double> &dr, std::vector<double> &A_0, int3 N);

void get_A2(std::vector<double> &dr, std::vector<double> &A_2, int3 N, double3 L, double3 r_min);

void get_A4(std::vector<double> &dr, std::vector<double> &A_4, int3 N, double3 L, double3 r_min);

#endif
