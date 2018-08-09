#ifndef _GALAXY_H_
#define _GALAXY_H_

#include <vector>
#include <gsl/gsl_integration.h>
#include "cosmology.h"
#include <vector_types.h>

class galaxy{
    double ra, dec, red, w, nbar;
    double3 cart;
    bool cart_set;
    
    public:
        galaxy(double RA, double DEC, double RED, double NZ, double W);
        
        void bin(std::vector<double> &delta, int3 N, double3 L, double3 r_min,
                 cosmology &cosmo, double3 &pk_nbw, double3 &bk_nbw, 
                 gsl_integration_workspace *ws);
        
        void set_cartesian(cosmology &cosmo, double3 r_min, gsl_integration_workspace *ws);
        
        double3 get_unshifted_cart(cosmology &cosmo, gsl_integration_workspace *ws);
    
        double3 get_cart();
};

#endif
