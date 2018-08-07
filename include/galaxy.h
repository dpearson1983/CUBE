#ifndef _GALAXY_H_
#define _GALAXY_H_

#include <vector>
#include <gsl/gsl_integration.h>
#include "cosmology.h"
#include "tpods.h"

class galaxy{
    double ra, dec, red, w, nbar;
    vec3<double> cart;
    bool cart_set;
    
    public:
        galaxy(double RA, double DEC, double RED, double NZ, double W);
        
        void bin(std::vector<double> &delta, vec3<int> N, vec3<double> L, vec3<double> r_min,
                 cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, 
                 gsl_integration_workspace *ws);
        
        void set_cartesian(cosmology &cosmo, vec3<double> r_min, gsl_integration_workspace *ws);
        
        vec3<double> get_unshifted_cart(cosmology &cosmo, gsl_integration_workspace *ws);
    
        vec3<double> get_cart();
};

#endif
