#include <vector>
#include <sstream>
#include <cmath>
#include <gsl/gsl_integration.h>
#include "../include/cic.h"
#include "../include/galaxy.h"
#include "../include/cosmology.h"
#include "../include/tpods.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

void galaxy::set_cartesian(cosmology &cosmo, vec3<double> r_min, gsl_integration_workspace *ws) {
    double D = cosmo.comoving_distance(galaxy::red, ws);
    
    galaxy::cart.x = D*cos(galaxy::dec*PI/180.0)*cos(galaxy::ra*PI/180.0) - r_min.x;
    galaxy::cart.y = D*cos(galaxy::dec*PI/180.0)*sin(galaxy::ra*PI/180.0) - r_min.y;
    galaxy::cart.z = D*sin(galaxy::dec*PI/180.0) - r_min.z;
    
    galaxy::cart_set = true;
}

vec3<double> galaxy::get_unshifted_cart(cosmology &cosmo, gsl_integration_workspace *ws) {
    double D = cosmo.comoving_distance(galaxy::red, ws);
    
    vec3<double> cart;
    
    cart.x = D*cos(galaxy::dec*PI/180.0)*cos(galaxy::ra*PI/180.0);
    cart.y = D*cos(galaxy::dec*PI/180.0)*sin(galaxy::ra*PI/180.0);
    cart.z = D*sin(galaxy::dec*PI/180.0);
    
    return cart;
}

vec3<double> galaxy::get_cart() {
    return galaxy::cart;
}

galaxy::galaxy(double RA, double DEC, double RED, double NZ, double W) {
    galaxy::ra = RA;
    galaxy::dec = DEC;
    galaxy::red = RED;
    galaxy::nbar = NZ;
    galaxy::w = W;
    galaxy::cart_set = false;
}

void galaxy::bin(std::vector<double> &delta, vec3<int> N, vec3<double> L, vec3<double> r_min,
                 cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, 
                 gsl_integration_workspace *ws) {
    if (!galaxy::cart_set) galaxy::set_cartesian(cosmo, r_min, ws);
    
    pk_nbw.x += galaxy::w;
    pk_nbw.y += galaxy::w*galaxy::w;
    pk_nbw.z += galaxy::nbar*galaxy::w*galaxy::w;
    
    bk_nbw.x += galaxy::w*galaxy::w*galaxy::w;
    bk_nbw.y += galaxy::nbar*galaxy::w*galaxy::w*galaxy::w;
    bk_nbw.z += galaxy::nbar*galaxy::nbar*galaxy::w*galaxy::w*galaxy::w;
    
    std::vector<size_t> indices;
    std::vector<double> weights;
    
    getCICInfo(galaxy::get_cart(), N, L, indices, weights);
    
    if (indices.size() == weights.size()) {
        for (int i = 0; i < indices.size(); ++i)
            delta[indices[i]] += galaxy::w*weights[i];
    } else {
        std::stringstream err_msg;
        err_msg << "Something has gone terribly wrong inside getCICInfo.\n";
        throw std::runtime_error(err_msg.str());
    }
}
