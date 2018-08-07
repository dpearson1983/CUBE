#include "../include/cosmology.h"
#include <cmath>
#include <gsl/gsl_integration.h>

#define c 299792.458
#define G 6.6740831E-11
#define sigma 5.67037321E-8
#define convert 1.0502650413E-30
#define pi 3.14159265359

struct intParams{
    double OmM, OmL, Omb, Tcmb, H_0;
};

double cosmology::E(double z) {
    return sqrt((1.0 + z)*(1.0 + z)*(1.0 + z)*cosmology::Om_M + cosmology::Om_L);
}

double cosmology::E_inv(double z, void *params) {
    intParams p = *(intParams *)params;
    return 1.0/sqrt((1.0 + z)*(1.0 + z)*(1.0 + z)*p.OmM + p.OmL);
}

double cosmology::rd_int(double z, void *params) {
    intParams p = *(intParams *)params;
    double coeff2 = (9.0*c*c*c*p.H_0*p.H_0*p.Omb*convert)/(128.0*sigma*pi*G*p.Tcmb*p.Tcmb*p.Tcmb*p.Tcmb);
    double E = sqrt((1.0 + z)*(1.0 + z)*(1.0 + z)*p.OmM + p.OmL);
    double term1 = (coeff2/((1.0 + z))) + 1.0;
    return 1.0/(sqrt(term1)*E);
}

double cosmology::rz(double z, void *params) {
    intParams p = *(intParams *)params;
    double D = c/(100.0*sqrt(p.OmM*(1.0 + z)*(1.0 + z)*(1.0 + z) + p.OmL));
    return D;
}

cosmology::cosmology(double H_0, double OmegaM, double OmegaL, double Omegab, double Omegac, double Tau,
                     double TCMB) {
    cosmology::Om_M = OmegaM;
    cosmology::Om_L = OmegaL;
    cosmology::Om_b = Omegab;
    cosmology::Om_c = Omegac;
    cosmology::tau = Tau;
    cosmology::T_CMB = TCMB;
    cosmology::h = H_0/100.0;
}

double cosmology::Omega_M() {
    return cosmology::Om_M;
}

double cosmology::Omega_L() {
    return cosmology::Om_L;
}

double cosmology::Omega_bh2() {
    return cosmology::Om_b*cosmology::h*cosmology::h;
}

double cosmology::Omega_ch2() {
    return cosmology::Om_c*cosmology::h*cosmology::h;
}

double cosmology::h_param() {
    return cosmology::h;
}

double cosmology::H0() {
    return cosmology::h*100.0;
}

double cosmology::H(double z) {
    return cosmology::H0()*cosmology::E(z);
}

double cosmology::D_A(double z) {
    double D, error;
    intParams p;
    p.OmM = cosmology::Om_M;
    p.OmL = cosmology::Om_L;
    p.Omb = cosmology::Om_b;
    p.Tcmb = cosmology::T_CMB;
    p.H_0 = cosmology::H0();
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000000);
    gsl_function F;
    F.function = &cosmology::E_inv;
    F.params = &p;
    gsl_integration_qags(&F, 0.0, z, 1E-6, 1E-6, 1000000, w, &D, &error);
    D *= (c/((1 + z)*cosmology::H0()));
    return D;
}

double cosmology::D_V(double z) {
    double D_ang = cosmology::D_A(z);
    double H_z = cosmology::H(z);
    double D = pow((c*z*(1.0 + z)*(1.0 + z)*D_ang*D_ang)/H_z, 1.0/3.0);
    return D;
}

double cosmology::Theta() {
    return cosmology::T_CMB/2.7;
}

double cosmology::z_eq() {
    double div = (cosmology::Theta()*cosmology::Theta()*cosmology::Theta()*cosmology::Theta());
    return (2.50E4*cosmology::Omega_M()*cosmology::h*cosmology::h)/div;
}

double cosmology::z_d() {
    double Om_Mh2 = cosmology::Omega_M()*cosmology::h*cosmology::h;
    double Om_bh2 = cosmology::Om_b*cosmology::h*cosmology::h;
    double b1 = 0.313*pow(Om_Mh2, -0.419)*(1.0 + 0.607*pow(Om_Mh2, 0.674));
    double b2 = 0.238*pow(Om_Mh2, 0.223);
    return 1345*(pow(Om_Mh2, 0.251)/(1.0 + 0.659*pow(Om_Mh2, 0.828)))*(1.0 + b1*pow(Om_bh2, b2));
}

double cosmology::R(double z) {
    double Om_bh2 = cosmology::Om_b*cosmology::h*cosmology::h;
    double Theta4 = cosmology::Theta()*cosmology::Theta()*cosmology::Theta()*cosmology::Theta();
    return (31.5*Om_bh2)/(Theta4*(z/1000.0));
}

double cosmology::k_eq() {
//     return sqrt(2.0*cosmology::Omega_M()*cosmology::H0()*cosmology::H0()*cosmology::z_eq());
    return (0.0746*cosmology::Omega_M()*cosmology::h*cosmology::h)/        
                     (cosmology::Theta()*cosmology::Theta());
}

double cosmology::r_d()  {
    double R_d = cosmology::R(cosmology::z_d());
    double R_eq = cosmology::R(cosmology::z_eq());
    double coeff = (2.0/(3.0*cosmology::k_eq()))*sqrt(6.0/R_eq);
    double nat_log = log((sqrt(1.0 + R_d) + sqrt(R_d + R_eq))/(1.0 + sqrt(R_eq)));
    return coeff*nat_log;
}

double cosmology::comoving_distance(double z, gsl_integration_workspace *w) {
    double D, error;
    intParams p;
    p.OmM = cosmology::Omega_M();
    p.OmL = cosmology::Omega_L();
    p.Omb = cosmology::Om_b;
    p.Tcmb = cosmology::T_CMB;
    p.H_0 = 100.0;
    gsl_function F;
    F.function = &cosmology::rz;
    F.params = &p;
    gsl_integration_qags(&F, 0.0, z, 1E-6, 1E-6, 1000000, w, &D, &error);
    return D;
}
