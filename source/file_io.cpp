#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <CCfits/CCfits>
#include <gsl/gsl_integration.h>
#include "../include/file_io.h"
#include "../include/tpods.h"
#include "../include/galaxy.h"
#include "../include/cosmology.h"

void getDR12Gals(std::string file, std::vector<galaxy> &gals, double z_min, double z_max) {
    std::unique_ptr<CCfits::FITS> pInfile(new CCfits::FITS(file, CCfits::Read, false));
    
    CCfits::ExtHDU &table = pInfile->extension(1);
    long start = 1L;
    long end = table.rows();
    
    std::vector<double> ra;
    std::vector<double> dec;
    std::vector<double> red;
    std::vector<double> nz;
    std::vector<double> w_fkp;
    std::vector<double> w_sys;
    
    table.column("RA").read(ra, start, end);
    table.column("DEC").read(dec, start, end);
    table.column("Z").read(red, start, end);
    table.column("NZ").read(nz, start, end);
    table.column("WEIGHT_FKP").read(w_fkp, start, end);
    table.column("WEIGHT_SYSTOT").read(w_sys, start, end);
    
    for (size_t i = 0; i < ra.size(); ++i) {
        if (red[i] >= z_min && red[i] < z_max) {
            galaxy gal(ra[i], dec[i], red[i], nz[i], w_sys[i]*w_fkp[i]);
            gals.push_back(gal);
        }
    }
}

void getDR12Rans(std::string file, std::vector<galaxy> &gals, double z_min, double z_max) {
    std::unique_ptr<CCfits::FITS> pInfile(new CCfits::FITS(file, CCfits::Read, false));
    
    CCfits::ExtHDU &table = pInfile->extension(1);
    long start = 1L;
    long end = table.rows();
    
    std::vector<double> ra;
    std::vector<double> dec;
    std::vector<double> red;
    std::vector<double> nz;
    std::vector<double> w_fkp;
    
    table.column("RA").read(ra, start, end);
    table.column("DEC").read(dec, start, end);
    table.column("Z").read(red, start, end);
    table.column("NZ").read(nz, start, end);
    table.column("WEIGHT_FKP").read(w_fkp, start, end);
    
    for (size_t i = 0; i < ra.size(); ++i) {
        if (red[i] >= z_min && red[i] < z_max) {
            galaxy gal(ra[i], dec[i], red[i], nz[i], w_fkp[i]);
            gals.push_back(gal);
        }
    }
}

vec3<double> getRMin(std::vector<galaxy> &gals, cosmology &cosmo, vec3<int> N, vec3<double> &L) {
    vec3<double> r_min = {1E17, 1E17, 1E17};
    vec3<double> r_max = {-1E17, -1E17, -1E17};
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < gals.size(); ++i) {
        vec3<double> pos = gals[i].get_unshifted_cart(cosmo, ws);
        if (pos.x < r_min.x) r_min.x = pos.x;
        if (pos.y < r_min.y) r_min.y = pos.y;
        if (pos.z < r_min.z) r_min.z = pos.z;
        if (pos.x > r_max.x) r_max.x = pos.x;
        if (pos.y > r_max.y) r_max.y = pos.y;
        if (pos.z > r_max.z) r_max.z = pos.z;
    }
    gsl_integration_workspace_free(ws);
    
    std::ofstream fout("grid_properties.log");
    
    L.x = r_max.x - r_min.x;
    L.y = r_max.y - r_min.y;
    L.z = r_max.z - r_min.z;
    
    fout << "Minimum box dimensions:\n";
    fout << "   L.x = " << L.x << "\n";
    fout << "   L.y = " << L.y << "\n";
    fout << "   L.z = " << L.z << "\n";
    
    double resx = L.x/N.x;
    double resy = L.y/N.y;
    double resz = L.z/N.z;
    
    fout << "\nRaw resolutions:\n";
    fout << " res.x = " << resx << "\n";
    fout << " res.y = " << resy << "\n";
    fout << " res.z = " << resz << "\n";
    
    double resolution = resx;
    if (resy > resolution) resolution = resy;
    if (resz > resolution) resolution = resz;
    
    resolution = floor(resolution*2 + 0.5)/2;
    
    fout << "\nAdopted resolution:\n";
    fout << "   resolution = " << resolution << "\n";
    
    // Centering the sample
    r_min.x -= (resolution*N.x - L.x)/2.0;
    r_min.y -= (resolution*N.y - L.y)/2.0;
    r_min.z -= (resolution*N.z - L.z)/2.0;
    
    fout << "\nMinimum box position:\n";
    fout << "   r_min.x = " << r_min.x << "\n";
    fout << "   r_min.y = " << r_min.y << "\n";
    fout << "   r_min.z = " << r_min.z << "\n";
    
    L.x = resolution*N.x;
    L.y = resolution*N.y;
    L.z = resolution*N.z;
    
    fout << "\nFinal box dimensions:\n";
    fout << "   L.x = " << L.x << "\n";
    fout << "   L.y = " << L.y << "\n";
    fout << "   L.z = " << L.z << "\n";
    fout.close();
    
    return r_min;
}

void readDR12(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> L, 
              vec3<double> r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, 
              double z_min, double z_max) {
    std::vector<galaxy> gals;
    getDR12Gals(file, gals, z_min, z_max);
    
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < gals.size(); ++i) {
        gals[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void readDR12Ran(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> &L, 
              vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, 
              double z_min, double z_max) {
    std::vector<galaxy> rans;
    getDR12Rans(file, rans, z_min, z_max);
    
    r_min = getRMin(rans, cosmo, N, L);
    
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < rans.size(); ++i) {
        rans[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void readPatchy(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> L, 
                vec3<double> r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw,
                double z_min, double z_max) {
    std::vector<galaxy> gals;
    
    std::ifstream fin(file);
    while (!fin.eof()) {
        double ra, dec, red, mstar, nbar, bias, veto_flag, fiber_collision;
        fin >> ra >> dec >> red >> mstar >> nbar >> bias >> veto_flag >> fiber_collision;
        if (red >= z_min && red < z_max) {
            double w_fkp = (veto_flag*fiber_collision)/(1.0 + nbar*10000.0);
            galaxy gal(ra, dec, red, nbar, w_fkp);
            gals.push_back(gal);
        }
    }
    fin.close();
    
//     vec3<double> r_min = getRMin(gals, cosmo, L);
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < gals.size(); ++i) {
        gals[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void readPatchyRan(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> &L, 
                vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, 
                double z_min, double z_max) {
    std::vector<galaxy> rans;
    
    std::ifstream fin(file);
    while (!fin.eof()) {
        double ra, dec, red, nbar, bias, veto_flag, fiber_collision;
        fin >> ra >> dec >> red >> nbar >> bias >> veto_flag >> fiber_collision;
        if (red >= z_min && red < z_max) {
            double w_fkp = (veto_flag*fiber_collision)/(1.0 + nbar*10000.0);
            galaxy ran(ra, dec, red, nbar, w_fkp);
            rans.push_back(ran);
        }
    }
    fin.close();
    
    r_min = getRMin(rans, cosmo, N, L);
    
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < rans.size(); ++i) {
        rans[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void setFileType(std::string typeString, FileType &type) {
    if (typeString == "DR12") {
        type = dr12;
    } else if (typeString == "patchy") {
        type = patchy;
    } else if (typeString == "DR12_ran") {
        type = dr12_ran;
    } else if (typeString == "patchy_ran") {
        type = patchy_ran;
    } else {
        std::stringstream err_msg;
        err_msg << "Unrecognized or unsupported file type.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void readFile(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> &L, 
              vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw,
              double z_min, double z_max, FileType type) {
    switch(type) {
        case dr12:
            readDR12(file, delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case patchy:
            readPatchy(file, delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case dr12_ran:
            readDR12Ran(file, delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case patchy_ran:
            readPatchyRan(file, delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
        default:
            std::stringstream err_msg;
            err_msg << "Unrecognized or unsupported file type.\n";
            throw std::runtime_error(err_msg.str());
            break;
    }
}

void writeBispectrumFile(std::string file, std::vector<vec3<double>> &ks, std::vector<double> &B) {
    std::ofstream fout(file);
    fout.precision(15);
    for (size_t i = 0; i < B.size(); ++i) {
        fout << ks[i].x << " " << ks[i].y << " " << ks[i].z << " " << B[i] << "\n";
    }
    fout.close();
}

void writeShellFile(std::string file, std::vector<double> &shell, vec3<int> N) {
    size_t N_tot = N.x*N.y*N.z;
    std::ofstream fout(file, std::ios::out|std::ios::binary);
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k < N.z; ++k) {
                int index = k + 2*(N.z/2 + 1)*(j + N.y*i);
                shell[index] /= N_tot;
                fout.write((char *) &shell[index], sizeof(double));
            }
        }
    }
    fout.close();
}

void writePowerSpectrumFile(std::string file, std::vector<double> &ks, std::vector<double> &P) {
    std::ofstream fout(file);
    fout.precision(15);
    for (int i = 0; i < ks.size(); ++i) {
        fout << ks[i] << " " << P[i] << "\n";
    }
    fout.close();
}

std::string filename(std::string base, int digits, int num, std::string ext) {
    std::stringstream file;
    file << base << std::setw(digits) << std::setfill('0') << num << "." << ext;
    return file.str();
}
