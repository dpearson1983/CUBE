#ifndef _HARPPI_H_
#define _HARPPI_H_

#include <string>
#include <vector>
#include <map>

struct typekey{
    std::string type, key;
};

class parameters{
    std::map <std::string, std::string> strings;
    std::map <std::string, int> ints;
    std::map <std::string, double> doubles;
    std::map <std::string, bool> bools;
    std::map <std::string, std::vector<double> > dvectors;
    std::map <std::string, std::vector<int> > ivectors;
    std::map <std::string, std::vector<std::string> > svectors;
    std::map <std::string, std::vector<bool> > bvectors;
    
    bool assignParams(std::string type, std::string key, std::string val);
    
    public:
        void readParams(char *file);
        
        void print();
        
        void check_min(std::vector<typekey> neededParams);
        
        double getd(std::string key, int element = 0);
        
        int geti(std::string key, int element = 0);
        
        bool getb(std::string key, int element = 0);
        
        std::string gets(std::string key, int element = 0);
        
        bool checkParam(std::string key);
        
        parameters(char *file);
};

#endif
