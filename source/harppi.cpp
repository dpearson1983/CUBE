/* harppi.cpp v1.1
 * Updated: August 2, 2016
 * 
 * LICENSE: GPL v3
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 * UPDATES:
 * Changed the error messaging when calling the getd, geti, gets, or getb
 * functions to be more informative. Now it should let users know if they
 * are simply using the wrong get function or if they are calling a 
 * parameter that does not exist.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "../include/harppi.h"

parameters::parameters(char *file) {
    parameters::readParams(file);
}

bool parameters::assignParams(std::string type, std::string key, std::string val) {
    if (type == "string") {
        parameters::strings.insert(std::pair<std::string, std::string>(key,val));
    } else if (type == "int") {
        parameters::ints.insert(std::pair<std::string, int>(key,atof(val.c_str())));
    } else if (type == "double") {
        parameters::doubles.insert(std::pair<std::string, double>(key,atof(val.c_str())));
    } else if (type == "bool") {
        bool b;
        std::istringstream(val) >> std::boolalpha >> b;
        parameters::bools.insert(std::pair<std::string, bool>(key, b));
    } else if (type == "vector<double>") {
        std::vector<double> vec;
        std::istringstream iss(val);
        std::string s;
        while (std::getline(iss, s, ',')) {
            vec.push_back(atof(s.c_str()));
        }
        parameters::dvectors.insert(std::pair<std::string, std::vector<double> >(key, vec));
    } else if (type == "vector<int>") {
        std::vector<int> vec;
        std::istringstream iss(val);
        std::string s;
        while (std::getline(iss, s, ',')) {
            vec.push_back(atof(s.c_str()));
        }
        parameters::ivectors.insert(std::pair<std::string, std::vector<int> >(key, vec));
    } else if (type == "vector<string>") {
        std::vector<std::string> vec;
        std::istringstream iss(val);
        std::string s;
        while (std::getline(iss, s, ',')) {
            vec.push_back(s);
        }
        parameters::svectors.insert(std::pair<std::string, std::vector<std::string> >(key, vec));
    } else if (type == "vector<bool>") {
        std::vector<bool> vec;
        std::istringstream iss(val);
        std::string s;
        while (std::getline(iss, s, ',')) {
            bool b;
            std::istringstream(s) >> std::boolalpha >> b;
            vec.push_back(b);
        }
        parameters::bvectors.insert(std::pair<std::string, std::vector<bool> >(key, vec));
    } else {
        std::cout << "WARNING: Unrecognized type specified in parameter file.\n";
        std::cout << "    Type " << type << " is not currently supported.\n";
        std::cout << "    Currently supported types are:\n";
        std::cout << "        string, int, double, bool, vector<string>\n";
        std::cout << "        vector<double>, and vector<int>" << std::endl;
        return false;
    }
    
    return true;
}

void parameters::readParams(char *file) {
    std::ifstream fin;
    std::string line, type, key, equal, val;
    bool check = true;
    
    fin.open(file, std::ios::in);
    while (std::getline(fin, line) && check) {
        std::istringstream iss(line);
        iss >> type >> key >> equal >> val;
        if (type != "#") {
            check = parameters::assignParams(type, key, val);
        } 
    }
    fin.close();
    
    if (!check) {
        std::stringstream message;
        message << "ERROR: All parameters have not been assigned" << std::endl;
        throw std::runtime_error(message.str());
    }
}

void parameters::print() {
    std::map<std::string, std::string>::iterator it_string = parameters::strings.begin();
    std::map<std::string, int>::iterator it_int = parameters::ints.begin();
    std::map<std::string, double>::iterator it_double = parameters::doubles.begin();
    std::map<std::string, bool>::iterator it_bool = parameters::bools.begin();
    std::map<std::string, std::vector<double> >::iterator it_dvectors = 
                                                        parameters::dvectors.begin();
    std::map<std::string, std::vector<int> >::iterator it_ivectors = 
                                                        parameters::ivectors.begin();
    std::map<std::string, std::vector<std::string> >::iterator it_svectors =
                                                        parameters::svectors.begin();
    std::map<std::string, std::vector<bool> >::iterator it_bvectors = 
                                                        parameters::bvectors.begin();
    
    for (it_string = parameters::strings.begin(); it_string != parameters::strings.end(); ++it_string)
        std::cout << "string " << it_string->first << " = " << it_string->second << std::endl;
    for (it_int = parameters::ints.begin(); it_int != parameters::ints.end(); ++it_int)
        std::cout << "int " << it_int->first << " = " << it_int->second << std::endl;
    for (it_double = parameters::doubles.begin(); it_double != parameters::doubles.end(); ++it_double)
        std::cout << "double " << it_double->first << " = " << it_double->second << std::endl;
    for (it_bool = parameters::bools.begin(); it_bool != parameters::bools.end(); ++it_bool)
        std::cout << "bool " << it_bool->first << " = " << std::boolalpha << it_bool->second << std::endl;
    for (it_dvectors = parameters::dvectors.begin(); it_dvectors != parameters::dvectors.end(); ++it_dvectors) {
        int numVals = parameters::dvectors[it_dvectors->first].size();
        std::cout << "vector<double> " << it_dvectors->first << " = ";
        for (int i = 0; i < numVals; ++i) {
            std::cout << parameters::dvectors[it_dvectors->first][i];
            if (i != numVals-1) std::cout << ",";
        }
        std::cout << std::endl;
    }
    for (it_ivectors = parameters::ivectors.begin(); it_ivectors != parameters::ivectors.end(); ++it_ivectors) {
        int numVals = parameters::ivectors[it_ivectors->first].size();
        std::cout << "vector<int> " << it_ivectors->first << " = ";
        for (int i = 0; i < numVals; ++i) {
            std::cout << parameters::ivectors[it_ivectors->first][i];
            if (i != numVals-1) std::cout << ",";
        }
        std::cout << std::endl;
    }
    for (it_svectors = parameters::svectors.begin(); it_svectors != parameters::svectors.end(); ++it_svectors) {
        int numVals = parameters::svectors[it_svectors->first].size();
        std::cout << "vector<string> " << it_svectors->first << " = ";
        for (int i = 0; i < numVals; ++i) {
            std::cout << parameters::svectors[it_svectors->first][i];
            if (i != numVals-1) std::cout << ",";
        }
        std::cout << std::endl;
    }
    for (it_bvectors = parameters::bvectors.begin(); it_bvectors != parameters::bvectors.end(); ++it_bvectors) {
        int numVals = parameters::bvectors[it_bvectors->first].size();
        std::cout << "vector<bool> " << it_bvectors->first << " = ";
        for (int i = 0; i < numVals; ++i) {
            std::cout << std::boolalpha << parameters::bvectors[it_bvectors->first][i];
            if (i != numVals-1) std::cout << ",";
        }
        std::cout << std::endl;
    }
}

void parameters::check_min(std::vector<typekey> neededParams) {
    int minNum = neededParams.size();
    int count = 0;
                                                        
    for (int i = 0; i < minNum; ++i) {
        if (neededParams[i].type == "int") {
            if (parameters::ints.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "double") {
            if (parameters::doubles.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "string") {
            if (parameters::strings.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "bool") {
            if (parameters::bools.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "vector<double>") {
            if (parameters::dvectors.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "vector<int>") {
            if (parameters::ivectors.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "vector<string>") {
            if (parameters::svectors.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "vector<bool>") {
            if (parameters::bvectors.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        }
    }
    
    if (count != minNum) {
        std::stringstream message;
        message << "ERROR: Minimum parameters not found." << std::endl;
        throw std::runtime_error(message.str());
    }
}

double parameters::getd(std::string key, int element) {
    if (parameters::ints.count(key) == 1) {
        return double(parameters::ints[key]);
    } else if (parameters::doubles.count(key) == 1) {
        return parameters::doubles[key];
    } else if (parameters::dvectors.count(key) == 1) {
        return parameters::dvectors[key][element];
    } else if (parameters::ivectors.count(key) == 1) {
        return double(parameters::ivectors[key][element]);
    } else if (parameters::strings.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a string not a numeric type.\n";
        message << "Use the gets() function instead of getd()." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::bools.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a boolean not a numeric type.\n";
        message << "Use the getb() function instead of getd()." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::svectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a vector<string> not a numeric type.\n";
        message << "Use the gets() function instead of getd()." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::bvectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a vector<bool> not a numeric type." << std::endl;
        throw std::runtime_error(message.str());
    } else {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " does not exist in the parameter file.\n";
        message << "Add " << key << " to your parameter file as a numeric type." << std::endl;
        throw std::runtime_error(message.str());
    }
}

int parameters::geti(std::string key, int element) {
    if (parameters::ints.count(key) == 1) {
        return parameters::ints[key];
    } else if (parameters::doubles.count(key) == 1) {
        return int(parameters::doubles[key]);
    } else if (parameters::dvectors.count(key) == 1) {
        return int(parameters::dvectors[key][element]);
    } else if (parameters::ivectors.count(key) == 1) {
        return parameters::ivectors[key][element];
    } else if (parameters::strings.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a string not a numeric type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::bools.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a boolean not a numeric type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::svectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a vector<string> not a numeric type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::bvectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a vector<bool> not a numeric type." << std::endl;
        throw std::runtime_error(message.str());
    } else {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " does not exist in the parameter file." << std::endl;
        throw std::runtime_error(message.str());
    }
}

bool parameters::getb(std::string key, int element) {
    if (parameters::bools.count(key) == 1) {
        return parameters::bools[key];
    } else if (parameters::bvectors.count(key) == 1) {
        return parameters::bvectors[key][element];
    } else if (parameters::ints.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is an int not a boolean type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::doubles.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is an double not a boolean type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::dvectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is an vector<double> not a boolean type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::ivectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is an vector<int> not a boolean type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::strings.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a string not a boolean type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::svectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a vector<string> not a boolean type." << std::endl;
        throw std::runtime_error(message.str());
    } else {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " does not exist in the parameter file." << std::endl;
        throw std::runtime_error(message.str());
    }
}

std::string parameters::gets(std::string key, int element) {
    if (parameters::strings.count(key) == 1) {
        return parameters::strings[key];
    } else if (parameters::svectors.count(key) == 1) {
        return parameters::svectors[key][element];
    } else if (parameters::ints.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is an int not a string type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::doubles.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a double not a string type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::dvectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a vector<double> not a string type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::ivectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a vector<int> not a string type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::bools.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a boolean not a string type." << std::endl;
        throw std::runtime_error(message.str());
    } else if (parameters::bvectors.count(key) == 1) {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is a vector<bool> not a string type." << std::endl;
        throw std::runtime_error(message.str());
    } else {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " does not exist in the parameter file." << std::endl;
        throw std::runtime_error(message.str());
    }
}

bool parameters::checkParam(std::string key) {
    if (parameters::strings.count(key) == 1) {
        return true;
    } else if (parameters::ints.count(key) == 1) {
        return true;
    } else if (parameters::doubles.count(key) == 1) { 
        return true;
    } else if (parameters::bools.count(key) == 1) {
        return true;
    } else if (parameters::dvectors.count(key) == 1) {
        return true;
    } else if (parameters::ivectors.count(key) == 1) {
        return true;
    } else if (parameters::svectors.count(key) == 1) {
        return true;
    } else if (parameters::bvectors.count(key) == 1) {
        return true;
    } else {
        return false;
    }
}
