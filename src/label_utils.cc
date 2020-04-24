#include "label_utils.h"

#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>

using std::getline;
using std::ifstream;
using std::istringstream;
using std::map;
using std::regex;
using std::regex_replace;
using std::string;

namespace edge {

map<int, string> ParseLabel(const string& label_path) {
  map<int, string> ret;
  ifstream label_file(label_path);
  if (!label_file.good()) return ret;
  for (string line; getline(label_file, line);) {
    istringstream ss(line);
    int id;
    ss >> id;
    line = regex_replace(line, regex("^[0-9]+ +"), "");
    ret.emplace(id, line);
  }
  return ret;
}

}  // namespace edge
