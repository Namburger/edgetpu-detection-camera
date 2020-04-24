#ifndef EDGE_LABEL_UTIL_H
#define EDGE_LABEL_UTIL_H

#include <fstream>
#include <map>
#include <regex>
#include <string>

namespace edge {

std::map<int, std::string> ParseLabel(const std::string& label_path);

}  // namespace edge

#endif
