#include "utils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <string>
#include <vector>

namespace utils {

std::string to_lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);

    return s;
}

void ensure_dir(const std::filesystem::path& p) {
    if (p.empty()) return;
    std::error_code ec;
    if (std::filesystem::exists(p, ec)) return;
    if (!std::filesystem::create_directories(p, ec)) {
        if (ec) throw std::runtime_error("Failed to create directory: " + p.string() + " : " + ec.message());
    }
}

double sum(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;

    return s;
}

std::string format_float(double v, int digits) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(digits) << v;
    
    return oss.str();
}

std::filesystem::path expand_user_path(const std::filesystem::path& p) {
    const std::string path = p.string();

    if (path.empty() || path[0] != '~') return std::filesystem::path(path);

    const char* home = std::getenv("HOME");
    if (!home || !*home) return std::filesystem::path(path);

    if (path == "~") return std::filesystem::path(std::string(home));
    if (path.size() >= 2 && path[1] == '/') {
        return std::filesystem::path(std::string(home) + path.substr(1));
    }

    return std::filesystem::path(path);
}

}