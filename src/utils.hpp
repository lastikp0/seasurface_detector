#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace utils {
std::string to_lower(std::string s);
void ensure_dir(const std::filesystem::path& p);

double sum(const std::vector<double>& v);
std::string format_float(double v, int digits);

std::filesystem::path expand_user_path(const std::filesystem::path& p);
}