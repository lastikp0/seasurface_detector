#pragma once

#include <fstream>
#include <string>

#include "detector.hpp"

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path);
    ~CsvWriter();

    void write_detection(const std::string& source, int frame, const Detection& d);
    void close();

private:
    std::ofstream out_;
};
