#include "csv.hpp"

#include <stdexcept>
#include <fstream>
#include <string>

CsvWriter::CsvWriter(const std::string& path) {
    out_.open(path);
    if (!out_) throw std::runtime_error("Failed to open CSV for writing: " + path);
    out_ << "source,frame,track_id,class_id,class_name,conf,x,y,w,h\n";
}

CsvWriter::~CsvWriter() { close(); }

void CsvWriter::write_detection(const std::string& source, int frame, const Detection& d) {
    if (!out_) return;
    out_ << source << ","
         << frame << ","
         << d.track_id << ","
         << d.class_id << ","
         << d.class_name << ","
         << d.confidence << ","
         << d.bbox.x << ","
         << d.bbox.y << ","
         << d.bbox.width << ","
         << d.bbox.height << "\n";
}

void CsvWriter::close() {
    if (out_.is_open()) out_.close();
}
