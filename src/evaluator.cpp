#include "evaluator.hpp"
#include <fstream>
#include <stdexcept>

static std::vector<std::string> split_csv_line(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

Evaluator::Evaluator(const std::string& gt_csv_path, bool class_agnostic)
    : class_agnostic_(class_agnostic) {
    load_gt_csv(gt_csv_path);
}

std::string Evaluator::key(const std::string& source, int frame) {
    return source + "#" + std::to_string(frame);
}

float Evaluator::iou(const cv::Rect& a, const cv::Rect& b) {
    int interArea = (a & b).area();
    int unionArea = a.area() + b.area() - interArea;
    if (unionArea <= 0) return 0.0f;
    return static_cast<float>(interArea) / static_cast<float>(unionArea);
}

void Evaluator::load_gt_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open GT CSV: " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("GT CSV is empty: " + path);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto cols = split_csv_line(line);
        if (cols.size() < 7) continue;

        const std::string source = cols[0];
        const int frame = std::stoi(cols[1]);
        const int class_id = std::stoi(cols[2]);
        const int x = std::stoi(cols[3]);
        const int y = std::stoi(cols[4]);
        const int w = std::stoi(cols[5]);
        const int h = std::stoi(cols[6]);

        gt_[key(source, frame)].push_back(GTBox{class_id, cv::Rect(x,y,w,h), false});
    }
}

void Evaluator::add_frame(const std::string& source, int frame,
                          const std::vector<Detection>& preds,
                          float iou_threshold) {
    auto it = gt_.find(key(source, frame));
    if (it == gt_.end()) {
        fp_ += (long long)preds.size();
        return;
    }

    auto& gts = it->second;
    for (auto& g : gts) g.matched = false;

    for (const auto& p : preds) {
        int best = -1;
        float best_iou = 0.0f;
        for (int i = 0; i < (int)gts.size(); ++i) {
            if (gts[i].matched) continue;
            if (!class_agnostic_ && gts[i].class_id != p.class_id) continue;
            float v = iou(p.bbox, gts[i].bbox);
            if (v > best_iou) { best_iou = v; best = i; }
        }
        if (best >= 0 && best_iou >= iou_threshold) {
            tp_++;
            gts[best].matched = true;
        } else {
            fp_++;
        }
    }

    for (const auto& g : gts) if (!g.matched) fn_++;
}

EvalResult Evaluator::finalize() const {
    EvalResult r;
    r.tp = tp_; r.fp = fp_; r.fn = fn_;
    r.precision = (tp_ + fp_) ? (double)tp_ / (double)(tp_ + fp_) : 0.0;
    r.recall = (tp_ + fn_) ? (double)tp_ / (double)(tp_ + fn_) : 0.0;
    return r;
}
