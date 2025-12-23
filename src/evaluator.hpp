#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include "detector.hpp"

struct EvalResult {
    long long tp = 0;
    long long fp = 0;
    long long fn = 0;
    double precision = 0.0;
    double recall = 0.0;
};

class Evaluator {
public:
    Evaluator(const std::string& gt_csv_path, bool class_agnostic);

    void add_frame(const std::string& source, int frame,
                   const std::vector<Detection>& preds,
                   float iou_threshold);

    EvalResult finalize() const;

private:
    struct GTBox {
        int class_id;
        cv::Rect bbox;
        bool matched = false;
    };

    bool class_agnostic_ = true;
    std::unordered_map<std::string, std::vector<GTBox>> gt_;
    long long tp_ = 0, fp_ = 0, fn_ = 0;

    static float iou(const cv::Rect& a, const cv::Rect& b);
    static std::string key(const std::string& source, int frame);

    void load_gt_csv(const std::string& path);
};
