#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include "detector.hpp"

class SimpleTracker {
public:
    explicit SimpleTracker(int max_missed = 30);
    void update(std::vector<Detection>& dets, float iou_threshold);

private:
    struct Track { cv::Rect bbox; int missed = 0; };

    int next_id_ = 0;
    int max_missed_ = 30;
    std::unordered_map<int, Track> tracks_;

    static float iou(const cv::Rect& a, const cv::Rect& b);
};
