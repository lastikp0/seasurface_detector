#include "tracker.hpp"
#include <vector>

SimpleTracker::SimpleTracker(int max_missed) : max_missed_(max_missed) {}

float SimpleTracker::iou(const cv::Rect& a, const cv::Rect& b) {
    int interArea = (a & b).area();
    int unionArea = a.area() + b.area() - interArea;
    if (unionArea <= 0) return 0.0f;
    return static_cast<float>(interArea) / static_cast<float>(unionArea);
}

void SimpleTracker::update(std::vector<Detection>& dets, float iou_threshold) {
    for (auto& kv : tracks_) kv.second.missed++;

    for (auto& d : dets) {
        int best_id = -1;
        float best_iou = 0.0f;

        for (const auto& kv : tracks_) {
            float v = iou(d.bbox, kv.second.bbox);
            if (v > best_iou) { best_iou = v; best_id = kv.first; }
        }

        if (best_id >= 0 && best_iou >= iou_threshold) {
            d.track_id = best_id;
            tracks_[best_id].bbox = d.bbox;
            tracks_[best_id].missed = 0;
        } else {
            int id = next_id_++;
            d.track_id = id;
            tracks_[id] = Track{d.bbox, 0};
        }
    }

    std::vector<int> remove_ids;
    remove_ids.reserve(tracks_.size());
    for (const auto& kv : tracks_) {
        if (kv.second.missed > max_missed_) remove_ids.push_back(kv.first);
    }
    for (int id : remove_ids) tracks_.erase(id);
}
