#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

struct Detection {
    int class_id = 0;
    std::string class_name = "swimmer";
    float confidence = 1.0f;
    cv::Rect bbox;
    int track_id = -1;
};

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& bgr) = 0;
};

class DummyDetector final : public IDetector {
public:
    std::vector<Detection> detect(const cv::Mat& bgr) override;
};

namespace Ort { class Session; }

class YoloOnnxDetector final : public IDetector {
public:
    struct Params {
        int imgsz = 640;
        float conf_thr = 0.25f;
        float nms_thr  = 0.45f;
        bool use_gpu = false;
    };

    YoloOnnxDetector(const std::string& onnx_path,
                     std::vector<std::string> class_names,
                     Params p);
    
    ~YoloOnnxDetector();

    std::vector<Detection> detect(const cv::Mat& bgr) override;

    static std::vector<std::string> load_class_names(const std::string& path);

private:
    Params p_;
    std::vector<std::string> class_names_;

    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;

    static cv::Mat letterbox(const cv::Mat& src, int new_w, int new_h,
                             float& scale, int& pad_w, int& pad_h);

    void decode_output(const cv::Mat& out,
                       std::vector<cv::Rect>& boxes,
                       std::vector<float>& scores,
                       std::vector<int>& class_ids) const;

    static void nms_classwise(const std::vector<cv::Rect>& boxes,
                              const std::vector<float>& scores,
                              const std::vector<int>& class_ids,
                              float conf_thr,
                              float nms_thr,
                              std::vector<int>& kept_indices);
};
