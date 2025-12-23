#include "detector.hpp"

#include <opencv2/dnn.hpp>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>

std::vector<Detection> DummyDetector::detect(const cv::Mat& bgr) {
    std::vector<Detection> out;
    if (bgr.empty()) return out;

    const int w = bgr.cols;
    const int h = bgr.rows;

    int bw = std::max(10, static_cast<int>(0.25 * w));
    int bh = std::max(10, static_cast<int>(0.18 * h));
    int x = (w - bw) / 2;
    int y = (h - bh) / 2;

    Detection d;
    d.class_id = 0;
    d.class_name = "swimmer";
    d.confidence = 1.0f;
    d.bbox = cv::Rect(x, y, bw, bh);
    out.push_back(d);
    return out;
}

static std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char)s[a])) a++;
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

std::vector<std::string> YoloOnnxDetector::load_class_names(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open classes file: " + path);

    std::vector<std::string> names;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (!line.empty()) names.push_back(line);
    }
    if (names.empty()) throw std::runtime_error("Classes file is empty: " + path);
    return names;
}

static std::string ort_status_message_and_release(OrtStatus* st) {
    if (!st) return {};
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    std::string msg = api ? api->GetErrorMessage(st) : "ORT status (no api)";
    if (api) api->ReleaseStatus(st);
    return msg;
}

static bool try_enable_ort_cuda(Ort::SessionOptions& so, int device_id = 0) {
    const OrtApiBase* base = OrtGetApiBase();
    if (!base) {
        std::cerr << "WARN: OrtGetApiBase() returned null. Using CPU.\n";
        return false;
    }

    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        std::cerr << "WARN: ORT GetApi(ORT_API_VERSION=" << ORT_API_VERSION
                  << ") returned null. Headers/library mismatch? Using CPU.\n";
        return false;
    }

    if (api->CreateCUDAProviderOptions &&
        api->UpdateCUDAProviderOptions &&
        api->ReleaseCUDAProviderOptions &&
        api->SessionOptionsAppendExecutionProvider_CUDA_V2) {

        OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
        {
            OrtStatus* st = api->CreateCUDAProviderOptions(&cuda_opts);
            if (st) {
                std::cerr << "WARN: CreateCUDAProviderOptions failed: "
                          << ort_status_message_and_release(st) << "Using CPU.\n";
                return false;
            }
        }

        std::string dev = std::to_string(device_id);
        const char* keys[] = {"device_id"};
        const char* vals[] = {dev.c_str()};
        {
            OrtStatus* st = api->UpdateCUDAProviderOptions(cuda_opts, keys, vals, 1);
            if (st) {
                std::cerr << "WARN: UpdateCUDAProviderOptions failed: "
                          << ort_status_message_and_release(st) << "Using CPU.\n";
                api->ReleaseCUDAProviderOptions(cuda_opts);
                return false;
            }
        }

        OrtSessionOptions* raw_so = so;
        {
            OrtStatus* st = api->SessionOptionsAppendExecutionProvider_CUDA_V2(raw_so, cuda_opts);
            if (st) {
                std::cerr << "WARN: SessionOptionsAppendExecutionProvider_CUDA_V2 failed: "
                          << ort_status_message_and_release(st) << "Using CPU.\n";
                api->ReleaseCUDAProviderOptions(cuda_opts);
                return false;
            }
        }

        api->ReleaseCUDAProviderOptions(cuda_opts);
        return true;
    }

    if (api->SessionOptionsAppendExecutionProvider_CUDA) {
        OrtCUDAProviderOptions opts{};
        opts.device_id = device_id;
        OrtSessionOptions* raw_so = so;
        OrtStatus* st = api->SessionOptionsAppendExecutionProvider_CUDA(raw_so, &opts);
        if (st) {
            std::cerr << "WARN: SessionOptionsAppendExecutionProvider_CUDA failed: "
                      << ort_status_message_and_release(st) << "Using CPU.\n";
            return false;
        }
        return true;
    }

    std::cerr << "WARN: This ORT build does not expose CUDA EP append APIs. Using CPU.\n";
    return false;
}

YoloOnnxDetector::~YoloOnnxDetector() = default;

YoloOnnxDetector::YoloOnnxDetector(const std::string& onnx_path,
                                   std::vector<std::string> class_names,
                                   Params p)
    : p_(p), class_names_(std::move(class_names)) {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "seasurface_detector");

    bool cuda_enabled = false;
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (p_.use_gpu) {
        std::cerr << "INFO: Trying to enable ORT CUDA EP...\n";
        cuda_enabled = try_enable_ort_cuda(so, 0);
        if (cuda_enabled) {
            std::cerr << "INFO: CUDA EP ENABLED" << "\n";
        }
    }

    try {
        session_ = std::make_unique<Ort::Session>(env, onnx_path.c_str(), so);
    } catch (const Ort::Exception& e) {
        if (!p_.use_gpu) throw;
        std::cerr << "WARN: ORT session creation failed with GPU settings: " << e.what()
                  << "\nWARN: Falling back to CPU session.\n";
        Ort::SessionOptions cpu_so;
        cpu_so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env, onnx_path.c_str(), cpu_so);
    }

    Ort::AllocatorWithDefaultOptions alloc;
#if ORT_API_VERSION >= 13
    {
        auto in0 = session_->GetInputNameAllocated(0, alloc);
        input_name_ = in0.get();
        auto out0 = session_->GetOutputNameAllocated(0, alloc);
        output_name_ = out0.get();
    }
#else
    {
        char* in0 = session_->GetInputName(0, alloc);
        input_name_ = in0;
        alloc.Free(in0);
        char* out0 = session_->GetOutputName(0, alloc);
        output_name_ = out0;
        alloc.Free(out0);
    }
#endif

    std::cerr << "INFO: ORT session created.\n";
}

cv::Mat YoloOnnxDetector::letterbox(const cv::Mat& src, int new_w, int new_h,
                                   float& scale, int& pad_w, int& pad_h) {
    if (src.empty()) return {};

    scale = std::min((float)new_w / (float)src.cols, (float)new_h / (float)src.rows);
    int rw = (int)std::round(src.cols * scale);
    int rh = (int)std::round(src.rows * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);

    pad_w = (new_w - rw) / 2;
    pad_h = (new_h - rh) / 2;

    cv::Mat out(new_h, new_w, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_w, pad_h, rw, rh)));
    return out;
}

void YoloOnnxDetector::decode_output(const cv::Mat& out,
                                     std::vector<cv::Rect>& boxes,
                                     std::vector<float>& scores,
                                     std::vector<int>& class_ids) const {
    boxes.clear();
    scores.clear();
    class_ids.clear();

    const int nc = (int)class_names_.size();
    const int attrs_noobj = 4 + nc;
    const int attrs_obj   = 5 + nc;

    cv::Mat m;

    if (out.dims == 3) {
        int d1 = out.size[1];
        int d2 = out.size[2];
        cv::Mat as2d(d1, d2, CV_32F, (void*)out.ptr<float>());

        if (d1 == attrs_noobj || d1 == attrs_obj) m = as2d.t();
        else if (d2 == attrs_noobj || d2 == attrs_obj) m = as2d;
        else {
            m = (d1 < d2) ? as2d.t() : as2d;
        }
    } else if (out.dims == 2) {
        if (out.cols == attrs_noobj || out.cols == attrs_obj) m = out;
        else if (out.rows == attrs_noobj || out.rows == attrs_obj) m = out.t();
        else {
            m = (out.rows < out.cols) ? out.t() : out;
        }
    } else {
        throw std::runtime_error("Unexpected output dims: " + std::to_string(out.dims));
    }

    if (!m.isContinuous()) {
        m = m.clone();
    }

    const int attrs = m.cols;
    bool has_obj = (attrs == attrs_obj);
    const int cls_start = has_obj ? 5 : 4;
    const int cls_count = attrs - cls_start;
    if (cls_count <= 0) throw std::runtime_error("Bad output layout (attrs=" + std::to_string(attrs) + ")");

    auto looks_norm = [&](const float* r) {
        return (r[0] <= 2.0f && r[1] <= 2.0f && r[2] <= 2.0f && r[3] <= 2.0f);
    };

    for (int i = 0; i < m.rows; ++i) {
        const float* r = m.ptr<float>(i);

        float x = r[0], y = r[1], w = r[2], h = r[3];
        if (looks_norm(r)) {
            x *= (float)p_.imgsz; y *= (float)p_.imgsz;
            w *= (float)p_.imgsz; h *= (float)p_.imgsz;
        }

        float obj = has_obj ? r[4] : 1.0f;

        int best_c = -1;
        float best_p = 0.0f;
        for (int c = 0; c < cls_count; ++c) {
            float p = r[cls_start + c];
            if (p > best_p) { best_p = p; best_c = c; }
        }

        float score = has_obj ? (obj * best_p) : best_p;
        if (score < p_.conf_thr) continue;

        float left = x - w * 0.5f;
        float top  = y - h * 0.5f;

        boxes.emplace_back((int)std::round(left), (int)std::round(top),
                           (int)std::round(w), (int)std::round(h));
        scores.push_back(score);
        class_ids.push_back(best_c);
    }
}

void YoloOnnxDetector::nms_classwise(const std::vector<cv::Rect>& boxes,
                                    const std::vector<float>& scores,
                                    const std::vector<int>& class_ids,
                                    float conf_thr,
                                    float nms_thr,
                                    std::vector<int>& kept_indices) {
    kept_indices.clear();
    if (boxes.empty()) return;

    std::unordered_map<int, std::vector<int>> by_class;
    by_class.reserve(16);
    for (int i = 0; i < (int)boxes.size(); ++i) by_class[class_ids[i]].push_back(i);

    for (auto& kv : by_class) {
        auto& idxs = kv.second;
        std::vector<cv::Rect> b; b.reserve(idxs.size());
        std::vector<float> s;    s.reserve(idxs.size());
        for (int id : idxs) { b.push_back(boxes[id]); s.push_back(scores[id]); }

        std::vector<int> keep_local;
        cv::dnn::NMSBoxes(b, s, conf_thr, nms_thr, keep_local);
        for (int k : keep_local) kept_indices.push_back(idxs[k]);
    }
}

static std::vector<int64_t> fix_dynamic_shape(std::vector<int64_t> shape, size_t elem_count) {
    int unknown = 0;
    int unknown_idx = -1;
    long long prod = 1;
    for (int i = 0; i < (int)shape.size(); ++i) {
        if (shape[i] <= 0) { unknown++; unknown_idx = i; }
        else prod *= (long long)shape[i];
    }
    if (unknown == 0) return shape;
    if (unknown > 1) return shape;
    if (prod <= 0) return shape;
    if ((long long)elem_count % prod != 0) return shape;
    long long missing = (long long)elem_count / prod;
    if (missing <= 0) return shape;
    shape[unknown_idx] = (int64_t)missing;
    return shape;
}

std::vector<Detection> YoloOnnxDetector::detect(const cv::Mat& bgr) {
    std::vector<Detection> dets;
    if (bgr.empty()) return dets;

    float scale = 1.0f;
    int pad_w = 0, pad_h = 0;
    cv::Mat inp = letterbox(bgr, p_.imgsz, p_.imgsz, scale, pad_w, pad_h);

    cv::Mat blob = cv::dnn::blobFromImage(inp, 1.0 / 255.0,
                                          cv::Size(p_.imgsz, p_.imgsz),
                                          cv::Scalar(), true, false);

    const float* blob_ptr = blob.ptr<float>(0);
    std::vector<float> input(blob_ptr, blob_ptr + (size_t)blob.total());

    std::array<int64_t, 4> input_shape{1, 3, p_.imgsz, p_.imgsz};
    Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        cpu_mem, input.data(), input.size(), input_shape.data(), input_shape.size()
    );

    Ort::RunOptions run_opts{nullptr};

    const char* in_names[]  = { input_name_.c_str() };
    const char* out_names[] = { output_name_.c_str() };

    auto outs = session_->Run(run_opts,
                              in_names, &input_tensor, 1,
                              out_names, 1);
    
    if (outs.empty() || !outs[0].IsTensor()) return dets;

    auto& out0 = outs[0];
    auto info = out0.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();

    size_t elem_count = info.GetElementCount();
    if (elem_count == 0) return dets;

    shape = fix_dynamic_shape(shape, elem_count);

    const float* out_data = out0.GetTensorData<float>();
    std::vector<float> out_copy(out_data, out_data + elem_count);

    cv::Mat out;
    if (shape.size() == 3) {
        if (shape[0] <= 0) shape[0] = 1;
        if (shape[1] <= 0 || shape[2] <= 0) {
            throw std::runtime_error("ORT output has invalid shape rank3: [" +
                                     std::to_string(shape[0]) + "," +
                                     std::to_string(shape[1]) + "," +
                                     std::to_string(shape[2]) + "]");
        }
        int sizes[3] = { (int)shape[0], (int)shape[1], (int)shape[2] };
        out = cv::Mat(3, sizes, CV_32F, (void*)out_copy.data());
    } else if (shape.size() == 2) {
        if (shape[0] <= 0 || shape[1] <= 0) {
            throw std::runtime_error("ORT output has invalid shape rank2: [" +
                                     std::to_string(shape[0]) + "," +
                                     std::to_string(shape[1]) + "]");
        }
        out = cv::Mat((int)shape[0], (int)shape[1], CV_32F, (void*)out_copy.data());
    } else {
        throw std::runtime_error("Unexpected ORT output rank: " + std::to_string(shape.size()));
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    decode_output(out, boxes, scores, class_ids);

    std::vector<int> keep;
    nms_classwise(boxes, scores, class_ids, p_.conf_thr, p_.nms_thr, keep);

    dets.reserve(keep.size());
    for (int idx : keep) {
        cv::Rect r = boxes[idx];

        float x = (r.x - pad_w) / scale;
        float y = (r.y - pad_h) / scale;
        float w = r.width / scale;
        float h = r.height / scale;

        cv::Rect mapped((int)std::round(x), (int)std::round(y),
                        (int)std::round(w), (int)std::round(h));
        mapped &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (mapped.area() <= 0) continue;

        Detection d;
        d.class_id = class_ids[idx];
        d.class_name = (d.class_id >= 0 && d.class_id < (int)class_names_.size())
                       ? class_names_[d.class_id]
                       : ("cls" + std::to_string(d.class_id));
        d.confidence = scores[idx];
        d.bbox = mapped;
        dets.push_back(d);
    }

    return dets;
}
