#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "detector.hpp"
#include "tracker.hpp"
#include "evaluator.hpp"
#include "csv.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

static const char* kKeys =
"{help h usage ? |      | print this message }"
"{input i        |      | path to image / directory / video (required)}"
"{output o       | out  | output directory }"
"{eval           |false | compute Precision/Recall (requires --gt_csv) }"
"{gt_csv         |      | ground truth csv for evaluation }"
"{class_agnostic |true  | if true, ignore class_id in eval }"
"{use_gpu        |false | use GPU for DNN inference if available }"
"{model          |      | path to ONNX model (YOLOv8, requires --classes) }"
"{classes        |      | path to classes.txt (one class name per line) }"
"{imgsz          |640   | inference image size (must match ONNX unless dynamic) }"
"{conf           |0.25  | confidence threshold }"
"{nms            |0.45  | NMS IoU threshold }"
"{track          |true  | enable simple tracking IDs }"
"{max_missed     |30    | tracker max missed frames }"
"{iou            |0.5   | IoU threshold for eval and tracking }";

static bool is_image_ext(const fs::path& p) {
    auto ext = utils::to_lower(p.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

static bool is_video_ext(const fs::path& p) {
    auto ext = utils::to_lower(p.extension().string());
    return ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov";
}

static std::vector<fs::path> list_images(const fs::path& dir) {
    std::vector<fs::path> out;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        if (is_image_ext(e.path())) out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

static cv::Scalar color_for_id(int id) {
    id = std::max(0, id);

    const int hue = (id * 37) % 180;
    const int sat = 220;
    const int val = 255;

    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, sat, val));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    cv::Scalar col(c[0], c[1], c[2]);

    if ((c[0] + c[1] + c[2]) < 60) {
        col = cv::Scalar(0, 255, 0);
    }
    return col;
}

static void draw_detections(cv::Mat& img, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::Rect r = d.bbox & cv::Rect(0,0,img.cols,img.rows);
        cv::rectangle(img, r, color_for_id(d.track_id >= 0 ? d.track_id : d.class_id), 2);

        std::string label = d.class_name;
        if (d.track_id >= 0) label += " id=" + std::to_string(d.track_id);
        label += " " + utils::format_float(d.confidence, 2);

        int baseLine = 0;
        auto textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::Rect bg(r.x, std::max(0, r.y - textSize.height - 6), textSize.width + 6, textSize.height + 6);
        cv::rectangle(img, bg, color_for_id(d.track_id >= 0 ? d.track_id : d.class_id), cv::FILLED);
        cv::putText(img, label, cv::Point(bg.x + 3, bg.y + bg.height - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1, cv::LINE_AA);
    }
}

struct TimingStats {
    std::vector<double> ms;
    void add(double v) { ms.push_back(v); }
    double sum() const { return utils::sum(ms); }
};

static int run_on_image(const fs::path& img_path,
                        const fs::path& out_dir,
                        const std::string& source_name,
                        IDetector& detector,
                        SimpleTracker* tracker,
                        CsvWriter& csv,
                        Evaluator* evaluator,
                        float iou_thr,
                        TimingStats& timing)
{
    cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "ERROR: failed to read image: " << img_path << "\n";
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto dets = detector.detect(img);
    if (tracker) tracker->update(dets, iou_thr);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing.add(ms);

    for (const auto& d : dets) csv.write_detection(source_name, 0, d);
    if (evaluator) evaluator->add_frame(source_name, 0, dets, iou_thr);

    cv::Mat vis = img.clone();
    draw_detections(vis, dets);

    fs::path out_path = out_dir / ("annotated_" + img_path.filename().string());
    if (!cv::imwrite(out_path.string(), vis)) {
        std::cerr << "ERROR: failed to write output image: " << out_path << "\n";
        return 1;
    }
    return 0;
}

static int run_on_video(const fs::path& video_path,
                        const fs::path& out_dir,
                        const std::string& source_name,
                        IDetector& detector,
                        SimpleTracker* tracker,
                        CsvWriter& csv,
                        Evaluator* evaluator,
                        float iou_thr,
                        TimingStats& timing)
{
    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened()) {
        std::cerr << "ERROR: failed to open video: " << video_path << "\n";
        return 1;
    }

    const int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!(fps > 0.0 && std::isfinite(fps))) fps = 30.0;

    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    fs::path out_path = out_dir / ("annotated_" + video_path.filename().string());
    cv::VideoWriter writer(out_path.string(), fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "ERROR: failed to open output video for writing: " << out_path << "\n";
        return 1;
    }

    cv::Mat frame;
    int frame_idx = 0;
    while (true) {
        if (!cap.read(frame)) break;

        auto t0 = std::chrono::steady_clock::now();
        auto dets = detector.detect(frame);
        if (tracker) tracker->update(dets, iou_thr);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing.add(ms);

        for (const auto& d : dets) csv.write_detection(source_name, frame_idx, d);
        if (evaluator) evaluator->add_frame(source_name, frame_idx, dets, iou_thr);

        cv::Mat vis = frame.clone();
        draw_detections(vis, dets);
        writer.write(vis);

        frame_idx++;
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        cv::CommandLineParser parser(argc, argv, kKeys);
        parser.about("seasurface_detector (C++17/OpenCV) - maritime surface object detection");
        if (parser.has("help") || argc == 1) {
            parser.printMessage();
            return 0;
        }

        const std::string input_raw = parser.get<std::string>("input");
        const std::string output_raw = parser.get<std::string>("output");

        const fs::path in_path = utils::expand_user_path(fs::path(input_raw));
        const fs::path out_dir = utils::expand_user_path(fs::path(output_raw));

        const bool eval = parser.get<bool>("eval");
        const bool class_agnostic = parser.get<bool>("class_agnostic");
        const bool use_gpu = parser.get<bool>("use_gpu");
        const std::string model_raw = parser.get<std::string>("model");
        const std::string classes_raw = parser.get<std::string>("classes");
        const int imgsz = parser.get<int>("imgsz");
        const float conf_thr = parser.get<float>("conf");
        const float nms_thr  = parser.get<float>("nms");

        if (imgsz <= 0) { std::cerr << "ERROR: --imgsz must be > 0\n"; return 2; }
        if (conf_thr < 0.0f || conf_thr > 1.0f) { std::cerr << "ERROR: --conf must be in [0,1]\n"; return 2; }
        if (nms_thr < 0.0f || nms_thr > 1.0f) { std::cerr << "ERROR: --nms must be in [0,1]\n"; return 2; }

        const bool track = parser.get<bool>("track");
        const int max_missed = parser.get<int>("max_missed");
        const float iou_thr = parser.get<float>("iou");

        if (input_raw.empty()) { std::cerr << "ERROR: --input is required\n"; return 2; }
        if (iou_thr <= 0.0f || iou_thr > 1.0f) { std::cerr << "ERROR: --iou must be in (0,1]\n"; return 2; }
        if (max_missed < 0) { std::cerr << "ERROR: --max_missed must be >= 0\n"; return 2; }

        utils::ensure_dir(out_dir);

        fs::path csv_path = out_dir / "dets.csv";

        utils::ensure_dir(csv_path.parent_path());

        CsvWriter csv(csv_path.string());

        std::unique_ptr<Evaluator> evaluator;
        std::string gt_csv_raw = parser.get<std::string>("gt_csv");
        if (eval) {
            if (gt_csv_raw.empty() || gt_csv_raw == "true" || gt_csv_raw == "false") {
                std::cerr << "ERROR: --eval true requires --gt_csv=<path>\n";
                return 2;
            }
            fs::path gt_csv_path = utils::expand_user_path(fs::path(gt_csv_raw));
            if (!fs::exists(gt_csv_path)) {
                std::cerr << "ERROR: gt_csv path does not exist: " << gt_csv_path << "\n";
                return 2;
            }
            evaluator = std::make_unique<Evaluator>(gt_csv_path.string(), class_agnostic);
        }

        std::unique_ptr<IDetector> detector;

        if (!model_raw.empty()) {
            fs::path model_path = utils::expand_user_path(fs::path(model_raw));
            if (!fs::exists(model_path)) {
                std::cerr << "ERROR: model path does not exist: " << model_path << "\n";
                return 2;
            }
        
            fs::path classes_path;
            if (!classes_raw.empty()) {
                classes_path = utils::expand_user_path(fs::path(classes_raw));
            } else {
                classes_path = model_path.parent_path() / "classes.txt";
            }
        
            if (!fs::exists(classes_path)) {
                std::cerr << "ERROR: classes file not found: " << classes_path
                          << " (use --classes=path/to/classes.txt)\n";
                return 2;
            }
        
            auto names = YoloOnnxDetector::load_class_names(classes_path.string());
        
            YoloOnnxDetector::Params p;
            p.imgsz = imgsz;
            p.conf_thr = conf_thr;
            p.nms_thr = nms_thr;
            p.use_gpu = use_gpu;
        
            detector = std::make_unique<YoloOnnxDetector>(model_path.string(), std::move(names), p);
            std::cerr << "INFO: Using ONNX detector: " << model_path << "\n";
        } else {
            detector = std::make_unique<DummyDetector>();
            std::cerr << "INFO: Using Dummy detector (no --model provided)\n";
        }


        std::unique_ptr<SimpleTracker> tracker_ptr;
        if (track) tracker_ptr = std::make_unique<SimpleTracker>(max_missed);

        TimingStats timing;

        if (!fs::exists(in_path)) { std::cerr << "ERROR: input path does not exist: " << in_path << "\n"; return 2; }

        int rc = 0;
        if (fs::is_directory(in_path)) {
            auto imgs = list_images(in_path);
            if (imgs.empty()) { std::cerr << "ERROR: no supported images found in directory: " << in_path << "\n"; return 2; }
            for (const auto& p : imgs) {
                rc = run_on_image(p, out_dir, p.filename().string(), *detector, tracker_ptr.get(), csv, evaluator.get(), iou_thr, timing);
                if (rc != 0) return rc;
            }
        } else {
            if (is_image_ext(in_path)) {
                rc = run_on_image(in_path, out_dir, in_path.filename().string(), *detector, tracker_ptr.get(), csv, evaluator.get(), iou_thr, timing);
            } else if (is_video_ext(in_path)) {
                rc = run_on_video(in_path, out_dir, in_path.filename().string(), *detector, tracker_ptr.get(), csv, evaluator.get(), iou_thr, timing);
            } else {
                cv::Mat probe = cv::imread(in_path.string(), cv::IMREAD_COLOR);
                if (!probe.empty()) {
                    rc = run_on_image(in_path, out_dir, in_path.filename().string(), *detector, tracker_ptr.get(), csv, evaluator.get(), iou_thr, timing);
                } else {
                    cv::VideoCapture cap(in_path.string());
                    if (cap.isOpened()) {
                        rc = run_on_video(in_path, out_dir, in_path.filename().string(), *detector, tracker_ptr.get(), csv, evaluator.get(), iou_thr, timing);
                    } else {
                        std::cerr << "ERROR: unsupported or unreadable input format: " << in_path << "\n";
                        return 2;
                    }
                }
            }
        }

        csv.close();

        std::cout << "Done.\n";
        std::cout << "Output dir: " << out_dir << "\n";
        std::cout << "Detections CSV: " << csv_path << "\n";

        if (!timing.ms.empty()) {
            std::cout << "Total time (ms): " << utils::format_float(timing.sum(), 3)
            << "\nTotal frames / images: " << timing.ms.size()
            << "\nAverage time per frame / image (ms): "
            << utils::format_float(timing.sum() / (float)timing.ms.size(), 3) << "\n";
        }

        if (evaluator) {
            auto res = evaluator->finalize();
            std::cout << "Eval (IoU=" << iou_thr << "): "
            << "TP=" << res.tp << " FP=" << res.fp << " FN=" << res.fn
            << " Precision=" << utils::format_float(res.precision, 4)
            << " Recall=" << utils::format_float(res.recall, 4) << "\n";
        }

        return rc;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
