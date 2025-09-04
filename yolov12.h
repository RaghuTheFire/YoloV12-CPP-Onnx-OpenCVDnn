#ifndef YOLOV12_H
#define YOLOV12_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

struct Detection {
  cv::Rect box;
  int classId;
  float confidence;
};

class YoloDetector {
public:
  explicit YoloDetector(const std::string& modelPath,
               float confThreshold = 0.5,
               float nmsThreshold = 0.4,
               cv::Size inputSize = cv::Size(640, 640),
               bool useCuda = false);

  std::vector<Detection> detect(const cv::Mat& frame);

  static void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

private:
  cv::dnn::Net net;
  float confThreshold;
  float nmsThreshold;
  cv::Size inputSize;
};

#endif // YOLOV12_H
