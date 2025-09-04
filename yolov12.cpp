#include "yolov12.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

YoloDetector::YoloDetector(const std::string& modelPath,
                           float confThreshold,
                           float nmsThreshold,
                           cv::Size inputSize,
                           bool useCuda)
    : confThreshold(confThreshold), nmsThreshold(nmsThreshold), inputSize(inputSize)
{
    // Check if model is ONNX or other format
    if (modelPath.find(".onnx") != string::npos) {
        net = readNetFromONNX(modelPath);
    } else {
        // Add support for other model formats if needed
        throw std::runtime_error("Unsupported model format. Only ONNX is supported.");
    }
    
    if (net.empty()) {
        throw std::runtime_error("Failed to load model: " + modelPath);
    }
    
    // Set backend preferences
    if (useCuda && cuda::getCudaEnabledDeviceCount() > 0) {
        try {
            net.setPreferableBackend(DNN_BACKEND_CUDA);
            net.setPreferableTarget(DNN_TARGET_CUDA);
            cout << "Using CUDA acceleration" << endl;
        } catch (const Exception& e) {
            cerr << "CUDA not available, falling back to CPU: " << e.what() << endl;
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
        }
    } else {
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        cout << "Using CPU" << endl;
    }
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    std::vector<Detection> detections;
    if (frame.empty()) {
        return detections;
    }

    // Prepare input blob
    Mat blob, input;
    Size targetSize = (inputSize.width == 0 || inputSize.height == 0) ? frame.size() : inputSize;
    
    // Resize while maintaining aspect ratio (letterbox)
    int maxSize = max(targetSize.width, targetSize.height);
    double ratio = min(static_cast<double>(maxSize) / frame.cols, 
                      static_cast<double>(maxSize) / frame.rows);
    Size newSize(static_cast<int>(frame.cols * ratio), 
                 static_cast<int>(frame.rows * ratio));
    
    resize(frame, input, newSize);
    
    // Create a black canvas
    Mat canvas = Mat::zeros(maxSize, maxSize, CV_8UC3);
    input.copyTo(canvas(Rect(0, 0, newSize.width, newSize.height)));
    
    // Create blob from image
    blobFromImage(canvas, blob, 1.0/255.0, Size(maxSize, maxSize), Scalar(0, 0, 0), true, false);
    
    // Set input
    net.setInput(blob);
    
    // Forward pass
    std::vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    if (outputs.empty()) {
        return detections;
    }
    
    // Parse outputs based on YOLO version
    int dimensions = outputs[0].size[1];
    int rows = outputs[0].size[2];
    
    // Determine YOLO version based on output shape
    bool isYoloV8Style = (dimensions > rows);
    
    if (isYoloV8Style) {
        // YOLOv8/v12 style: [batch, channels, num_boxes]
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];
        outputs[0] = outputs[0].reshape(1, dimensions);
        transpose(outputs[0], outputs[0]);
    } else {
        // YOLOv5 style: [batch, num_boxes, channels]
        outputs[0] = outputs[0].reshape(1, outputs[0].size[1]);
    }
    
    float* data = (float*)outputs[0].data;
    
    // Calculate scaling factors
    float xFactor = static_cast<float>(frame.cols) / maxSize;
    float yFactor = static_cast<float>(frame.rows) / maxSize;
    float ratioFactor = static_cast<float>(maxSize) / std::max(frame.cols, frame.rows);
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    
    // Parse detections
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        
        if (confidence >= confThreshold) {
            // Get class scores
            Mat scores(1, dimensions - 4, CV_32FC1, data + 4);
            Point classIdPoint;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
            
            if (maxClassScore >= confThreshold) {
                float centerX = data[0] / ratioFactor;
                float centerY = data[1] / ratioFactor;
                float width = data[2] / ratioFactor;
                float height = data[3] / ratioFactor;
                
                int left = static_cast<int>((centerX - width / 2) * xFactor);
                int top = static_cast<int>((centerY - height / 2) * yFactor);
                int w = static_cast<int>(width * xFactor);
                int h = static_cast<int>(height * yFactor);
                
                // Ensure bounding box is within image boundaries
                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                w = max(1, min(w, frame.cols - left));
                h = max(1, min(h, frame.rows - top));
                
                boxes.push_back(Rect(left, top, w, h));
                confidences.push_back(static_cast<float>(maxClassScore));
                classIds.push_back(classIdPoint.x);
            }
        }
        data += dimensions;
    }
    
    // Apply NMS
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    // Create detection results
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = classIds[idx];
        detections.push_back(det);
    }
    
    return detections;
}

void YoloDetector::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    // Define colors for different classes
    vector<Scalar> colors = {
        Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 0, 0),
        Scalar(255, 255, 0), Scalar(255, 0, 255), Scalar(0, 255, 255),
        Scalar(128, 128, 0), Scalar(128, 0, 128), Scalar(0, 128, 128)
    };
    
    for (const auto& det : detections) {
        // Choose color based on class ID
        Scalar color = colors[det.classId % colors.size()];
        
        // Draw bounding box
        rectangle(frame, det.box, color, 2);
        
        // Create label
        string label = format("Class:%d Conf:%.2f", det.classId, det.confidence);
        
        // Display label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int top = max(det.box.y, labelSize.height);
        rectangle(frame, Point(det.box.x, top - labelSize.height),
                  Point(det.box.x + labelSize.width, top + baseLine), color, FILLED);
        putText(frame, label, Point(det.box.x, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }
}
