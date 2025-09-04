#include "yolov12.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;

// Function to handle RTSP stream processing
void processRTSPStream(const string& rtspUrl, const string& modelPath, bool useCuda) 
{
    try {
        // Initialize YOLO detector
        YoloDetector detector(modelPath, 0.5, 0.4, Size(640, 640), useCuda);
        
        // Open RTSP stream
        VideoCapture cap(rtspUrl);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open RTSP stream: " << rtspUrl << endl;
            return;
        }
        
        cout << "Successfully opened RTSP stream: " << rtspUrl << endl;
        cout << "Press 'q' to quit, 'p' to pause" << endl;
        
        Mat frame;
        bool paused = false;
        int frameCount = 0;
        auto startTime = chrono::steady_clock::now();
        
        while (true) {
            if (!paused) {
                // Read frame from stream
                if (!cap.read(frame)) {
                    cerr << "Error: Failed to read frame from stream" << endl;
                    break;
                }
                
                // Run object detection
                auto detectStart = chrono::steady_clock::now();
                vector<Detection> detections = detector.detect(frame);
                auto detectEnd = chrono::steady_clock::now();
                
                // Calculate FPS and detection time
                frameCount++;
                auto currentTime = chrono::steady_clock::now();
                double elapsedSeconds = chrono::duration<double>(currentTime - startTime).count();
                double fps = frameCount / elapsedSeconds;
                double detectionTime = chrono::duration<double, milli>(detectEnd - detectStart).count();
                
                // Draw detections on frame
                YoloDetector::drawDetections(frame, detections);
                
                // Display FPS and detection time
                string fpsText = format("FPS: %.2f", fps);
                string detectionTimeText = format("Detection: %.2f ms", detectionTime);
                putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                putText(frame, detectionTimeText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
                // Display frame
                imshow("RTSP Stream with YOLO Detection", frame);
            }
            
            // Handle keyboard input
            int key = waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' or ESC to quit
                break;
            } else if (key == 'p') { // 'p' to pause/resume
                paused = !paused;
                cout << (paused ? "Paused" : "Resumed") << endl;
            }
        }
        
        cap.release();
        destroyAllWindows();
        
    } 
    catch (const exception& e) 
      {
        cerr << "Error: " << e.what() << endl;
    }
}

int main(int argc, char** argv) 
{
    // Default values
    string rtspUrl = "rtsp://username:password@your_camera_ip:554/stream";
    string modelPath = "yolov12.onnx";
    bool useCuda = false;
    
    // Parse command line arguments
    if (argc > 1) rtspUrl = argv[1];
    if (argc > 2) modelPath = argv[2];
    if (argc > 3) useCuda = string(argv[3]) == "true";
    
    cout << "RTSP URL: " << rtspUrl << endl;
    cout << "Model Path: " << modelPath << endl;
    cout << "Use CUDA: " << (useCuda ? "Yes" : "No") << endl;
    
    // Process RTSP stream
    processRTSPStream(rtspUrl, modelPath, useCuda);
    
    return 0;
}
