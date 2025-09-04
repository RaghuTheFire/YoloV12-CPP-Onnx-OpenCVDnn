Usage Instructions

Compile the code:

bash
g++ -std=c++11 -o yolov12_rtsp main.cpp yolov12.cpp `pkg-config --cflags --libs opencv4`

Run with default parameters:

bash
./yolov12_rtsp

Run with custom parameters:

bash
./yolov12_rtsp "rtsp://admin:password@192.168.1.100:554/stream" "yolov12.onnx" true
