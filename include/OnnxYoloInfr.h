#ifndef ONNX_YOLO_INFR_H
#define ONNX_YOLO_INFR_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "DropOldQueue.h"
#include "FdThreadPool.h"
#include "ThreadDecoder.h"
#include <vector>
#include <thread>
#include <atomic>
#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <arpa/inet.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

inline constexpr size_t MAX_BATCH = 8;
inline constexpr size_t MAX_WAIT_MS_ = 8;

struct Detection {
    int x, y, w, h;    // 框的左上角坐标和宽高(真实坐标)
    float conf;          // 置信度
    int class_id;        // 类别ID
};

struct preImgInf
{
    float scale;
    int dx;
    int dy;
};

struct InferResult {
    int fd;
    std::vector<Detection> det;
    uint64_t timestamp;
};

struct preImg
{
    cv::Mat blob;   // 预处理图像
    preImgInf preInf;   // 预处理信息
};


class OnnxYoloInfr
{
private:
    ThreadDecoder* td_;

    Ort::Env env_;
    Ort::Session session_;

    std::vector<const char*> input_names_ = {"images"};
    std::vector<const char*> output_names_ = {"output0"};

    std::vector<std::thread> workers_; // 工作线程
    std::condition_variable c_v_;
    std::mutex mtx_w;
    std::atomic<bool> running_ = {true};

    // 缓存帧 buffer，用vector要手动锁
    std::mutex mtx_b;
    std::vector<VideoFrame> buffer_;

    DropOldQueue<InferResult> result_queue_; // 输出队列

    void infer_batch(); // 推理

    preImg preprocess(const cv::Mat& img);// 预处理

    std::vector<std::vector<Detection>> postprocess(
        std::vector<Ort::Value>& output_tensors,
        const std::vector<preImgInf>& preInfs, 
        float conf_thresh = 0.5f
    );// 后处理：模型输出 → 检测框

    void nms(std::vector<Detection>& dets, float iou_thresh = 0.45f);// NMS

    void dynamicBatchInferThread(); // 动态batch推理线程

    void sendResultThread(); //发送结果线程

public:
    //初始化 session_
    OnnxYoloInfr(const char* model_path, ThreadDecoder* td,int device_id = 0);
    ~OnnxYoloInfr();

    DropOldQueue<InferResult>& get_output_queue() { return result_queue_; }
    
    void run();// 启动推理和发送工作线程

};


#endif // 
