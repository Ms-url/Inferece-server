#ifndef ONNX_YOLO_INFR_H
#define ONNX_YOLO_INFR_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "DropOldQueue.h"
#include "FdThreadPool.h"
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
inline constexpr int NUM_DECODE_THREADS_ = 4;
inline constexpr int MAX_NUM_DECODERS_ = 10;

// 跨平台字节交换
static inline uint64_t ntohll_custom(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(x);
#else
    return __bswap_64(x);   // 需 <byteswap.h>
#endif
}

// 传输头
struct PacketHeader {
    uint32_t nalu_len;   // 网络序
    uint8_t  flags;      // 0x01=IDR, 0x02=config, 0x00=normal
    uint64_t timestamp;  // 毫秒, 网络序
} __attribute__((packed));

// 解码器
struct StreamDecoder
{
    AVCodecContext* decoder_ctx;
    SwsContext* sws_ctx;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;

    std::vector<uint8_t> sps_data;
    std::vector<uint8_t> pps_data;
    bool sps_received = false;
    bool pps_received = false;
    bool extradata_set = false;

    // 禁止拷贝（防止浅拷贝 double free）
    StreamDecoder(const StreamDecoder&) = delete;
    StreamDecoder& operator=(const StreamDecoder&) = delete;

    // 移动构造，允许移动
    StreamDecoder(StreamDecoder&& other) noexcept
        : decoder_ctx(other.decoder_ctx), sws_ctx(other.sws_ctx), frame(other.frame), pkt(other.pkt),
          sps_data(std::move(other.sps_data)), pps_data(std::move(other.pps_data)),sps_received(other.sps_received),
          pps_received(other.pps_received),extradata_set(other.extradata_set)
        {
        other.decoder_ctx = nullptr;
        other.sws_ctx = nullptr;
        other.frame = nullptr;
        other.pkt = nullptr;
        other.sps_received = false;
        other.pps_received = false;
        other.extradata_set = false;
    }

    // 构造函数
    StreamDecoder(AVCodecContext* decoder_ctx, SwsContext* sws_ctx, AVFrame* frame, AVPacket* pkt)
        : decoder_ctx(decoder_ctx), sws_ctx(sws_ctx),frame(frame), pkt(pkt) {}

    // 析构时释放资源
    ~StreamDecoder()
    {
        if (decoder_ctx) avcodec_free_context(&decoder_ctx);
        if (sws_ctx) sws_freeContext(sws_ctx);
        if (frame) av_frame_free(&frame);
        if (pkt) av_packet_free(&pkt);
    }
};

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

struct VideoFrame {
    int fd;              // 哪路流
    uint64_t timestamp;  // 时间戳
    cv::Mat image;       // BGR 图像
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

    Ort::Env env_;
    Ort::Session session_;

    FdThreadPool* tp_; //解码线程池

    std::vector<const char*> input_names_ = {"images"};
    std::vector<const char*> output_names_ = {"output0"};

    std::vector<std::thread> workers_; // 工作线程
    std::condition_variable c_v_;
    std::mutex mtx_w;
    std::atomic<bool> running_ = {true};

    // 缓存帧 buffer，用vector要手动锁
    std::mutex mtx_b;
    std::vector<VideoFrame> buffer_;

    // 解码器
    std::mutex mtx_m; // mutexes_专属锁
    std::map<int, std::unique_ptr<StreamDecoder>> decoders_;
    std::map<int, std::shared_ptr<std::mutex>> mutexes_; // decode锁

    DropOldQueue<VideoFrame> frame_queue_;   // 输入队列
    DropOldQueue<InferResult> result_queue_; // 输出队列

    void infer_batch(); // 推理
    // void decodeHandle(int fd, AVCodecContext* decoder_ctx, SwsContext* sws_ctx, std::vector<uint8_t> nalu); // 解码业务函数

    preImg preprocess(const cv::Mat& img);// 预处理

    std::vector<std::vector<Detection>> postprocess(
        std::vector<Ort::Value>& output_tensors,
        const std::vector<preImgInf>& preInfs, 
        float conf_thresh = 0.5f
    );// 后处理：模型输出 → 检测框

    void nms(std::vector<Detection>& dets, float iou_thresh = 0.45f);// NMS

    void dynamicBatchInferThread(); // 动态batch推理线程

    void sendResultThread(); //发送结果线程

    bool initDecoder(int fd);
    bool decodeNALU(int fd,  std::vector<uint8_t>& data, int len);
    void processBuffer(int fd, std::vector<uint8_t>& buffer);

public:
    //初始化 session_
    OnnxYoloInfr(const char* model_path, int device_id = 0);
    ~OnnxYoloInfr();

    bool removeFd(int fd); // 移除fd对应的 解码器、锁、线程池中未执行task

    DropOldQueue<VideoFrame>& get_input_queue() { return frame_queue_; }
    DropOldQueue<InferResult>& get_output_queue() { return result_queue_; }

    void decodeTaskAdd(int fd, std::vector<uint8_t>& buffer); // 分配解码器, 将解码业务插入线程池，解码器到达上限时拒绝
    
    void run();// 启动推理和发送工作线程

};


#endif // ONNX_YOLO_INFR_H
