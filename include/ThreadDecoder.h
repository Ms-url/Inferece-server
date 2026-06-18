#ifndef THREAD_DECODE_H
#define THREAD_DECODE_H

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

inline constexpr size_t MAX_FRAME = 8;
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

    // 禁止拷贝（防止浅拷贝 double free）
    StreamDecoder(const StreamDecoder&) = delete;
    StreamDecoder& operator=(const StreamDecoder&) = delete;

    // 移动构造，允许移动
    StreamDecoder(StreamDecoder&& other) noexcept
        : decoder_ctx(other.decoder_ctx), sws_ctx(other.sws_ctx), frame(other.frame), pkt(other.pkt)
        {
        other.decoder_ctx = nullptr;
        other.sws_ctx = nullptr;
        other.frame = nullptr;
        other.pkt = nullptr;
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

struct VideoFrame {
    int fd;              // 哪路流
    uint64_t timestamp;  // 时间戳
    cv::Mat image;       // BGR 图像
};

class ThreadDecoder
{
private:
    FdThreadPool* tp_; //解码线程池

    // 解码器
    std::mutex mtx_m; // mutexes_专属锁
    std::map<int, std::unique_ptr<StreamDecoder>> decoders_;
    std::map<int, std::shared_ptr<std::mutex>> mutexes_; // decode锁

    DropOldQueue<VideoFrame> frame_queue_;   // 解码完成队列

    bool initDecoder(int fd);
    bool decodeNALU(int fd,  std::vector<uint8_t>& data, int len);
    void processBuffer(int fd, std::vector<uint8_t>& buffer);

public:
    
    ThreadDecoder();
    ~ThreadDecoder();

    bool removeFd(int fd); // 移除fd对应的 解码器、锁、线程池中未执行task

    DropOldQueue<VideoFrame>& get_frame_queue() { return frame_queue_; }

    void decodeTaskAdd(int fd, std::vector<uint8_t>& buffer); // 分配解码器, 将解码业务插入线程池，解码器到达上限时拒绝
    
};


#endif // THREAD_DECODE_H
