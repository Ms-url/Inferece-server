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
#include <memory>

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
    std::mutex mtx_map; // map 锁
    std::map<int, AVCodecContext*> decoders_;
    std::map<int, AVFrame*> frames_;
    std::map<int, AVPacket*> pkts_;
    std::map<int, SwsContext*> sws_ctxs_;
    std::map<int, std::shared_ptr<std::mutex>> mutexes_; // decode锁

    std::map<int, std::shared_ptr<DropOldQueue<std::vector<uint8_t>>>> fd_nalu_;   // <fd,等待解码队列>
    DropOldQueue<VideoFrame> frame_queue_;   // 解码完成队列

    bool initDecoder(int fd);
    bool decodeNALU(int fd, int len, uint64_t timestamp_ms);
    void processBuffer(int fd, std::vector<uint8_t>& buffer);

public:
    
    ThreadDecoder();
    ~ThreadDecoder();

    bool removeFd(int fd); // 移除fd对应的 解码器、锁、线程池中未执行task

    DropOldQueue<VideoFrame>& get_frame_queue() { return frame_queue_; }

    void decodeTaskAdd(int fd, std::vector<uint8_t>& buffer); // 分配解码器, 将解码业务插入线程池，解码器到达上限时拒绝
    
};


#endif // THREAD_DECODE_H
