#include <iostream>
#include <cstring>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <endian.h>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

struct PacketHeader {
    uint32_t nalu_len;   // 网络序
    uint8_t  flags;      // 0x01=IDR, 0x02=config, 0x00=normal
    uint64_t timestamp;  // 毫秒, 网络序
} __attribute__((packed));

// 跨平台字节交换
static inline uint64_t ntohll_custom(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(x);
#else
    return __bswap_64(x);   // 需 <byteswap.h>
#endif
}

// 安全接收指定长度数据
bool recv_all(int sock, uint8_t* buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t got = recv(sock, buf + total, len - total, 0);
        if (got <= 0) {
            if (got == -1) perror("recv");
            return false;
        }
        total += got;
    }
    return true;
}

// 接收一个完整包：header + nalu 数据
bool recv_packet(int sock, std::vector<uint8_t>& nalu, uint8_t& flags, uint64_t& timestamp_ms) {
    PacketHeader header;
    if (!recv_all(sock, reinterpret_cast<uint8_t*>(&header), sizeof(header))) {
        std::cerr << "Failed to read header\n";
        return false;
    }
    uint32_t len = ntohl(header.nalu_len);
    if (len == 0 || len > 10 * 1024 * 1024) {
        std::cerr << "Invalid length: " << len << std::endl;
        return false;
    }
    flags = header.flags;
    timestamp_ms = ntohll_custom(header.timestamp);
    nalu.resize(len);
    if (!recv_all(sock, nalu.data(), len)) {
        std::cerr << "Failed to read NALU data\n";
        return false;
    }
    std::cout << "Received packet: len=" << len << ", flags=0x"
              << std::hex << (int)flags << std::dec << ", ts=" << timestamp_ms << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    int port = (argc > 1) ? std::stoi(argv[1]) : 8888;

    // ---------- 网络监听 ----------
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("socket"); return -1; }
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(listen_fd); return -1;
    }
    if (listen(listen_fd, 1) < 0) {
        perror("listen"); close(listen_fd); return -1;
    }
    std::cout << "Server listening on port " << port << std::endl;

    int client_fd = accept(listen_fd, nullptr, nullptr);
    if (client_fd < 0) { perror("accept"); close(listen_fd); return -1; }
    std::cout << "Client connected\n";
    close(listen_fd);

    // ---------- 初始化解码器（无 extradata，纯 Annex B 模式） ----------
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "H.264 decoder not found\n";
        close(client_fd);
        return -1;
    }
    AVCodecContext* dec_ctx = avcodec_alloc_context3(codec);
    if (!dec_ctx) {
        std::cerr << "alloc decoder context failed\n";
        close(client_fd);
        return -1;
    }

    // 直接打开解码器，不设置 extradata
    if (avcodec_open2(dec_ctx, codec, nullptr) < 0) {
        std::cerr << "Failed to open decoder\n";
        avcodec_free_context(&dec_ctx);
        close(client_fd);
        return -1;
    }
    std::cout << "Decoder opened (Annex B mode)\n";

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    if (!frame || !pkt) {
        std::cerr << "alloc frame/packet failed\n";
        avcodec_free_context(&dec_ctx);
        close(client_fd);
        return -1;
    }

    SwsContext* sws_ctx = nullptr;
    cv::Mat bgr_frame;
    bool first_frame = true;
    int frame_count = 0;
    std::vector<uint8_t> nalu_buf;
    uint8_t flags;
    uint64_t timestamp_ms;

    // ---------- 主循环：接收并解码 ----------
    while (recv_packet(client_fd, nalu_buf, flags, timestamp_ms)) {
        // 所有包统一作为 Annex B 数据送入解码器，不再区分 flags
        pkt->data = nalu_buf.data();
        pkt->size = nalu_buf.size();
        pkt->pts = pkt->dts = AV_NOPTS_VALUE;

        int ret = avcodec_send_packet(dec_ctx, pkt);
        av_packet_unref(pkt);   // 安全清理，不释放 vector 的数据
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            std::cerr << "send_packet error: " << ret << std::endl;
            break;
        }

        // 取出所有解码完成的帧
        while (true) {
            ret = avcodec_receive_frame(dec_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                std::cerr << "receive_frame error: " << ret << std::endl;
                goto cleanup;
            }

            frame_count++;
            std::cout << "Decoded frame #" << frame_count
                      << ", flags=0x" << std::hex << (int)flags
                      << std::dec << ", ts=" << timestamp_ms << std::endl;

            // 首次初始化颜色转换器
            if (first_frame) {
                sws_ctx = sws_getContext(frame->width, frame->height,
                                         (AVPixelFormat)frame->format,
                                         frame->width, frame->height,
                                         AV_PIX_FMT_BGR24,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!sws_ctx) {
                    std::cerr << "sws_getContext failed\n";
                    goto cleanup;
                }
                bgr_frame.create(frame->height, frame->width, CV_8UC3);
                first_frame = false;
            }

            // YUV → BGR
            uint8_t* dst_data[1] = { bgr_frame.data };
            int dst_stride[1] = { (int)bgr_frame.step[0] };
            sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                      dst_data, dst_stride);

            // 在此处理 bgr_frame，例如保存图片、发送给推理模块等
            // cv::imwrite("frame_" + std::to_string(frame_count) + ".jpg", bgr_frame);
        }
    }

cleanup:
    if (sws_ctx) sws_freeContext(sws_ctx);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec_ctx);
    close(client_fd);
    return 0;
}