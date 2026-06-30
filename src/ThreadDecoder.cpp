#include "ThreadDecoder.h"
#include "Logger.h"  // 添加日志头文件

/**
 * 初始化 session_
 */
ThreadDecoder::ThreadDecoder()
        :tp_(new FdThreadPool(NUM_DECODE_THREADS_)),
         frame_queue_(MAX_FRAME){}

/**
 * 析构
 */
ThreadDecoder::~ThreadDecoder(){
    LOG_INFO("ThreadDecoder destructor called");

    // 收集所有 fd
    std::vector<int> fds;
    for (const auto& [fd, _] : decoders_) {
        fds.push_back(fd);
    }
    // 逐个删
    for (int fd : fds) {
        removeFd(fd);
    }
    LOG_DEBUG("Cleaning up %zu decoders", fds.size());

    delete(tp_); // 删除线程池
    LOG_DEBUG("Thread pool deleted");
    
    avformat_network_deinit();
    LOG_INFO("ThreadDecoder cleanup complete");
}


/**
 * 移除fd对应的 解码器、锁、线程池中未执行task
 * close fd 后 removeDecode 要把线程池里在排队的 fd 也删掉，
 * 不然 解码器已经被删了 会出错
 */ 
bool ThreadDecoder::removeFd(int fd){
    LOG_INFO("Removing fd %d from ThreadDecoder", fd);

    // 清除线程池待执行队列中的 fd 的任务
    int removed_tasks = tp_->removeFdTask(fd);
    if (removed_tasks > 0) {
        LOG_DEBUG("Removed %d pending tasks for fd %d from thread pool", removed_tasks, fd);
    }

    {
        std::lock_guard<std::mutex> lock(mtx_map);
        auto it = decoders_.find(fd);
        if (it!=decoders_.end()){
            std::lock_guard lock(*mutexes_[fd]); // 锁住解码器，防止进入解码
            if (sws_ctxs_[fd]) sws_freeContext(sws_ctxs_[fd]);
            av_frame_free(&frames_[fd]);
            av_packet_free(&pkts_[fd]);
            avcodec_free_context(&decoders_[fd]);
        }
        decoders_.erase(fd);
        frames_.erase(fd);
        pkts_.erase(fd);
        sws_ctxs_.erase(fd);
    }

    LOG_INFO("fd %d successfully removed from ThreadDecoder", fd);
    return true;
}


/**
 * 调该函数必须锁住
 */
bool ThreadDecoder::initDecoder(int fd) {
    LOG_INFO("Initializing H.264 decoder for fd %d", fd);

    auto de_it = decoders_.find(fd);
    bool max_dec = decoders_.size() >= MAX_NUM_DECODERS_;
    
    if (max_dec) {
        LOG_WARNING("fd %d: Max decoders reached (%d), rejecting", fd, MAX_NUM_DECODERS_);
        return false; // 最大解码器数量限制
    }

    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        LOG_ERROR("H.264 decoder not found");
        return false;
    }

    AVCodecContext* dec_ctx = avcodec_alloc_context3(codec);
    if (!dec_ctx) {
        LOG_ERROR("Failed to allocate decoder context for fd %d", fd);
        return false;
    }

    if (avcodec_open2(dec_ctx, codec, nullptr) < 0) {
        LOG_ERROR("Failed to open decoder for fd %d", fd);
        avcodec_free_context(&dec_ctx);
        return false;
    }
    LOG_DEBUG("Decoder opened successfully (Annex B mode) for fd %d", fd);

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    if (!frame || !pkt) {
        LOG_ERROR("Failed to allocate frame or packet for fd %d", fd);
        avcodec_free_context(&dec_ctx);
        if (frame) av_frame_free(&frame);
        if (pkt) av_packet_free(&pkt);
        return false;
    }

    SwsContext* sws_ctx = nullptr;  // 初始化

    // 存入 map
    {
        // 在外面锁住整个函数
        // std::lock_guard<std::mutex> lock(mtx_map);
        decoders_[fd] = dec_ctx;
        frames_[fd] = frame;
        pkts_[fd] = pkt;
        sws_ctxs_[fd] = sws_ctx;
        mutexes_[fd] = std::make_shared<std::mutex>();
    }

    LOG_INFO("fd %d: Decoder resources allocated, total decoders: %zu", fd, decoders_.size());
    return true;
}

/**
 * 解码nalu，用于插入线程池
 */
bool ThreadDecoder::decodeNALU(int fd, int len, uint64_t timestamp_ms) {
    LOG_DEBUG("decodeNALU entry: fd=%d, data size=%d bytes", fd, len);

    std::vector<uint8_t> data = (*fd_nalu_[fd]).pop();

    AVCodecContext* dec_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    SwsContext* sws_ctx = nullptr;
    std::shared_ptr<std::mutex> mtx_dec;

    {
        std::lock_guard<std::mutex> lock(mtx_map); // map 锁
        auto it = decoders_.find(fd);
        if (it == decoders_.end()) {
            LOG_DEBUG("Decoder not initialized for fd %d, initializing now", fd);
            if (!initDecoder(fd)) {
                LOG_ERROR("Failed to initialize decoder for fd %d", fd);
                return false;
            }
            // 获取
            dec_ctx = decoders_[fd];
            frame = frames_[fd];
            pkt = pkts_[fd];
            sws_ctx = sws_ctxs_[fd];
        } else {
            dec_ctx = it->second;
            frame = frames_[fd];
            pkt = pkts_[fd];
            sws_ctx = sws_ctxs_[fd];
        }
        // 解码器专属锁
        mtx_dec = mutexes_[fd];
    }

    std::lock_guard lock(*mtx_dec); // 锁住解码流程
    // 抢到锁后，先检查该解码器是否被删除
    if (!dec_ctx || !frame || !pkt) {
        LOG_ERROR("Decoder components are null for fd %d", fd);
        return false;
    }

    // 准备 packet
    pkt->data = data.data();
    pkt->size = data.size();
    pkt->pts = pkt->dts = AV_NOPTS_VALUE;
    LOG_DEBUG("Packet prepared: size=%d", pkt->size);

    // 发送 packet
    int ret = avcodec_send_packet(dec_ctx, pkt);
    av_packet_unref(pkt);  // 引用计数减1，不影响 data 指向的外部内存
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        LOG_ERROR("avcodec_send_packet failed: %d", ret);
        return false;
    }
    LOG_DEBUG("Packet sent to decoder, ret=%d", ret);

    cv::Mat bgr_frame;
    int frame_count = 0;

    // 循环接收帧
    while (true) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN)) {
            LOG_DEBUG("No more frames available now (EAGAIN)");
            break;
        }
        if (ret == AVERROR_EOF) {
            LOG_DEBUG("End of stream reached");
            break;
        }
        if (ret < 0) {
            LOG_ERROR("avcodec_receive_frame error: %d", ret);
            break;
        }

        frame_count++;
        LOG_DEBUG("Frame #%d decoded: %dx%d, format=%d", 
                  frame_count, frame->width, frame->height, frame->format);

        // 初始化颜色转换上下文（懒加载）
        if (!sws_ctx) {
            LOG_DEBUG("Creating SwsContext for fd %d (width=%d, height=%d)", 
                      fd, frame->width, frame->height);
            sws_ctx = sws_getContext(frame->width, frame->height,
                                     (AVPixelFormat)frame->format,
                                     frame->width, frame->height,
                                     AV_PIX_FMT_BGR24,
                                     SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!sws_ctx) {
                LOG_ERROR("sws_getContext failed for fd %d", fd);
                av_frame_unref(frame);
                break;
            }
            // 存入 map，供后续使用
            {
                std::lock_guard<std::mutex> lock(mtx_map);
                sws_ctxs_[fd] = sws_ctx;
            }
        }

        // 创建 BGR 图像并转换
        bgr_frame.create(frame->height, frame->width, CV_8UC3);
        uint8_t* dst_data[1] = { bgr_frame.data };
        int dst_stride[1] = { static_cast<int>(bgr_frame.step[0]) };
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                  dst_data, dst_stride);
        LOG_DEBUG("YUV->BGR conversion done for frame #%d", frame_count);

        // 推入队列
        frame_queue_.push(VideoFrame{fd, timestamp_ms, std::move(bgr_frame)});
        LOG_DEBUG("Frame #%d pushed to queue, queue size now %zu", 
                  frame_count, frame_queue_.size());

        // 清理帧引用
        av_frame_unref(frame);
    }

    LOG_DEBUG("decodeNALU exit: fd=%d, total frames decoded in this call=%d", fd, frame_count);
    return true;
}

/**
 * 从缓冲区中读取nalu，读取到完整nalu后插入线程池
 */
void ThreadDecoder::processBuffer(int fd, std::vector<uint8_t>& buffer) {
    LOG_DEBUG("fd %d: Processing buffer, size: %zu bytes", fd, buffer.size());
    
    size_t pos = 0;
    int processed_nalus = 0;
    int skipped_bytes = 0;

    const size_t headerSize = sizeof(PacketHeader);
    if (buffer.size() < headerSize) {
        return;
    }

    while(pos + headerSize <= buffer.size()){
        // 从 buffer 提取头部
        PacketHeader header;
        std::memcpy(&header, buffer.data() + pos, headerSize);
        uint32_t nalu_len = ntohl(header.nalu_len);
        uint8_t flags = header.flags;
        uint64_t timestamp_ms = ntohll_custom(header.timestamp);

        if (nalu_len == 0 || nalu_len > 10 * 1024 * 1024) {
            std::cerr << "Invalid length: " << nalu_len << std::endl;
            return ;
        }
        
        // 边界检查：确保完整NALU在buffer内
        if (pos + headerSize + nalu_len > buffer.size()) {
            LOG_DEBUG("fd %d: Incomplete NALU, waiting for more data", fd);
            break;
        }

        std::vector<uint8_t> nalu_data(
            buffer.begin() + pos + headerSize,
            buffer.begin() + pos + headerSize + nalu_len
        );
        // 验证起始码
        bool has_startcode = false;
        if (nalu_len >= 4 && 
            nalu_data[0] == 0 && nalu_data[1] == 0 && nalu_data[2] == 0 && nalu_data[3] == 1) {
            has_startcode = true;
        } else if (nalu_len >= 3 && nalu_data[0] == 0 && nalu_data[1] == 0 && nalu_data[2] == 1) {
            has_startcode = true;
        }

        if (!has_startcode) {
            LOG_WARNING("fd %d: Data at timestamp %d doesn't start with start code", fd, timestamp_ms);
            skipped_bytes++;
            pos += headerSize + nalu_len;
            continue;
        }

        LOG_INFO("Received packet: len=%d, flags=0x%02X, ts=%llu", nalu_len, (unsigned int)flags, timestamp_ms);

        // 加入nalu等待队列
        (*fd_nalu_[fd]).push(std::move(nalu_data));
        // 解码插入线程池
        tp_->Submit(fd, [this, fd, nalu_len, timestamp_ms]{
            decodeNALU(fd, nalu_len, timestamp_ms);
        });

        processed_nalus++;
        pos += (headerSize + nalu_len);
    }
    
    if (pos > 0) {
        buffer.erase(buffer.begin(), buffer.begin() + pos);
        LOG_DEBUG("fd %d: Removed %zu bytes from buffer, %zu bytes remaining, processed %d NALUs, skipped %d bytes", 
                  fd, pos, buffer.size(), processed_nalus, skipped_bytes);
    }
    
    if (processed_nalus == 0 && skipped_bytes == 0 && buffer.size() > headerSize) {
        LOG_DEBUG("fd %d: Buffer has data but no complete NALU found, waiting for more data", fd);
    }
}

/**
 * 从缓冲区读取数据并将解码操作插入线程池
 */ 
void ThreadDecoder::decodeTaskAdd(int fd, std::vector<uint8_t>& buffer){
    LOG_DEBUG("fd %d: Adding decode task ", fd);

    auto it = fd_nalu_.find(fd);
    if (it == fd_nalu_.end()){
        // 初始化该 fd 的 nalu 等待队列
        fd_nalu_[fd] = std::make_shared<DropOldQueue<std::vector<uint8_t>>>(12);
    }
    processBuffer(fd, buffer);
    
    // 检查解码器状态
    if (buffer.size() > 1024 * 1024) {  // 缓冲超过1MB
        /**
         * 清除缓冲 / 丢弃帧
         * */ 
        LOG_WARNING("fd %d: Buffer size growing large: %zu bytes", fd, buffer.size());
    }
}
