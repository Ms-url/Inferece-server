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
    LOG_INFO("OnnxYoloInfr destructor called");

    // 收集所有 fd
    std::vector<int> fds;
    for (const auto& [fd, _] : decoders_) {
        fds.push_back(fd);
    }
    LOG_DEBUG("Cleaning up %zu decoders", fds.size());
    
    // 逐个删
    for (int fd : fds) {
        {
            std::lock_guard lock(*mutexes_[fd]);
            decoders_.erase(fd);
            LOG_DEBUG("Removed decoder for fd %d", fd);
        }  
        std::lock_guard<std::mutex> lock(mtx_m); // mutexes锁
        mutexes_.erase(fd);  
    }

    delete(tp_); // 删除线程池
    LOG_DEBUG("Thread pool deleted");
    
    avformat_network_deinit();
    LOG_INFO("OnnxYoloInfr cleanup complete");
}


/**
 * 移除fd对应的 解码器、锁、线程池中未执行task
 * close fd 后 removeDecode 要把线程池里在排队的 fd 也删掉，
 * 不然 解码器已经被删了 会出错
 */ 
bool ThreadDecoder::removeFd(int fd){
    LOG_INFO("Removing fd %d from OnnxYoloInfr", fd);
    
    // 清除线程池待执行队列中的 fd 的任务
    int removed_tasks = tp_->removeFdTask(fd);
    if (removed_tasks > 0) {
        LOG_DEBUG("Removed %d pending tasks for fd %d from thread pool", removed_tasks, fd);
    }

    LOG_DEBUG("debug point 1");

    // 拿到这个 fd 的 mutex
    auto it = mutexes_.find(fd);
    if (it == mutexes_.end()) {
        LOG_WARNING("fd %d not found in mutexes map", fd);
        return false;
    }

    LOG_DEBUG("debug point 2");
    
    {
        std::lock_guard<std::mutex> lock(*(it->second));  // decode锁住
        LOG_DEBUG("debug point 3");
        decoders_.erase(fd);
        LOG_DEBUG("Decoder for fd %d removed", fd);
    }

    LOG_DEBUG("debug point 4");
    
    std::lock_guard<std::mutex> lock(mtx_m); // mutexes专属锁
    mutexes_.erase(fd);
    LOG_INFO("fd %d successfully removed from OnnxYoloInfr", fd);
    return true;
}


/**
 * 调该函数必须锁住
 */
bool ThreadDecoder::initDecoder(int fd) {
    LOG_DEBUG("Initializing H.264 decoder");
    AVCodecContext* decoder_ctx; 
    SwsContext* sws_ctx = nullptr;
 
    LOG_INFO("fd %d: Allocating new decoder", fd);
    bool max_dec;

    auto de_it = decoders_.find(fd);
    max_dec = decoders_.size() >= MAX_NUM_DECODERS_;
    
    if (max_dec) {
        LOG_WARNING("fd %d: Max decoders reached (%d), rejecting", fd, MAX_NUM_DECODERS_);
        return false; // 最大解码器数量限制
    }
    // 为新 fd 创建解码器
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        LOG_ERROR("fd %d: H.264 decoder not found", fd);
        return false;
    }
    
    decoder_ctx = avcodec_alloc_context3(codec);
    if (!decoder_ctx){
        LOG_ERROR("fd %d: Failed to allocate decoder context", fd);
        return false;
    }
    // decoder_ctx->thread_count = 4;
    decoder_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    
    if (avcodec_open2(decoder_ctx, codec, nullptr) < 0) {
        LOG_ERROR("fd %d: Failed to open decoder", fd);
        avcodec_free_context(&decoder_ctx);
        return false;
    }

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    if (!frame || !pkt) {
        LOG_ERROR("Failed to allocate frame or packet");
        if (frame) av_frame_free(&frame);
        if (pkt) av_packet_free(&pkt);
        avcodec_free_context(&decoder_ctx);
        return false;
    }
    // 加入map
    decoders_[fd] = std::make_unique<StreamDecoder>(decoder_ctx, sws_ctx, frame, pkt);
    mutexes_[fd] = std::make_shared<std::mutex>();
    
    LOG_INFO("fd %d: Decoder allocated successfully, total decoders: %zu", fd, decoders_.size());
    
    return true;
}

bool ThreadDecoder::decodeNALU(int fd, std::vector<uint8_t>& data, int len) {
    LOG_DEBUG("Decoding NALU, data size: %d bytes", len);

    StreamDecoder* decoder = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtx_m);
        auto it = decoders_.find(fd);
        if (it == decoders_.end()) {
            LOG_DEBUG("Decoder not initialized, initializing now");
            if (!initDecoder(fd)) {
                LOG_ERROR("Failed to initialize decoder");
                return false;
            }
            decoder = decoders_[fd].get();
        } else {
            decoder = it->second.get();
        }
    }
    
    if (!decoder) {
        LOG_ERROR("Decoder is null for fd %d", fd);
        return false;
    }

    AVCodecContext* decoder_ctx = decoder->decoder_ctx;
    AVFrame* frame = decoder->frame;
    AVPacket* pkt = decoder->pkt;
    SwsContext* sws_ctx = decoder->sws_ctx;
    // YUV → BGR for OpenCV
    cv::Mat bgr_frame;

    // 检测NALU类型（仅用于日志）
    int start_code_len = 0;
    if (len >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
        start_code_len = 4;
    } else if (len >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1) {
        start_code_len = 3;
    }
    
    if (start_code_len > 0 && len > start_code_len) {
        uint8_t nalu_type = data[start_code_len] & 0x1F;
        switch (nalu_type) {
            case 5:LOG_DEBUG("===========Received key frame (IDR)");break;
            case 7:LOG_DEBUG("===========Received SPS (Sequence Parameter Set)");break;
            case 8:LOG_DEBUG("===========Received PPS (Picture Parameter Set)");break;
            case 1:LOG_DEBUG("===========Received non-IDR frame");break;
            default:LOG_DEBUG("===========Received NALU type: %d", nalu_type);break;
        }
    } else {
        LOG_WARNING("Invalid NALU: no start code found, len=%d", len);
    }

    pkt->data = data.data();
    pkt->size = data.size();
    pkt->pts = pkt->dts = AV_NOPTS_VALUE;
    
    // 发送到解码器
    int ret = avcodec_send_packet(decoder_ctx, pkt);
    av_packet_unref(pkt);  // 发送后立即释放引用
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        std::cerr << "send_packet error: " << ret << std::endl;
        return false;
    }

    bool has_frame = false;
    int frame_count = 0;
    while (true) {
        ret = avcodec_receive_frame(decoder_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR("Failed to receive frame: %s", errbuf);
            break;
        }
        
        has_frame = true;
        frame_count++;
        LOG_DEBUG("Received decoded frame #%d, size: %dx%d", 
                  frame_count, frame->width, frame->height);
        
        if (!sws_ctx) {
            sws_ctx = sws_getContext(
                frame->width, frame->height, (AVPixelFormat)frame->format,
                frame->width, frame->height, AV_PIX_FMT_BGR24,
                SWS_BILINEAR, nullptr, nullptr, nullptr
            );
            if (!sws_ctx) {
                LOG_ERROR("fd %d: Failed to create sws context", fd);
                av_packet_unref(pkt);
                av_packet_free(&pkt);
                av_frame_free(&frame);
                return false;
            }
            LOG_DEBUG("fd %d: SWS context created for %dx%d", fd, frame->width, frame->height);
            bgr_frame.create(frame->height, frame->width, CV_8UC3);
        }
        
        uint8_t* dest_data[1] = { bgr_frame.data };
        int dest_stride[1] = { (int)bgr_frame.step[0] };
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                    dest_data, dest_stride);
        
        frame_queue_.push(std::move(VideoFrame{fd, 0, bgr_frame}));

        av_frame_unref(frame);
    }
    
    if (frame_count > 0) {
        LOG_DEBUG("Decoded %d frames from NALU", frame_count);
    }
    
    return has_frame;
}

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

        std::cout << "Received packet: len=" << nalu_len << ", flags=0x"
                << std::hex << (int)flags << std::dec << ", ts=" << timestamp_ms << std::endl;
        
        decodeNALU(fd, nalu_data, nalu_len);
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
 * 
 */ 
void ThreadDecoder::decodeTaskAdd(int fd, std::vector<uint8_t>& buffer){
    LOG_DEBUG("fd %d: Adding decode task to thread pool", fd);
    
    processBuffer(fd, buffer);
    
    // 检查解码器状态
    if (buffer.size() > 1024 * 1024) {  // 缓冲超过1MB
        /**
         * 清除缓冲 / 丢弃帧
         *  */ 
        LOG_WARNING("fd %d: Buffer size growing large: %zu bytes", fd, buffer.size());
    }
}
