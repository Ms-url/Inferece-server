#include "OnnxYoloInfr.h"
#include "Logger.h"  // 添加日志头文件

/**
 * 初始化 session_
 */
OnnxYoloInfr::OnnxYoloInfr(const char* model_path, ThreadDecoder* td, int device_id)
        :env_(ORT_LOGGING_LEVEL_WARNING, "YOLO"),
         session_(nullptr),
         td_(td),
         result_queue_(MAX_BATCH)
{
    LOG_INFO("Initializing OnnxYoloInfr with model: %s, device_id: %d", model_path, device_id);
    
    avformat_network_init();
    LOG_DEBUG("FFmpeg network initialized");
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    LOG_DEBUG("ONNX Runtime session options configured");
    
    // GPU 推理
    // OrtCUDAProviderOptions cuda_options;
    // session_options.AppendExecutionProvider_CUDA(cuda_options);
    
    try {
        session_ = Ort::Session(env_, model_path, session_options);
        LOG_INFO("ONNX Runtime session created successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create ONNX Runtime session: %s", e.what());
        throw;
    }
}

/**
 * 析构
 */
OnnxYoloInfr::~OnnxYoloInfr(){
    LOG_INFO("OnnxYoloInfr destructor called");
    
    {
        std::lock_guard<std::mutex> lock(mtx_w);
        running_ = false;
    }
    c_v_.notify_all();
    LOG_DEBUG("Waiting for worker threads to finish");
    for(auto& work: workers_){
        work.join();
    }
    LOG_DEBUG("All worker threads joined");
    
    avformat_network_deinit();
    LOG_INFO("OnnxYoloInfr cleanup complete");
}

/**
 * 推理
 */
void OnnxYoloInfr::infer_batch(){
    LOG_INFO("Starting batch inference");
    
    std::vector<preImgInf> preInfs;
    std::vector<VideoFrame> batch;
    {
        std::lock_guard<std::mutex> lock(mtx_b);
        batch = std::move(buffer_);
        buffer_.clear();
    }
    LOG_DEBUG("Processing batch of size %zu", batch.size());

    // ---- 构造输入 tensor ----
    std::vector<int64_t> input_shape = {static_cast<int64_t>(batch.size()), 3, 640, 640 };
    std::vector<float> input_data(batch.size() * 3 * 640 * 640);
    LOG_DEBUG("Allocated input tensor of size %zu", input_data.size());

    for (size_t i = 0; i < batch.size(); i++) {
        preImg pre_img = preprocess(batch[i].image);
        preInfs.push_back(pre_img.preInf);
        cv::Mat blob = pre_img.blob;
        memcpy(input_data.data() + i * 3 * 640 * 640, blob.data, 3 * 640 * 640 * sizeof(float));
    }
    LOG_DEBUG("Preprocessed %zu images", batch.size());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, 
        OrtMemTypeDefault
    ); // 获取内存信息用于创建张量

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(), 
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // ---- 推理 ----
    LOG_DEBUG("Running ONNX inference");
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        input_names_.data(), &input_tensor, 1, 
        output_names_.data(), 1
    );
    LOG_DEBUG("Inference completed");

    // ---- 后处理 ----
    LOG_DEBUG("Starting post-processing");
    std::vector<std::vector<Detection>> dets = postprocess(output_tensors, preInfs, 0.3f);   // 置信度阈值0.5
    for(int i=0; i < dets.size(); i++){
        nms(dets[i], 0.45f); 
    }

    std::vector<InferResult> results;
    for (size_t i = 0; i < batch.size(); i++) {
        InferResult res;
        res.fd = batch[i].fd;
        res.timestamp = batch[i].timestamp;
        res.det = dets[i];
        results.push_back(std::move(res)); // 插入结果队列
    }

    // 丢进结果队列（满了自动丢旧的）
    for (auto& res : results) {
        result_queue_.push(std::move(res));
    }
    LOG_INFO("Batch inference complete, %zu results queued", results.size());
}

/**
 * 预处理
 * Letterbox 处理
 * 转blob，归一化到0~1，通道顺序RGB
 */
preImg OnnxYoloInfr::preprocess(const cv::Mat& img) {
    int img_w = img.cols;
    int img_h = img.rows;

    // Letterbox 处理（保持宽高比填充）
    int target_size = 640;
    float scale = std::min((float)target_size / img_w, (float)target_size / img_h);
    int new_w = int(img_w * scale);
    int new_h = int(img_h * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::Mat canvas = cv::Mat::ones(target_size, target_size, CV_8UC3) * 114;
    int dx = (target_size - new_w) / 2;
    int dy = (target_size - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(dx, dy, new_w, new_h)));

    // 转blob
    // blobFromImage自动输出NCHW
    cv::Mat blob;
    cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, cv::Size(target_size, target_size), cv::Scalar(), true, false);

    return {blob,{scale,dx,dy}};
}

/**
 * 后处理：模型输出 → 检测框
 * YOLOv8输出shape: [b, 84, 8400]
 * 84 = 4(坐标) + 80(COCO类别数)
 * 8400 = 最大检测数量
 */
std::vector<std::vector<Detection>> OnnxYoloInfr::postprocess(
    std::vector<Ort::Value>& output_tensors,
    const std::vector<preImgInf>& preInfs ,
    float conf_thresh ) 
{
    LOG_DEBUG("Starting postprocess with %zu images", preInfs.size());

    std::vector<std::vector<Detection>> batch_dets;

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    int batch_size = output_shape[0];
    int num_classes = output_shape[1] - 4;
    int num_anchors = output_shape[2];  // 8400

    LOG_DEBUG("Output shape: batch=%d, classes=%d, anchors=%d", batch_size, num_classes, num_anchors);

    for (int b = 0; b < batch_size; b++){
        std::vector<Detection> dets;
        for (int i = 0; i < num_anchors; i++) {
            float* col = output_data + b*num_anchors*(num_classes + 4) + i;
            // 找最大类别
            int class_id = 0;
            float max_cls_score = 0;
            for (int c = 0; c < num_classes; c++) {
                float score = col[ (4 + c)*num_anchors ];

                if (score > max_cls_score) {
                    max_cls_score = score ;
                    class_id = c;
                }
            }

            if (max_cls_score > 0.5) {
                int dx = preInfs[b].dx;
                int dy = preInfs[b].dy;
                float scale = preInfs[b].scale;
                // 解码坐标 (x, y, w, h)
                float raw_x = (col[0] - col[2*num_anchors] / 2 - dx) / scale;
                float raw_y = (col[num_anchors] - col[3*num_anchors] / 2 - dy) / scale;
                float raw_w = col[2*num_anchors] / scale;
                float raw_h = col[3*num_anchors] / scale;
                // 转换为整数坐标
                int x = static_cast<int>(raw_x);
                int y = static_cast<int>(raw_y);
                int w = static_cast<int>(raw_w);
                int h = static_cast<int>(raw_h);
                dets.push_back({std::max(0, x), std::max(0, y), w, h, max_cls_score, class_id});
            }
        }
        batch_dets.push_back(dets);
        LOG_INFO("Image %d: %zu detections before NMS", b, dets.size());
    }
    
    LOG_DEBUG("Postprocess complete");
    return batch_dets;
}

/**
 * nms
 */
void OnnxYoloInfr::nms(std::vector<Detection>& dets, float iou_thresh) {
    if (dets.empty()) {
        LOG_DEBUG("NMS: empty detections, skipping");
        return;
    }
    
    LOG_DEBUG("NMS: processing %zu detections with IoU threshold %.2f", dets.size(), iou_thresh);
    
    // 按置信度降序排序
    std::sort(dets.begin(), dets.end(), 
        [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    
    std::vector<bool> removed(dets.size(), false);
    
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        
        // 预计算当前框的边界（假设使用左上角坐标）
        const int ix1_i = dets[i].x;
        const int iy1_i = dets[i].y;
        const int ix2_i = dets[i].x + dets[i].w;
        const int iy2_i = dets[i].y + dets[i].h;
        const float area_i = static_cast<float>(dets[i].w * dets[i].h);
        
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            
            // 只在同类之间做NMS
            if (dets[i].class_id != dets[j].class_id) continue;
            
            // 计算交集
            const int ix1 = std::max(ix1_i, dets[j].x);
            const int iy1 = std::max(iy1_i, dets[j].y);
            const int ix2 = std::min(ix2_i, dets[j].x + dets[j].w);
            const int iy2 = std::min(iy2_i, dets[j].y + dets[j].h);
            
            const int inter_w = std::max(0, ix2 - ix1);
            const int inter_h = std::max(0, iy2 - iy1);
            const float inter_area = static_cast<float>(inter_w * inter_h);
            
            // 计算IoU
            const float union_area = area_i + dets[j].w * dets[j].h - inter_area;
            const float iou = inter_area / union_area;
            
            if (iou > iou_thresh) {
                removed[j] = true;
            }
        }
    }
    
    // 收集结果
    std::vector<Detection> result;
    result.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!removed[i]) {
            LOG_DEBUG("ADD BOX: cls: %d", dets[i].class_id);
            result.push_back(std::move(dets[i]));
        }
    }
    
    LOG_DEBUG("NMS: %zu detections remaining after NMS", result.size());
    dets = std::move(result);
}

/**
 * 动态batch推理
 */
void OnnxYoloInfr::dynamicBatchInferThread(){
    LOG_INFO("Dynamic batch inference thread started");
    
    std::vector<Ort::Value> input_tensors;
    std::vector<int64_t> input_shape = {1, 3, 640, 640};  // YOLO 输入尺寸

    while (true) {
        VideoFrame frame = td_->get_frame_queue().pop();  // 阻塞等一帧
        LOG_DEBUG("Received frame from queue, fd: %d", frame.fd);
        
        {
            std::lock_guard<std::mutex> lock(mtx_b);
            buffer_.push_back(std::move(frame));
        }

        // 判断是否触发推理
        bool should_infer = false;
        {
            std::lock_guard<std::mutex> lock(mtx_b);
            should_infer = (buffer_.size() >= MAX_BATCH);
            // should_infer = (buffer_.size() >= 1);
        }

        if (should_infer) {
            LOG_DEBUG("Triggering inference: buffer size reached %d", MAX_BATCH);
            infer_batch();
        } else {
            // 没攒够，等超时
            std::this_thread::sleep_for(std::chrono::milliseconds(MAX_WAIT_MS_));
            bool tmp = false;
            {
                std::lock_guard<std::mutex> lock(mtx_b);
                if (!buffer_.empty()) {
                    tmp = true;
                }
            }
            if(tmp){
                LOG_DEBUG("Triggering inference: timeout with %zu frames in buffer", buffer_.size());
                infer_batch();
            }
        }
    }
}

/**
 * 发送结果
 * 发送时检查 fd 是否已经关闭
 */
void OnnxYoloInfr::sendResultThread() {
    LOG_INFO("Send result thread started");
    
    while (running_) { 
        InferResult result = result_queue_.pop();
        LOG_DEBUG("fd %d: Sending result with %zu detections", result.fd, result.det.size());
        
        // 检查 fd 有效性
        if (result.fd < 0) {
            LOG_WARNING("Invalid fd %d, skipping result", result.fd);
            continue;
        }
        
        // 用 fcntl 检查 fd 是否已关闭
        int ret = fcntl(result.fd, F_GETFD);
        if (ret == -1) {
            LOG_WARNING("fd %d is closed, skipping result", result.fd);
            close(result.fd);
            continue;
        }
        
        // 构造发送缓冲区
        std::vector<uint8_t> buf;
        buf.reserve( 4 + result.det.size() * 24);
        
        // timestamp: 8字节
        // auto ts_bytes = reinterpret_cast<const uint8_t*>(&result.timestamp);
        // buf.insert(buf.end(), ts_bytes, ts_bytes + 8);
        
        // 检测数量: 4字节
        uint32_t count = static_cast<uint32_t>(result.det.size());
        auto cnt_bytes = reinterpret_cast<const uint8_t*>(&count);
        buf.insert(buf.end(), cnt_bytes, cnt_bytes + 4);
        
        // 每个 Detection: 24字节 (4+4+4+4+4+4)
        for (const auto& d : result.det) {
            buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&d.x),
                    reinterpret_cast<const uint8_t*>(&d.x) + 4);
            buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&d.y),
                    reinterpret_cast<const uint8_t*>(&d.y) + 4);
            buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&d.w),
                    reinterpret_cast<const uint8_t*>(&d.w) + 4);
            buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&d.h),
                    reinterpret_cast<const uint8_t*>(&d.h) + 4);
            buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&d.conf),
                    reinterpret_cast<const uint8_t*>(&d.conf) + 4);
            buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&d.class_id),
                    reinterpret_cast<const uint8_t*>(&d.class_id) + 4);
        }
        
        LOG_DEBUG("fd %d: Sending %zu bytes", result.fd, buf.size());
        // 发送，MSG_NOSIGNAL 避免 SIGPIPE
        ssize_t sent = send(result.fd, buf.data(), buf.size(), MSG_NOSIGNAL);
        if (sent < 0) {
            LOG_ERROR("fd %d: Send error: %s", result.fd, strerror(errno));
            close(result.fd);
            td_->removeFd(result.fd);
        } else if (sent < static_cast<ssize_t>(buf.size())) {
            LOG_WARNING("fd %d: Partial send: %zd/%zu bytes", result.fd, sent, buf.size());
        } else {
            LOG_DEBUG("fd %d: Result sent successfully (%zu bytes)", result.fd, buf.size());
        }
    }
    
    LOG_INFO("Send result thread stopped");
}

/**
 * 启动推理和发送工作线程
 */
void OnnxYoloInfr::run(){
    LOG_INFO("Starting OnnxYoloInfr worker threads");
    
    LOG_DEBUG("Starting dynamic batch inference thread");
    workers_.emplace_back([this]{
        dynamicBatchInferThread();
    });

    // LOG_DEBUG("Starting result send thread");
    // workers_.emplace_back([this]{
    //      sendResultThread();
    // });
    
    LOG_INFO("OnnxYoloInfr worker threads all started successfully");
}