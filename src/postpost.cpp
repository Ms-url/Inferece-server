#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <cmath>
#include <numeric>

struct Detection {
    cv::Rect bbox;
    int frame_id;
    // 可加 confidence, class_id 等
};

struct Track {
    int id;
    std::deque<Detection> history;   // 按时间顺序
    int lost_count = 0;              // 连续丢失帧数
    bool active = true;

    Track(int id, const Detection& det) : id(id) {
        history.push_back(det);
    }

    // 预测当前帧中心位置（使用匀速模型）
    cv::Point2f predict() const {
        if (history.size() == 1) {
            // 只有一个点，直接用该点
            const auto& det = history.back();
            return cv::Point2f(det.bbox.x + det.bbox.width/2.0f,
                               det.bbox.y + det.bbox.height/2.0f);
        } else {
            // 至少两个点：外推速度
            const auto& p0 = history[history.size()-2];
            const auto& p1 = history.back();
            float dt = p1.frame_id - p0.frame_id;
            if (dt <= 0) dt = 1;
            cv::Point2f c0(p0.bbox.x + p0.bbox.width/2.0f,
                           p0.bbox.y + p0.bbox.height/2.0f);
            cv::Point2f c1(p1.bbox.x + p1.bbox.width/2.0f,
                           p1.bbox.y + p1.bbox.height/2.0f);
            cv::Point2f vel = (c1 - c0) / dt;  // 每帧速度
            return c1 + vel;   // 预测下一帧
        }
    }

    // 计算速度标准差（衡量运动平滑性）
    float speedStd() const {
        if (history.size() < 3) return 0.0f;
        std::vector<float> speeds;
        for (size_t i = 1; i < history.size(); ++i) {
            const auto& prev = history[i-1];
            const auto& curr = history[i];
            float dt = curr.frame_id - prev.frame_id;
            if (dt <= 0) continue;
            cv::Point2f c_prev(prev.bbox.x + prev.bbox.width/2.0f,
                               prev.bbox.y + prev.bbox.height/2.0f);
            cv::Point2f c_curr(curr.bbox.x + curr.bbox.width/2.0f,
                               curr.bbox.y + curr.bbox.height/2.0f);
            float dist = cv::norm(c_curr - c_prev);
            speeds.push_back(dist / dt);
        }
        if (speeds.empty()) return 0.0f;
        float mean = std::accumulate(speeds.begin(), speeds.end(), 0.0f) / speeds.size();
        float sq_sum = 0.0f;
        for (float s : speeds) sq_sum += (s - mean) * (s - mean);
        return std::sqrt(sq_sum / speeds.size());
    }
};

class SimpleTracker {
public:
    SimpleTracker(float dist_thresh = 50.0f, int max_lost = 5, int min_hits = 5,
                  float max_speed_var = 10.0f)
        : dist_thresh_(dist_thresh), max_lost_(max_lost),
          min_hits_(min_hits), max_speed_var_(max_speed_var), next_id_(0) {}

    // 主接口：输入当前帧检测框列表和帧号，返回筛选后的稳定轨迹（拷贝）
    std::vector<Track> update(const std::vector<cv::Rect>& detections, int frame_id) {
        // 1. 预测所有活跃轨迹的位置
        std::vector<cv::Point2f> predictions;
        for (auto& t : tracks_) {
            predictions.push_back(t.predict());
        }

        // 2. 关联（贪心最近邻）
        std::vector<bool> track_matched(tracks_.size(), false);
        std::vector<bool> det_matched(detections.size(), false);


        // 对于每条轨迹，找最近的未匹配检测
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (!tracks_[i].active) continue;
            float best_dist = dist_thresh_;
            int best_j = -1;
            for (size_t j = 0; j < detections.size(); ++j) {
                if (det_matched[j]) continue;
                cv::Point2f c(detections[j].x + detections[j].width/2.0f,
                              detections[j].y + detections[j].height/2.0f);
                float d = cv::norm(predictions[i] - c);
                if (d < best_dist) {
                    best_dist = d;
                    best_j = j;
                }
            }
            if (best_j >= 0) {
                track_matched[i] = true;
                det_matched[best_j] = true;
                // 更新轨迹：添加新检测，重置丢失计数
                Detection det{detections[best_j], frame_id};
                tracks_[i].history.push_back(det);
                tracks_[i].lost_count = 0;
            }
        }

        // 3. 处理未匹配的检测 -> 新建轨迹
        for (size_t j = 0; j < detections.size(); ++j) {
            if (!det_matched[j]) {
                Detection det{detections[j], frame_id};
                tracks_.emplace_back(next_id_++, det);
            }
        }

        // 4. 处理未匹配的轨迹 -> 增加丢失计数，若超阈值则标记为非活跃（稍后删除）
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (!track_matched[i] && tracks_[i].active) {
                tracks_[i].lost_count++;
                if (tracks_[i].lost_count > max_lost_) {
                    tracks_[i].active = false;
                }
            }
        }

        // 5. 筛选稳定轨迹（长度足够且运动平滑）
        std::vector<Track> stable;
        for (const auto& t : tracks_) {
            if (t.history.size() >= min_hits_ && t.speedStd() < max_speed_var_) {
                stable.push_back(t);
            }
        }
        return stable;
    }

private:
    float dist_thresh_;
    int max_lost_;
    int min_hits_;
    float max_speed_var_;
    int next_id_;
    std::vector<Track> tracks_;
};

// int main() {
//     SimpleTracker tracker(50.0f, 5, 5, 10.0f);  // 距离阈值50像素，最大丢失5帧，最少5个命中，速度标准差<10

//     cv::VideoCapture cap("video.mp4");
//     int frame_id = 0;
//     while (cap.read(frame)) {
//         // 运行YOLO得到检测框列表（仅位置，或可携带置信度）
//         std::vector<cv::Rect> dets = runYOLO(frame); // 自定义

//         std::vector<Track> stable = tracker.update(dets, frame_id++);

//         // 绘制稳定轨迹（最新框）
//         for (const auto& t : stable) {
//             const auto& last = t.history.back();
//             cv::rectangle(frame, last.bbox, cv::Scalar(0,255,0), 2);
//             cv::putText(frame, std::to_string(t.id), last.bbox.tl(),
//                         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255));
//         }
//         cv::imshow("Tracking", frame);
//         cv::waitKey(1);
//     }
// }