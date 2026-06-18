#ifndef DROP_OLD_QUEUE_H
#define DROP_OLD_QUEUE_H

#include <deque>
#include <mutex>
#include <condition_variable>
#include <chrono>

template<typename T>
class DropOldQueue {
    std::deque<T> q_;
    size_t max_size_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;

public:
    DropOldQueue(size_t max_size) : max_size_(max_size) {}

    // 满了自动丢最旧的
    void push(T&& item) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (q_.size() >= max_size_) {
            q_.pop_front();  // 丢最旧的
        }
        q_.push_back(std::move(item));
        cv_.notify_one();
    }

    // 阻塞等数据
    T pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !q_.empty(); });
        T item = std::move(q_.front());
        q_.pop_front();
        return item;
    }

    // 非阻塞取
    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return q_.size();
    }
};

#endif // DROP_OLD_QUEUE_H