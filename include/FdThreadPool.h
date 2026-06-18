#pragma once
#include <thread>
#include <vector>
#include <queue>
#include <unordered_set>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
/**
 * 优先将未在池中的 fd 任务加入池中
 */
class FdThreadPool
{
private:
    /* data */
    int new_fd_i_; // 新fd的位置
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    std::deque<int> fd_duque_;          // fd 等待队列从前向后遍历
    std::unordered_set<int> fd_pool_;   // fd 池，O(1)查重
    std::mutex mtx_;
    std::condition_variable c_v_;
    std::atomic<bool> stop_ = {false};

public:
    FdThreadPool(size_t n);
    ~FdThreadPool();
    bool newFdTask();
    int removeFdTask(int fd); // 删除tasks_中 fd 对应的task
    void Submit(int fd, std::function<void()>); //提交任务到线程池
};