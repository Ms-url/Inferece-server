#include <FdThreadPool.h>

/**
 * 删除tasks_中 fd 对应的task
 * remove + erase	 O(n)	每个保留元素最多移动 1 次
 * 循环 + 逐个erase  O(k×n)  每次 erase 都要移动后面所有元素
 * 快慢指针实现 remove
 */
int FdThreadPool::removeFdTask(int fd){
    int count = 0;
    // 双指针实现 remove 的效果
    auto fd_write_it = fd_duque_.begin();  // 写指针
    auto fd_read_it = fd_duque_.begin();   // 读指针
    auto task_write_it = tasks_.begin();  
    auto task_read_it = tasks_.begin();   

    std::lock_guard<std::mutex> lock(mtx_);
    while (fd_read_it != fd_duque_.end()) {
        if (*fd_read_it != fd) {           // 不是要删的 → 保留
            *fd_write_it = *fd_read_it;       // 写到写指针位置
            *task_write_it = *task_read_it;
            ++fd_write_it;                 // 写指针前进
            ++task_write_it;
            ++count;
        }
        ++task_read_it; // 读指针始终前进
        ++fd_read_it;                      
    }

    // 删除尾部多余元素
    fd_duque_.erase(fd_write_it, fd_duque_.end());
    tasks_.erase(task_write_it, tasks_.end());
    return count;
}

/**
 * 检查队列中是否有未在池中的 fd
 * 该函数会更改私有变量 new_fd_i 
 * 调用该函数前必须锁住
 */
bool FdThreadPool::newFdTask(){
    int i = 0;
    auto it = fd_duque_.begin();
    while (it != fd_duque_.end()) {
        if (fd_pool_.find(*it) == fd_pool_.end()) {  // 池中没有 
            new_fd_i_ = i;
            return true;                          
        }
        ++i;
        ++it;
    }
    return false;
}

// 构造函数
FdThreadPool::FdThreadPool(size_t n):new_fd_i_(0){
    for(size_t i=0;i<n;i++){
        workers_.emplace_back([this]{
            while(true){
                std::function<void()> task;
                int new_fd;
                {
                    std::unique_lock<std::mutex> lock(mtx_); 
                    c_v_.wait(lock,[this]{
                        return stop_||(!tasks_.empty() && newFdTask());// newFdTask()会更改私有变量 new_fd_i
                    });
                    if (stop_ && tasks_.empty()) return;
                    
                    new_fd = fd_duque_[new_fd_i_]; 
                    fd_pool_.insert(new_fd); // 新 fd 入池
                    fd_duque_.erase(fd_duque_.begin() + new_fd_i_); // 从队列中删除 fd

                    task = std::move(tasks_[new_fd_i_]); // move移动，销毁空壳
                    tasks_.erase(tasks_.begin() + new_fd_i_); // 删除 fd 对应的空壳
                }
                try{
                    task();
                }catch(...){
                    
                } // 确保从池中移除 fd
                std::lock_guard<std::mutex> lock(mtx_); 
                fd_pool_.erase(new_fd);
            }
        });
    }
}

//析构函数
FdThreadPool::~FdThreadPool(){
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    c_v_.notify_all();
    for(auto& work: workers_){
        work.join();
    }
}

/**
 * 提交任务
 * fd 和 fn 严格对齐
 */
void FdThreadPool::Submit(int fd,std::function<void()> fn){
    {
        std::lock_guard<std::mutex> lock(mtx_);
        tasks_.push_back(std::move(fn));
        fd_duque_.push_back(fd); // fd 和 fn 严格对齐
    }
    c_v_.notify_one();
}





