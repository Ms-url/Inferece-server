#include "MainReactor.h"
#include "Logger.h"

MainReactor::MainReactor(ThreadDecoder* td, int port) :port_(port), th_decoder(td)
{
    // 初始化日志系统
    Logger& logger = Logger::getInstance();
    
    // 设置日志级别（生产环境建议 INFO，开发环境 DEBUG）
    #ifdef DEBUG
        logger.setLogLevel(LogLevel::LVL_DEBUG);
    #else
        logger.setLogLevel(LogLevel::LVL_INFO);
    #endif
    
    // 启用控制台输出
    logger.setConsoleOutput(true);
    
    // 设置日志文件，启用轮转，每个文件最大10MB
    logger.setLogFile("logs/mainreactor.log", true, 10);
    
    LOG_DEBUG("========================================");
    LOG_INFO("MainReactor started with port: %d", port);
    LOG_INFO("Log level: %s", 
             #ifdef DEBUG
             "DEBUG"
             #else
             "INFO"
             #endif
    );
    LOG_DEBUG("========================================");
}

MainReactor::~MainReactor()
{
    LOG_INFO("MainReactor destructor called");
    // 从 epoll 中删除所有 conn_fd
    for (auto& fd : conn_fds_) {
        epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
        LOG_DEBUG("Removed fd %d from epoll", fd);
    }
    // close 所有 conn_fd
    for (auto& fd : conn_fds_) {
        close(fd);
        LOG_DEBUG("Closed fd %d", fd);
    }
    close(server_fd_);
    close(epoll_fd_);
    LOG_INFO("MainReactor shutdown complete");
}

/**
 * 初始化
 */
void MainReactor::init(){
    LOG_INFO("Initializing MainReactor on port %d", port_);
    
    // 创建socket
    server_fd_ = socket(AF_INET,SOCK_STREAM,0);
    if (server_fd_ < 0) {
        LOG_ERROR("socket creation failed: %s", strerror(errno));
        throw std::runtime_error("server_fd socket failed");
    }
    LOG_DEBUG("Socket created successfully, fd: %d", server_fd_);

    // 设置socker选项，允许地址重用
    int opt = 1;
    if(setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))<0){
        LOG_ERROR("setsockopt failed: %s", strerror(errno));
        throw std::runtime_error("server_fd setsockopt failed");
    }
    
    // 绑定地址和端口
    struct sockaddr_in server_addr = {};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_);
    if(bind(server_fd_,(struct sockaddr*)&server_addr, sizeof(server_addr))<0){
        LOG_ERROR("bind failed on port %d: %s", port_, strerror(errno));
        throw std::runtime_error("server_fd bind failed");
    }
    LOG_DEBUG("Bind successful on port %d", port_);
    
    // 监听
    if(listen(server_fd_, 20)<0){
        LOG_ERROR("listen failed: %s", strerror(errno));
        throw std::runtime_error("server_fd listen failed");
    }
    
    // 设置非阻塞
    fcntl(server_fd_, F_SETFL, fcntl(server_fd_, F_GETFL, 0)|O_NONBLOCK);
    LOG_DEBUG("Server socket set to non-blocking mode");

    // 创建epoll
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
        LOG_ERROR("epoll_create failed: %s", strerror(errno));
        throw std::runtime_error("epoll_create failed");
    }
    fcntl(epoll_fd_, F_SETFL, fcntl(epoll_fd_, F_GETFL)|O_NONBLOCK);// 防御性编程，可以不用这行

    struct epoll_event ev = {};
    ev.events = EPOLLET | EPOLLIN; // ET触发
    ev.data.fd = server_fd_;
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, server_fd_, &ev);

    LOG_INFO("MainReactor listening on port %d", port_);
}

/**
 * 添加fd
 */
void MainReactor::addFd(int fd){
    LOG_DEBUG("Adding fd %d to epoll", fd);
    
    //设置非阻塞
    int flags = fcntl(fd, F_GETFL, 0); // 必须用这个flags
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    
    epoll_event ev = {};
    ev.events = EPOLLET|EPOLLIN;
    ev.data.fd = fd;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
        LOG_ERROR("epoll_ctl add fd %d failed: %s", fd, strerror(errno));
        return;
    }
    {
        std::lock_guard<std::mutex> lock(conn_mtx_);
        conn_fds_.insert(fd);
    }
    LOG_DEBUG("fd %d added to epoll successfully", fd);
}

/**
 * 移除fd
 */
void MainReactor::removeFd(int fd) {
    LOG_INFO("MainReactor Removing fd %d", fd);
    
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    close(fd);
    {
        std::lock_guard<std::mutex> lock(conn_mtx_);
        conn_fds_.erase(fd);
    }
    LOG_INFO("fd %d disconnected", fd);
}


/**
 * accept
 */
void MainReactor::acceptHandled(){
    LOG_DEBUG("Handling accept");
    
    // accept 连接
    // ET触发必须读到EAGAIN
    while(true){
        sockaddr_in client_addr;
        socklen_t len = sizeof(client_addr);
        int conn_fd = accept(server_fd_, (sockaddr*) &client_addr, &len);

        if (conn_fd == -1) {
            if (errno == EAGAIN) break;  // ET触发必须读到EAGAIN
            LOG_ERROR("accept failed: %s", strerror(errno));
            break;
        }

        if (conn_fd >= 0) {
            // 添加到 epoll
            addFd(conn_fd);

            std::vector<uint8_t> new_buffer;
            connectionBuffers_[conn_fd] = std::move(new_buffer);

            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
            LOG_INFO("Accepted connection from %s, fd: %d", ip, conn_fd);
        }
    }
}

/**
 * read收流，将业务分发
 */
void MainReactor::readHandled(int fd){
    LOG_DEBUG("fd %d: data received", fd);
    
    auto it = connectionBuffers_.find(fd);
    if (it == connectionBuffers_.end()) {
        return;
    }
    
    std::vector<uint8_t>& buffer = it->second;

    char temp_buf[65536];
    while (true) {
        int ret = recv(fd, temp_buf, sizeof(temp_buf), 0);
        if (ret > 0) {
            buffer.insert(buffer.end(), temp_buf, temp_buf + ret);
        } else if (ret == 0) {
            std::cout << "连接关闭: fd=" << fd << std::endl;
            th_decoder->removeFd(fd);
            removeFd(fd);
            return;
        } else {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                std::cerr << "读取错误: fd=" << fd << std::endl;
                th_decoder->removeFd(fd);
                removeFd(fd);
                return;
            }
            break;
        }
    }
    
    if (!buffer.empty()) {
        th_decoder->decodeTaskAdd(fd, buffer);
    }
}

/**
 * run
 */
void MainReactor::run(){
    LOG_INFO("MainReactor run started");
    
    struct epoll_event events[MAX_EVENTS];
    
    while (running_) {
       int n = epoll_wait(epoll_fd_, events, MAX_EVENTS, -1);
       LOG_DEBUG("epoll_wait returned %d events", n);
       
       for(int i = 0; i < n; i++){
            // 先复制 fd 值，避免直接访问联合体成员
            int fd = events[i].data.fd;
            
            if (events[i].events & (EPOLLERR | EPOLLHUP)) {
                LOG_WARNING("fd %d: error or hangup", fd);
                if (fd != server_fd_) {
                    th_decoder->removeFd(fd);
                    removeFd(fd);
                }
                continue;
            }

            if(fd == server_fd_){
                LOG_DEBUG("Accept event on server fd");
                // accept接收新连接
                acceptHandled();
            }
            else if (events[i].events & EPOLLIN) {
                // read
                readHandled(fd);
            }
            else{
                LOG_ERROR("Unknown event on fd %d", fd);
                throw std::runtime_error("MainReactor unkown fd");
            }
       }
    }
    
    LOG_INFO("MainReactor run stopped");
}