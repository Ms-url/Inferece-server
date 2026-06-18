#pragma once
#include <unistd.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <stdexcept>
#include <fcntl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>
#include <set>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "OnnxYoloInfr.h"

class MainReactor
{
private:
    
    /* data */
    static const int BUFFER_SIZE = 4096;
    static const int MAX_EVENTS = 1024;

    int port_;
    int epoll_fd_;
    int server_fd_;

    std::map<int, std::vector<uint8_t>> connectionBuffers_; // <fd, buffer>

    OnnxYoloInfr* onnx_yolo_;

    std::mutex conn_mtx_;
    std::set<int> conn_fds_; // 保存客户端 fd

    bool running_ = true;

    void acceptHandled(); // accept接收
    void readHandled(int fd); // read收流
    void addFd(int fd);
    void removeFd(int fd);
    bool findValidNALU(int fd, uint32_t& out_len, size_t& out_offset);

public:
    MainReactor(OnnxYoloInfr* oy , int port );
    ~MainReactor();
    void init(); // 初始化
    void run(); // epoll线程：accpet连接，注册事件到从Reactor
};

