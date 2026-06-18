#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <sys/stat.h>
#include <unistd.h>

enum class LogLevel {
    LVL_DEBUG = 0,    // 添加前缀
    LVL_INFO = 1,
    LVL_WARNING = 2,
    LVL_ERROR = 3,
    LVL_FATAL = 4
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLogLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_level_ = level;
    }

    void setLogFile(const std::string& filename, bool enableRotation = true, size_t maxSizeMB = 10) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_filename_ = filename;
        enable_rotation_ = enableRotation;
        max_file_size_ = maxSizeMB * 1024 * 1024; // 转换为字节
        
        openLogFile();
    }

    void setConsoleOutput(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        console_output_ = enable;
    }

    template<typename... Args>
    void log(LogLevel level, const std::string& format, Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (level < log_level_) {
            return;
        }

        std::string message = formatString(format, std::forward<Args>(args)...);
        std::string log_entry = formatLogEntry(level, message);
        
        // 输出到控制台
        if (console_output_) {
            std::cout << log_entry;
        }
        
        // 输出到文件
        if (log_file_.is_open()) {
            log_file_ << log_entry;
            log_file_.flush();
            
            // 检查是否需要轮转
            if (enable_rotation_) {
                checkAndRotate();
            }
        }
    }

    // 便捷方法
    template<typename... Args>
    void debug(const std::string& format, Args&&... args) {
        log(LogLevel::LVL_DEBUG, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(const std::string& format, Args&&... args) {
        log(LogLevel::LVL_INFO, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warning(const std::string& format, Args&&... args) {
        log(LogLevel::LVL_WARNING, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(const std::string& format, Args&&... args) {
        log(LogLevel::LVL_ERROR, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void fatal(const std::string& format, Args&&... args) {
        log(LogLevel::LVL_FATAL, format, std::forward<Args>(args)...);
    }

    // 强制刷新日志
    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (log_file_.is_open()) {
            log_file_.flush();
        }
    }

private:
    Logger() : log_level_(LogLevel::LVL_INFO), 
               console_output_(true),
               enable_rotation_(false),
               max_file_size_(10 * 1024 * 1024) {
        // 默认不启用轮转
    }
    
    ~Logger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void openLogFile() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
        
        // 创建目录（如果不存在）
        size_t pos = log_filename_.find_last_of('/');
        if (pos != std::string::npos) {
            std::string dir = log_filename_.substr(0, pos);
            mkdir(dir.c_str(), 0755);
        }
        
        log_file_.open(log_filename_, std::ios::out | std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Failed to open log file: " << log_filename_ << std::endl;
        }
    }

    void checkAndRotate() {
        if (!log_file_.is_open()) return;
        
        // 获取当前文件大小
        std::streampos current_pos = log_file_.tellp();
        if (current_pos >= static_cast<std::streampos>(max_file_size_)) {
            log_file_.close();
            
            // 轮转日志文件
            rotateLogFile();
            
            // 重新打开日志文件
            openLogFile();
        }
    }

    void rotateLogFile() {
        // 生成带时间戳的备份文件名
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&time_t_now);
        
        std::stringstream ss;
        ss << log_filename_ << "."
           << std::setfill('0') << std::setw(4) << (tm->tm_year + 1900)
           << std::setfill('0') << std::setw(2) << (tm->tm_mon + 1)
           << std::setfill('0') << std::setw(2) << tm->tm_mday
           << "_"
           << std::setfill('0') << std::setw(2) << tm->tm_hour
           << std::setfill('0') << std::setw(2) << tm->tm_min
           << std::setfill('0') << std::setw(2) << tm->tm_sec;
        
        std::string backup_name = ss.str();
        
        // 重命名当前日志文件
        if (rename(log_filename_.c_str(), backup_name.c_str()) != 0) {
            std::cerr << "Failed to rotate log file: " << errno << std::endl;
        } else {
            std::cout << "Log rotated to: " << backup_name << std::endl;
        }
    }

    template<typename... Args>
    std::string formatString(const std::string& format, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            return format;  // 空包直接返回原字符串
        } else {
            int size = snprintf(nullptr, 0, format.c_str(), std::forward<Args>(args)...);
            if (size <= 0) return format;
            
            std::string result(size + 1, '\0');
            snprintf(&result[0], size + 1, format.c_str(), std::forward<Args>(args)...);
            result.resize(size);
            return result;
        }
    }
    
    std::string getCurrentTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::string getLevelString(LogLevel level) {
        switch (level) {
            case LogLevel::LVL_DEBUG:   return "DEBUG";
            case LogLevel::LVL_INFO:    return "INFO";
            case LogLevel::LVL_WARNING: return "WARNING";
            case LogLevel::LVL_ERROR:   return "ERROR";
            case LogLevel::LVL_FATAL:   return "FATAL";
            default:                return "UNKNOWN";
        }
    }

    std::string formatLogEntry(LogLevel level, const std::string& message) {
        std::stringstream ss;
        ss << "[" << getCurrentTime() << "] ";
        ss << "[" << getLevelString(level) << "] ";
        ss << "[" << getpid() << "] ";  // 添加进程ID
        ss << message;
        if (!message.empty() && message.back() != '\n') {
            ss << std::endl;
        }
        return ss.str();
    }

    std::ofstream log_file_;
    std::string log_filename_;
    LogLevel log_level_;
    bool console_output_;
    bool enable_rotation_;
    size_t max_file_size_;
    std::mutex mutex_;
};

// 宏定义简化使用
#define LOG_DEBUG(...)   Logger::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...)    Logger::getInstance().info(__VA_ARGS__)
#define LOG_WARNING(...) Logger::getInstance().warning(__VA_ARGS__)
#define LOG_ERROR(...)   Logger::getInstance().error(__VA_ARGS__)
#define LOG_FATAL(...)   Logger::getInstance().fatal(__VA_ARGS__)
#define LOG_FLUSH()      Logger::getInstance().flush()

#endif // LOGGER_H