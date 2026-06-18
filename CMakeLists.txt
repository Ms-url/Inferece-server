cmake_minimum_required(VERSION 3.10)
project(YoloServer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DDEBUG)
# 头文路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# 指定 ONNX Runtime 路径
set(ONNXRUNTIME_ROOT /home/zzm/anaconda3/envs/lymc)
include_directories(${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
link_directories(${ONNXRUNTIME_ROOT}/lib)

find_package(OpenCV REQUIRED)
# find_package(onnxruntime REQUIRED)

add_executable(yolo_server
    src/main.cpp
    src/OnnxYoloInfr.cpp
    src/MainReactor.cpp
    src/FdThreadPool.cpp
    src/ThreadDecoder.cpp
)

target_link_libraries(yolo_server
    ${OpenCV_LIBS}
    onnxruntime
    avformat avcodec avutil swscale
    pthread
)
# add_executable(yolo_server
#     src/main.cpp
# )

# target_link_libraries(yolo_server
#     ${OpenCV_LIBS}
#     avformat avcodec avutil swscale
#     pthread
# )