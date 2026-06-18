#include "MainReactor.h"
#include "OnnxYoloInfr.h"
#include "ThreadDecoder.h"

int main(){

    int port=8888;
    const char* model_path = "/home/zzm/lym_c/project_yolo_server/yolov8s.onnx";

    ThreadDecoder thread_decoder;
    OnnxYoloInfr yoloInf(model_path, &thread_decoder);
    MainReactor main_reactor(&thread_decoder, port);

    main_reactor.init();
    
    yoloInf.run(); // 启动推理线程
    main_reactor.run();
    
    return 0;
}

