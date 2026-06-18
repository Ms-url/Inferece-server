#include "MainReactor.h"
#include "OnnxYoloInfr.h"

int main(){

    int port=8888;
    const char* model_path = "/home/zzm/lym_c/project_yolo_server_1/yolov8s.onnx";

    OnnxYoloInfr yoloInf(model_path);
    MainReactor main_reactor(&yoloInf, port);

    main_reactor.init();
    
    yoloInf.run(); // 启动推理线程
    main_reactor.run();
    
    return 0;
}

