#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include "load_image.h"


using namespace cv;
using namespace std;

int main() {
    
    std::cout << std::fixed << std::setprecision(4);
    torch::jit::script::Module module = torch::jit::load("path to model");
    torch::data::transforms::Normalize<> normalize_transform({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    torch::Device device(torch::kCUDA);
        
    Mat image_bgr, image, image1;
    image_bgr = imread("path to image");

    cvtColor(image_bgr, image_bgr, COLOR_BGR2RGB);
    image_bgr.convertTo(image_bgr, CV_32FC3, 1.0f / 255.0f);
    resize(image_bgr, image, {448, 448}, INTER_NEAREST);

    auto input_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3});
    input_tensor = input_tensor.permute({2, 0, 1});
    

    cout<<input_tensor.sizes()<<'\n';

    module.eval();
    module.to(device);
    torch::Tensor tensor_image = normalize_transform(input_tensor).unsqueeze_(0);

    std::vector<torch::jit::IValue> input;
    input.push_back(tensor_image.to(at::kCUDA));
    
    at::Tensor output = module.forward(input).toTensor();
    
    at::Tensor output1 = torch::softmax(output, 1);
    
    std::cout<<output1<<'\n';
    
}
    