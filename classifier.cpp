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
    
//     torch::jit::script::Module modulee = torch::jit::load("/data2/aditya/classifiers/LHSRHS/normalize1.pt");
    std::cout << std::fixed << std::setprecision(4);
    torch::jit::script::Module module = torch::jit::load("/data2/aditya/classifiers/LHSRHS/lhsrhs_script.zip");
    torch::data::transforms::Normalize<> normalize_transform({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    torch::Device device(torch::kCUDA);
//     torch::Tensor load_image(std::function<torch::Tensor(torch::Tensor)> transform = [] (torch::Tensor x) { return x; });
        
    Mat image_bgr, image, image1;
    image_bgr = imread("/data2/aditya/classifiers/LHSRHS/new_data801/LHS/(8)_LHS_head_lamp.png");
//     cvtColor(image_bgr, image, COLOR_BGR2RGB);
//     resize(image, image1, {448, 448});
    cvtColor(image_bgr, image_bgr, COLOR_BGR2RGB);
    image_bgr.convertTo(image_bgr, CV_32FC3, 1.0f / 255.0f);
    resize(image_bgr, image, {448, 448}, INTER_NEAREST);

    auto input_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3});
    input_tensor = input_tensor.permute({2, 0, 1});
    
//     cout<<image1<<'\n';
    
//     torch::Tensor tensor_image = torch::from_blob(image1.data, {1, 3, image1.rows, image1.cols}, at::kByte);
//     tensor_image = tensor_image.to(torch::kFloat32);
    cout<<input_tensor.sizes()<<'\n';
    
//    std::vector<torch::jit::IValue> input;
//    input.emplace_back(tensor_image);
    
//     cout<<tensor_image<<'\n';
    
//    at::Tensor inpute = modulee.forward(input).toTensor();
    module.eval();
    module.to(device);
    torch::Tensor tensor_image = normalize_transform(input_tensor).unsqueeze_(0);
//     tensor_image.to(device);

//     cout<<tensor_image1<<'\n';
    std::vector<torch::jit::IValue> input;
    input.push_back(tensor_image.to(at::kCUDA));
//     cout<<input.size()<<'\n';
//     torch::Tensor input1 = input
//     input.to(device);
    
    at::Tensor output = module.forward(input).toTensor();
    
    at::Tensor output1 = torch::softmax(output, 1);
    
    std::cout<<output1<<'\n';
    
}
    