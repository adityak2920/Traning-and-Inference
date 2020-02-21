// including all the header files needed for running this script
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
    // Setting precision to 4 decimal places.
    std::cout << std::fixed << std::setprecision(4);

    // Loading the trained classifier model for prediction.
    torch::jit::script::Module module = torch::jit::load("path to model");
    // Initialising Normalization transform for preprocessing
    torch::data::transforms::Normalize<> normalize_transform({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    // Creating a variabe CUDA for running on GPU
    torch::Device device(torch::kCUDA);

    // Creating a mat object and then reading image using OpenCV.    
    Mat image_bgr, image, image1;
    image_bgr = imread("path to image");
    // As OpenCV reads image in BGR so converting to RGB, normaizing and then resizing the image
    cvtColor(image_bgr, image_bgr, COLOR_BGR2RGB);
    image_bgr.convertTo(image_bgr, CV_32FC3, 1.0f / 255.0f);
    resize(image_bgr, image, {448, 448}, INTER_NEAREST);

    // It basically exposes the given data(in our case Mat object) as tensor without taking ownership of the original data
    auto input_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3});
    // Converts dimension from [height, width, channels] to [channels, height, width]
    input_tensor = input_tensor.permute({2, 0, 1});
    
    cout<<input_tensor.sizes()<<'\n';

    // Shifting the model from training to evaluation model, then shifting the model to CUDA(GPU) and normalizing using previously initialised std and var.
    module.eval();
    module.to(device);
    torch::Tensor tensor_image = normalize_transform(input_tensor).unsqueeze_(0);

    // Creating a vector to store multidimensional data and then storing.
    std::vector<torch::jit::IValue> input;
    input.push_back(tensor_image.to(at::kCUDA));

    // passing the input to model and then applying softmax
    at::Tensor output = module.forward(input).toTensor();
    at::Tensor output1 = torch::softmax(output, 1);
    
    std::cout<<output1<<'\n';
    
}
    