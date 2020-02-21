# Classifiers training using fastai and inference using torchscript
  This repository contains file for training classifiers in [fastai](https://www.fast.ai/) and inference using [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html). Fastai is a great library, it provides a great APIs build on top of PyTorch like databunch and learner for creating dataloaders and training scheduler respectively. It uses lot's of state of the art methods for training and data creation like cyclic learning rates with one cyle policy, lot's of augmentations, etc. As we know that we python APIs of PyTorch and fastai and PyThon is slow as compared to other languagaes like C++. When we use our models in prduction, we need lesser inference time so there we can use C++. Recently, PyTorch introduced TorchScript, an intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment of C++. This repository contains on of the best methods to training a classifier and inference in TorchScript.
  
  It contains several files and notebooks which can be used for specific purposes:-                                           
  1. `notebooks/classifier.ipynb` - This is a very well commented notebook for training a classifier in fastai.
  2. `notebooks/torchscript.ipynb` - This is a notebook to convert PyTorch model to torchscript model(a serialized intermediate representation). In this notebook, I have shown how to convert a function to torchscript form which then can be used in C++.
  3. `classifier.cpp` - This is a script written in C++ for inference of classifier. In this script, we are using model which we saved using notebooks. Here, we are using OpenCV for reading image and preprocessing.
  4. `CMakeLists.txt` - cmake file for building and running the classifier written in C++.
  
  
## Requirements
  ``` 
      Python>=3.6
      torch >= 1.4                                                                                                             
      OpenCV >= 4.1
      torchvision >= 0.4
      fastai >= 1.0
```
  For using this project you need to follow some steps:
    1. First clone this repo in your system and then navigate to repository folder
      ``` 
      git clone https://github.com/adityak2920/Traning-and-Inference.git
      cd Traning-and-Inference
      ```
    2. Create a folder build 
      ```
      mkdir build && cd build
      ```
    3. Before procedding further we need to download latest distribution of libtorch to build our application which you can download from [here](https://pytorch.org/). After downloading unzip the folder and your directory should looks like this:
    ```
    libtorch/
            bin/
            include/
            lib/
            share/
            build-hash
            build-version
    ```
    4. Now run these commands to build the application
      ```
      cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
      cmake --build . --config Release
      ```
    5. Now you cab run your app with(in you can specify name of app in cmake file)
      ```
        ./your_app
      ```
    6. From next time you can recompile and run your app using
      ```
      make
      ./your_app
      ```
      
   For further information regarding this you can learn from following sources:-                                               
      1. [fastai coursev-3](https://course.fast.ai/)                                                                             
      2. [Intro to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)                       
      3. [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)
   
   Thanks to [PyTorch Forums](https://discuss.pytorch.org/) for resolving my doubts in implementing this project.
    
    
    
