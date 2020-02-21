# Classifiers training using fastai and inference using torchscript
  This repository contains file for training classifiers in [fastai](https://www.fast.ai/) and inference using TorchScript. Fastai is a great library, it provides a great APIs build on top of PyTorch like databunch and learner for creating dataloaders and training scheduler respectively. It uses lot's of state of the art methods for training and data creation like cyclic learning rates with one cyle policy, lot's of augmentations, etc. As we know that we python APIs of PyTorch and fastai and PyThon is slow as compared to other languagaes like C++. When we use our models in prduction, we need lesser inference time so there we can use C++. Recently, PyTorch introduced TorchScript, an intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment of C++. This repository contains on of the best methods to training a classifier and inference in TorchScript.
  
  It contains several files and notebooks which can be used for specific purposes:-                                           
  1. `notebooks/classifier.ipynb` - This is a very well commented notebook for training a classifier in fastai.
  2. `notebooks/torchscript.ipynb` - This is a notebook to convert PyTorch model to torchscript model(a serialized intermediate representation). In this notebook, I have shown how to convert a function to torchscript form which then can be used in C++.
  3. `classifier.cpp` - This is a script written in C++ for inference of classifier. In this script, we are using model which we saved using notebooks. Here, we are using OpenCV for reading image and preprocessing.
  4. `CMakeLists.txt` - cmake file for building and running the classifier written in C++.
  
  
  
