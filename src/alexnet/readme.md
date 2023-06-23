# AlexNet

## Paper notes
* Dataset: ImageNet LSVRC-2010 (1.2 mil images)
* top-1 and top-5 error rates of 37.5% and 17.0%
* Architecture:
    * 60 mil params
    * 5 conv layers
    * 3 linear layers
    * uses relu
    * uses LRN
    * uses max-pooling
    * uses dropout
* Image preprocessing
    * Resize shorter side to 256, and crop in the middle 256*256
    * Subtracting mean from entire training set for each pixel

## Conv and MaxPool layer
```
Given:
* input tensor with h = 10 and w = 10 (.)
* kernel_size = 3 (#)
* padding = 1 (-)
* dilation = 2 (distance between #)
* stride = 4 (how much jump between steps)

step 1                      step 2
# - # - # - - - - - - -     - - - - # - # - # - - -
- . . . . . . . . . . -     - . . . . . . . . . . -
- . . . . . . . . . . -     - . . . . . . . . . . -
# . # . # . . . . . . -     - . . . # . # . # . . -
- . . . . . . . . . . -     - . . . . . . . . . . -
- . . . . . . . . . . -     - . . . . . . . . . . -
# . # . # . . . . . . -     - . . . # . # . # . . -
- . . . . . . . . . . -     - . . . . . . . . . . -
- . . . . . . . . . . -     - . . . . . . . . . . -
- . . . . . . . . . . -     - . . . . . . . . . . -
- . . . . . . . . . . -     - . . . . . . . . . . -
- - - - - - - - - - - -     - - - - - - - - - - - -
```

## Local Response Normalization (LRM)
Inspired by real biological neuron, which is called [lateral inhibition](https://en.wikipedia.org/wiki/Lateral_inhibition). LRN works by supressing neighbouring weak neuron output around the strong neuron output so that the strong one stands out more.

# Implementation details
* In train data augmentation, the correct augmentation to choose is [TenCrop](https://pytorch.org/vision/stable/generated/torchvision.transforms.TenCrop.html#torchvision.transforms.TenCrop), but I decided to use RandomCrop because TenCrop messes with batchsize due to returning as a tuple.
* Model architecture is quite different from PyTorch implementation (num conv channels), but should be close to the paper.

# References
* [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
