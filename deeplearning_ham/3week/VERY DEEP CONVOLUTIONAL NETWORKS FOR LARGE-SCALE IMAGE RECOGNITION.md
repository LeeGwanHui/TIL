# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 논문 리뷰

## Abstract
- In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting.

### introduction
- In this paper, we address another important aspect of ConvNet architecture design – its depth. To this end, we fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small ( 3 × 3) convolution filters in all layers.

### CONVNET configurations
- Input size : 224 x 224 RGB image.
- filter size : 3x3 filter, 1x1 filter(a linear transformation of the input channels의 역할을 한다.)
- convolution stride : 1 pixel
- padding size : 1 pixel for 3x3 conv.layer (because the spatial resolution is preserved after convolution)
- pooling : Max-pooling 사용 (2x2 pixel window, with stride 2)
- the configuration of the fully connected layers is the same in all networks
- All hidden layers are equipped with the rectification (ReLU (Krizhevsky et al., 2012)) non-linearity.
 ![title](./img/01_VGG.PNG)
