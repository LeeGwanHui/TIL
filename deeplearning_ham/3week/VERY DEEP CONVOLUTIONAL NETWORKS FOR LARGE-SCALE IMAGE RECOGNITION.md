# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 논문 리뷰
Karen Simonyan & Andrew Zisserman
## Abstract
- In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting.
- 이 논문에서 보이고자 하는 것은 convolutional network depth에 따른 정확도 측면의 효과 이다.
- 또한 이건 3x3 convolution filter의 layer만을 사용하는데 이를 사용한 결과 16-19 weight layer에 현저한 향상을 살펴볼 수가 있었다.
- ImageNet Challenge 2014에 참여한 모델이다.
## introduction
- conv.net을 사용하여 image recognition에 많은 발전이 일어났다. 이를 가능케 해준 것은 GPU의 발전과 ILSVRC의 덕분이라고 할 수 있다. 
- In this paper, we address another important aspect of ConvNet architecture design – its depth. To this end, we fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small ( 3 × 3) convolution filters in all layers.
## CONVNET configurations
### architecture
- Input size : 224 x 224 RGB image.
- filter size : 3x3 filter, 1x1 filter(a linear transformation of the input channels의 역할)
- convolution stride : 1 pixel
- padding size : 1 pixel for 3x3 conv.layer (because the spatial resolution is preserved after convolution)
- pooling : Max-pooling 사용 (2x2 pixel window, with stride 2)
- the configuration of the fully connected layers is followed by three FC layers.
- the final layer is the soft-max layer.
- All hidden layers are equipped with the rectification (ReLU (Krizhevsky et al., 2012)) non-linearity.
### configurations
<img src="./img/02_VGG.PNG" width="500" height="500">  

- VGG19(E)  
<img src="./img/01_VGG.PNG" width="500" height="300">

### Discussion
- 3x3 conv.layer의 사용 
- - 3x3 receptive field의 conv.layer을 사용하면 사실상 5x5 receptive field를 커버할 수 있다. 
<img src="./img/03_VGG.PNG" width="300" height="100">  
- - 그럼 이 방법이 7x7을 하나 쓰는 것과 3x3을 3개를 써 receptive field를 커버하는 하는 것과 어떤 장점이 있을까?
- - 첫번째 : We incorporate three non-linear rectification layers instead of a single one, which makes the decision function more discriminative.
- - 두번째 : we decrease the number of parameters : 3개의 3x3 -> 3(3x3 $C^2$), 1개의 7x7 -> (7x7 $C^2$)
- - 여기서 의미하는 C는 channel을 의미한다. 
- 1x1 conv.layer의 사용
- - 첫번쨰 : to increase the non-linearity of the decision function without affecting the receptive fields of the conv.layers.
- - 

## Classification framework
### Training
- the training is carried out by optimising the multinomial logistic regression objective using mini-batch gradient descent (based on back-propagation) with momentum.
- batch size : 256
- momentum : 0.9
- L2 regularization
- learning rate : 초기 $10^{-2}$ -> then decreased by a factor of 10 when the validation set accuracy stopped improving.
- the learning was stopped after 370K iterations (74 epochs)
- We conjecture that in spite of the larger number of parameters and the greater depth of our nets compared to (Krizhevsky et al., 2012), the nets required less epochs to converge due to (a) implicit regularisation imposed by greater depth and smaller conv. filter sizes; (b) pre-initialisation of certain layers
- The initialisation of the network weights is important, since bad initialisation can stall learning due
to the instability of gradient in deep nets
- - 초기화는 pre-training 없이 random initialization procedure을 통해 구현할 수 있다.
### Training image size
- crop size : 224x224
- S is the smallest side of an isotropically-rescaled training image, from which the ConvNet input is cropped.
- S는 224이상의 어떤 값도 될 수 있다.
- two approaches for setting the training scale S.
- - the fist is to fix S,which corresponds to single-scale training (note that image content within the sampled crops can still represent multi scale image statistics)
- - The second approach to setting S is multi-scale training, where each training image is individually rescaled by randomly sampling S from a certain range [$S_{min}, S_{max}$] (we used $S_{min}$ = 256 and $S_{max}$ = 512).