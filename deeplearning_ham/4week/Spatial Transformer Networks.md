# Spatial(공간) Transformer Networks 논문 리뷰

##  Abstract
- CNNs are still limited by the lack of ability to be spatially invariant to the input data in a computationally and parameter efficient manner.
- 이 논문에서는 Spatial Transformer이라고 명명한 learnable module을 소개할 것이다. 
- 이 module은 data의 네트워크 내부에서 공간적인 취급이 가능토록 해준다.
- 이 module은 이미 존재하는 Convolutaional architecture에 적용할 수 있고 스스로 feature map과 조건은 공간적으로 변형시켜줘준다.
- We show that the use of spatial transformers results in models which learn invariance to translation, scale, rotation and more generic warping

## introduction
- A desirable property of a system which is able to reason about images is to disentangle object pose and part deformation from texture and shape.
- The introduction of local max-pooling layers in CNNs has helped to satisfy this property by allowing a network to be somewhat spatially invariant to the position of features.
- 하지만 maxpooling의 경우에는 3x3이라던지 그 공간적인 특징이 이미 정해져 있고 작기 때문에 deep hierarchy에서만 잘 적용되며 중간 feature map에서는 input data에 대한 공간적 불변성을 보장하지 않는다.
- 이 논문에서는 Spatial Transformer module을 제한하는데 표준 CNN model에 포함되어 공간적인 불변성을 보장해줄 것이다.
- The action of the spatial transformer is conditioned on individual data samples, with the appropriate behaviour learnt during training for the task in question (without extra supervision).
- the spatial transformer module is a dynamic mechanism that can actively spatially transform an image (or a feature map) 
- This allows networks which include spatial transformers to not only select regions of an image that are most relevant (attention), but also to transform those regions to a canonical, expected pose to simplify recognition in the following layers.
<img src="./img/00_figure.PNG">   

## Related Work
- 2D affine에 대해  
이동, 스케일링 등의 변환 전체를 식으로 표현한 것을 말한다.  
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=baejun_k&logNo=221207284223

