# Decoupling Representation and Classifier for long-tailed recognition

## Abstract 
- long-tail distribution은 class imbalance problem을 어떻게 다루어야 하는지에 관한 문제를 말한다.
- 기존의 연구의 경우에는 class-balancing 전략을 주로 사용하였다. 
- - 예를 들면 loss re-weighting , data re-sampling, transfer learning from head-to tail-classes
- 기존 연구 대부분은 공통적으로 learning representation과 classifier의 scheme을 사용한다.
- 이 연구에서는 representation learning과 classification의 learning procedure을 분단할 것이다.
- The findings are surprising
- - data imbalance might not be an issue in learning high-quality representations
- - with representations learned with the simplest instance-balanced (natural) sampling, it is also possible to achieve strong long-tailed recognition ability by adjusting only the classifier

## Introduction
- ImageNet Challenge의 dataset은 인위적으로 balancing을 진항한 dataset으로 각각의 object/class instance의 수가 일정하다.
- 그러나 실제로는 그렇지 않으며 그렇기 때문에 좋은 성능을 내는 모델도 실제로는 정확도가 확연히 떨어진다.
- long-tailed date의 가장 큰 문제는 training 단계에서 instance-rich(or head) classes가 지배한다는 점이다.
- - 이렇게 되면 head classes에 해당하는 것에 대한 정확도는 뀌어나지만 tail classes에 대한 정확도는 떨어진다는 단점이 생긴다.
- 이를 위한 접근 방법중 하나로는 re-sample the date 혹은 design specific loss function 방법이 있다. 
- 또다른 방향으로는 tail classes에 대한 인식 성능을 향상 시키는 방법으로 transferring knowledge from the head classes을 사용하는 방법이다.
- 가장 흔한 접근은 sampling strtegies,losses functions 또는 보다 복잡한 model을 design하는 것이다.
- 앞서 언급한 대부분의 방법은 data representation과 함께 인식에 사용되는 classifier을 공동으로 학습한다.
- 그러나 이런 방법은 어떻게 lon-tailed recognition 능력이 향상되었는지 아는 것이 unclear하다.
- - better representation or shifting classifier decision boundaries을 통해 data의 imbalance을 더 좋게 만들기 때문인지 알 수 없다.
- 그래서 이 연구에서는 representation learning과 classification을 분해한다.
- - learning representation에서
- - * the model is exposed to the training instances and trained through different sampling strategies or losses.
- - classification에서
- - * upon the learned representations, the model recognizes the long-tailed classes through various classifiers.

1. we train model to learn representations with different sampling strategies, including the standard instance-based sampling, class-balanced sampling and a mixture of them
2. we study three different basic approaches to obtain a classifier with balanced decision boundaries, on top of the learned representations
- - re-training the parametric linear classifier in a class-balancing manner(i.e., re-sampling)
- - non-parametric nearest class mean classifier, which classifies the data based on their closest class-specific mean representations from the training set
- - normalizing  the classifier weights, which adjusts the weight magnitude directly to be more balanced, adding a temperature to modulate the normalization procedure.
- contribution은 아래와 같다.
- -  instance-balanced sampling learns the best and most generalizable representations.
- -  It is advantageous in long-tailed recognition to re-adjust the decision boundaries speci fied by the jointly learned classifier during representation learning -> weight normalization 
- - By applying the decoupled learning scheme to standard networks (e.g., ResNeXt), we achieve significantly higher accuracy than well established state-of-the-art methods (dif ferent sampling strategies, new loss designs and other complex modules) on multiple long tailed recognition benchmark datasets

## Related work
long-tail problem 연구는 3가지 방향으로 계속되어졌다.
- Data distribution re-balancing : 
- - These methods include over-sampling for the minority classes, under-sampling for the mahority classes.
- - class-balanced sampling based on the number of samples for each class.
- Class-balanced Losses 
- - 각 class에 대한 training samples에 다른 losses를 할당하는 방법이 있다.
- - The loss can vary at class-level for matching a given data distribution and improving the generalization of tail classes.
- Transfer learning from head - to tail classes
- - Transfer-learning based methods address the issue of imbalanced training data by transferring features learned from head classes with abundant training instances to under-represented tail classes.

## Learning representations for long-tailed recognition
- notation
- - X = ${x_i, y_i}$ is a training set
- - $y_i$ is the label for data point $x_i$ 
- - $n_j$ 는 class j에 대한 training sample의 수를 의미한다. 
- - n 은 전체 traning sample의 수를 의미한다. n = $\Sigma_{j=1}^C n_j $
- - 일반성을 잃지 않고 sample 수에 따라 내림차순으로 정리한다고 생각해보자. 즉 if i<j , then $n_i \geq n_j$ 
- - $n_1$ >> $n_C$ : long-tail distribution
- - f(x; $\theta$) = z 은 x에 대한 representation으로써 x를 deep CNN model parameter $\theta$ 와 함께 통과시킨 것을 의미한다.
- - the final class prediction $\tilde{y}$ is given by a classifier function g, such that $\tilde{y}$ = argmax g(z)
- - g는 대체로 linear classifier로 g(z) = $W^Tz+b$ 로 나타낼 수 있다. 여기서 W는 classifier weight matrix이다. b는 bias을 의미한다.
- Sampling strategie
- - 