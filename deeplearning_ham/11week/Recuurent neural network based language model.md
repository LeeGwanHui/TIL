# Recurrent neural network based language model(RNNLM)

## Abstract
- 이 논문에서는 RNN 기반의 language model을 제안하였고 기존의 모델보다 perplexity를 50% 줄인 결과를 얻을 수 있었다.
- We provide ample empiri cal evidence to suggest that connectionist language models are superior to standard n-gram techniques, except their high computational (training) complexity.

## Introduction
- The goal of statistical language modeling is to predict the next word in textual data given context; thus we are dealing with sequential data prediction problem when constructing language models.
- -	Statistical language modeling의 목표는 주어진 context에서 textual data의 다음 word를 예측하는 것이므로, language models를 구성할 때 sequential data prediction problem을 다루어야 한다.
- Even the most widely used and general models, based on n gram statistics, assume that language consists of sequences of atomic symbols - words - that form sentences, and where the end of sentence symbol plays important and very special role.
- - Statistical models를 얻기 위한 많은 방법들이 제안되었지만, 가장 주로 사용되고 일반적인 models는 n-gram statistics를 기반으로 하여 언어는 문장을 구성하는 words의 sequences로 구성되고, 문장 symbol의 끝부분(where the end of sentence symbol)은 중요하고 매우 특별한 역할을 수행한다고 가정한다.
- -	하지만 simple N-gram models에 비해 language modeling에서 상당한 진전이 있었는지는 의문이 남는다.
- -	만약 models이 sequential data를 더 잘 측정하는 능력에 의한 발전을 측정하게 된다면, 이에 대한 대답은 cache models와 class-based models의 도입으로 인해 상당한 개선이 달성되었다는 것이다.
- Real-world speech recognition이나 machine translation system에서의 language models는 많은 양의 data로 설계되고, 통상적으로 더 많은 data가 필요하다는 관념이 있다.
- 기존 연구를 통해 설계된 models는 복잡하고 제한된 양의 training data에서만 잘 작동하며, 제안된 많은 advanced language modeling techniques는 약간의 개선만 이루어냈으며, 실제로 사용되는 경우는 거의 없다.

## Model description
- 이미 language modeling으로써 neural network을 사용한 모델은 Bengio가 제안하였다.
- - 그는 fixed length context와 함께 feedforward neural network을 사용하였다.
- Neural network를 이용한 sequential data modeling은 이미 이전에 제안되었으며, fixed-length context를 이용한 feedforward neural networks를 통해 class-based model를 포함한 다른 technique에 기반한 몇 개의 다른 models들의 조합보다 성능이 더 좋다는 것을 확인하였다.
- 하지만 feedforward network은 결점이 존재하는데, 오직 5개에서 10개의 fixed length context를 보고 다음 word를 예측한다는 것이다.
- 사람은 더 긴 context를 보고 성공적인 예측을 한다.
- Also, cache models provide complementary information to neural network models, so it is natural to think about a model that would encode temporal information implicitly for contexts with arbitrary lengths.
 <img src="./img/14_language model.PNG">   

- Recurrent neural network는 limited size of context을 사용하지 않는다. 
- -	Recurrent neural networks는 limited size의 context를 이용하지 않으며, recurrent connection을 이용하여 정보들이 임의의 시간동안 networks 안에서 순환할 수 있다.
- - 하지만 stochastic gradient descent를 통해 long-term dependencies를 학습하는 것은 어려울 수 있다.
- 저자들은 recurrent의 장점을 고려하여 simple recurrent neural network로 불리는 architecture를 사용하였다.
- 이는 recurrent neural network에서 구현과 학습 과정이 가장 쉬운 version이다.
- input layer x, hidden layer s , output layer y로 표기할 것이다.
- 그래서 time 5에서 network의 input은 x(t), output y(t), hidden layer은 s(t)로 표기할 것이다.
- 즉 hidden layer의 상태가 s(t)일 때, input vector x(t)는 current word를 표현하는 w와 time t-1에서 context layer s의 neuron에서의 output인 s(t-1)을 concatenating하여 나타낸다.
$$ x(t) = w(t) + s(t-1) $$
$$ s_j(t) = f(\Sigma_i x_i(t)u_{ji})$$
$$ y_k(t) = g(\Sigma_j s_j(t)v_{kj})$$
- f(z) is sigmoid activation function
$$ f(z) = \frac{1}{1+e^{-x}}$$
$$ g(z_m) = \frac{e^{z_m}{\Sigma_k e^{z_k}}}$$
-	다음으로는 training detail에 대해 설명되어있다.
- s(0)는 0.1과 같은 작은 수로 이루어진 vector로 설정해서 initialization해준다.
- -	만약 data 수가 매우 많다면, initialization은 별 상관 없다.
- x(t)는 1-of-N coding(one-hot encoding과 동일)된 time t에서의 word
- -	이 때 x의 size는 vocabulary V의 size와 동일(보통 3만~20만)
- -	Context layer s의 size는 보통 30~500
- -	이때 context layer의 size는 training data의 양을 고려해야 한다.

- Weights는 zero-mean and 0.1 variance random Gaussian noize로 initialization
- SGD 사용
- Learning rate = 0.1
- Learning decay 사용. Validation data의 log-likelihood에 변화 없으면, 절반으로 감소
- 한 번 더 개선이 없으면 training 종료
- 일반적으로 10 – 20 epochs에서 training이 종료되었음.
-	Regularization은 사용하지 않음. 

- output layer y(t)는 probability distribution of next word given previous word w(t) and context s(t-1)을 표현한다.
- Cross entropy loss 사용
- - error(t) = desired(t) - y(t) 
- - desired is a vector using 1-of-N coding representing the word that should have been predicted in a particular context and y(t) is the actual output from the network.

- 저자들은 long term memory는 context units(즉, neuron?)가 활성화되는 것에 있는 것이 아니라 synapses(즉, neuron과의 연결?) 자체에 있다고 한다. 따라서 network는 testing phase에서도 학습해야 한다고 주장한다. 이를 dynamic model이라고 부른다.
- Dynamic model은 fixed learning rate 0.1을 사용하며, test data에 대해 한번만 학습한다.
- 최적화된 solution은 아니지만, 이를 통해 perplexity가 감소하는 것을 확인하였다.
- 이 과정을 통해 model은 new domains에서도 자동적으로 적응할 수 있게 된다.
- 저자들이 제안한 model은 truncated backpropagation through time with tau=1로 학습된 것이다.
- -	이는 최적의 방법이 아니며, 일반적으로는 backpropagation through time(BPTT) algorithm을 사용한다.
- RNN LM과 이전의 feedforward neural networks를 비교해보면, RNN LM은 hidden layer의 size만 설정하면 되지만, feedforward neural networks는 words를 projection할 dimension의 size, hidden layer의 size와 context-length를 설정해주어야 한다.

### Optimization
- Performance를 개선하기 위해, 저자들은 training text에서 적게 나타나는 모든 단어들을 special rare token으로 통합하였다. 이는 threshold를 설정하여 이보다 적게 나타나는 단어들에 해당한다.
- 이를 통해, word-probabilities는 다음과 같이 계산된다.  
 <img src="./img/15_language model.PNG">   

- $C_rare$ 는 vocabulary에서 threshold보다 적게 발생하는 단어들의 개수이다.
- 모든 rare words는 동일하게 다루어지며, 즉 확률은 uniformly distributed이다.

## WSJ experiments
- 저자들은 몇 개의 standard speech recognition tasks에서 RNN LM의 performance를 측정하였다.

## Conclusion and future work
-	저자들은 RNN이 backoff models보다 data의 수가 적음에도 불구하고 더 좋은 성능을 보이는 것을 확인하였다.
-	Word error rate(WER)가 더 작고, data의 수가 더 적음에도 더 좋은 성능을 보이는 것을 확인하였다.
-	이를 통해 저자들은 language model은 단지 n-grams을 세는 것이고, 더 좋은 성능을 이끌어내는 유일한 방법은 새로운 training data를 사용하는 것뿐이라는 속설을 깼다.
-	또한 저자들은 on-line learning(본 논문에서 dynamic models라고 부르기도 하였다.)의 효과도 확인하였다.
-	Online-learning은 cache-like and trigger-like information을 얻는 자연스러운 방법을 제공하기에 더 연구해볼 가치가 있다고 주장하였다.
-	새로운 정보를 얻는다는 점에서도 online-learning은 필수적이라고 한다.
-	RNN의 BPTT algorithm 또한 더 많은 성능 개선을 이끌어낼 수 있다고 주장하였다.
-	하지만 cache model이 BPTT를 통해 training된 dynamic models에게도 complementary information을 제공하는 것으로 보아 simple RNN이 long context information을 진정으로 포착해내지는 못한다고 보았다.
-	저자들은 RNN model이 ML, data compression, cognitive sciences research와 연관이 있기에 연구해야 한다고 주장하며 마무리한다.
