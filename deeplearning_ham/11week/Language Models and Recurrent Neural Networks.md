# Language Models and Recurrent Neural Networks
https://youtu.be/OyCZvSDxMAk
## Language Model
### Language Model 이란
- Language model 이란 단어의 시퀀스(문장)에 대해서, 얼마나 자연스러운 문장인지를 확률을 이용해 예측하는 모델
- 주어진 단어의 시퀀스에 대해, 다음에 나타날 단어가 어떤 것인지를 예측하는 작업을 Language Modeling 이라고 함.
 <img src="./img/00_language model.PNG">   
 <img src="./img/01_language model.PNG">   

- 일반적인 모델은 각 나타날 확률을 구해 가장 높은 값을 선택한다. 
 <img src="./img/02_language model.PNG">   

 - 위에 W는 word를 의미하며 순차적으로 단어가 출현할 확률을 나타낼 수 있다. 

- language model은 문장의 확률 또는 단어의 등장 확률을 예측하는데 사용된다.
- 기계 번역, 음성 인식, 자동 완성 등에 활용될 수 있다.   
 <img src="./img/03_language model.PNG">   

### N-gram Language Model
- language model 중의 한 model
- Neural Network 이전에 사용되었던 Language Model
- 예측에 사용할 앞 단어들의 개수를 정하여 모델링하는 방법
- Definition : n-gram 은 n개의 연이은 단어의 뭉치
- ex) the students opened their
- - uni-grams : "the", "students", "opened", "their"
- - bi-grams : "the students", "students opened", "opened their"
- - tri-grams : "the students opened", "students opened their"
- - 4-grams : "the students opened their"  

 <img src="./img/04_language model.PNG">   
 <img src="./img/05_language model.PNG">   

- n-gram language models의 문제점
- - sparsity problems : n이 커질수록 안좋아지며, 일반적으로 n<5로 설정함
- - storage problems : n이 커지거나 corpus가 증가하면 모델의 크기가 증가함

### Neural Network Language Model
Window-based Neural Network Language Model(NNLM)
- 최초의 신경망 기반 language model은 2003년 bengio가 출판함.(curse of dimensionality을 해결하기 위해)
- 이 모델은 language model이면서 동시에 단어의 distributed representation을 학습함.
 <img src="./img/06_language model.PNG">   
 <img src="./img/07_language model.PNG">   

- 위의 모델은 단어의 embedding을 통한 sparsity problem 해결했다는 의의가 있다.
- 관측된 n-gram을 저장할 필요가 없다.

- NNLM의 문제점
- - Fixed window is too small
- - Window가 커질수록 W도 커짐 -> window 크기의 한계
- - $x_1$ 과 $x_2$ 는 완전히 다른 가중치 W가 곱해지기 때문에 No symmetry함. 

## Recurrent Neural Network(RNN)

### RNN Language Model
  <img src="./img/08_language model.PNG">   

- ex
  <img src="./img/09_language model.PNG">   

### Training RNN-LM
  <img src="./img/10_language model.PNG">   
  <img src="./img/11_language model.PNG">   

## Evaluating Language Model 
  <img src="./img/12_language model.PNG">   
  <img src="./img/13_language model.PNG">   
