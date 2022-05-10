# LSTM
## RNN
- 인간은 매순간 처음부터 생각하지 않는다.
- 예를 들면 영화의 일어난 event의 종류를 분류하고자 할 때 인간은 전에 있던 생각을 떨쳐버리지 않고 다시 처음부터 생각하지 않을 것이다.
- 그러나 neural network는 이같이 전의 event의 의미로 다음의 event를 파악하는 방법이 명확하지 않았다.
- RNN이 이 문제를 해결하였다.   
<img src="./img/16_language model.PNG">   
- 위와 같은 구조를 통해 정보를 저장할 수 있다.
- 다음과 같은 A loop는 한 step에서 next step으로 정보를 전달할 수 있도록한다.
- RNN은 이해하기 힘든 구조지만 아래와 같이 바꿔서 생각하면 이해를 도울 수 있다.
<img src="./img/17_language model.PNG">   
- This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists.
- Essential to these successes is the use of “LSTMs,” a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version. 

## The Problem of Long-Term Dependencies
- RNN의 특징은 현재의 frame을 이해하기 위해 이전의 framed을 사용한다는 점이다. -> 단 상황에 의존한다.
- where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information.(당연한 것으로 알 수 있을때)
<img src="./img/18_language model.PNG">   
- It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.(문맥을 파악해서 결론을 내려야 할때)
- Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.
<img src="./img/19_language model.PNG">   

- In theory, RNNs are absolutely capable of handling such “long-term dependencies.
- Sadly, in practice, RNNs don’t seem to be able to learn them.
- Thankfully, LSTMs don’t have this problem!
- 위의 내용을 요약하자면 long-term dependencies를 이론적으로 RNN에서 다루는 것은 가능하지만 실제에서는 불가능하다. LSTM은 이 같은 문제가 생기지 않는다.

## LSTM Netowrks(Long Short Term Memory networks)
- Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.
- Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!  
<img src="./img/20_language model.PNG">   

- RNN은 같은 반복구조를 위와 같이 가지고 있다.

- LSTMs also have this chain like structure, but the repeating module has a different structure.  
<img src="./img/21_language model.PNG">   

- Instead of having a single neural network layer, there are four, interacting in a very special way.
- 표기법은 아래와 같다.   
<img src="./img/22_language model.PNG">   

## The Core Idea Behind LSTMs
- The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.  
<img src="./img/23_language model.PNG">   

- The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.
- Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.  
<img src="./img/24_language model.PNG">   

- gate는 sigmoid layer을 사용하여 구현하는데 0의 값은 아무것도 통과시키지 않는 것이고 1은 모든 것을 통과시키는 것이다.

## Step-by-Step LSTM Walk Through
- The first step in our LSTM is to decide what information we’re going to throw away from the cell state.
- - forget gate layer에서 이루어짐.  
<img src="./img/25_language model.PNG">   

- The next step is to decide what new information we’re going to store in the cell state.(2part로 나뉨)
- - input gate layer : decides which values we'll update
- - tanh layer : creates a vector of new candidate values $\tilde{C_t}$
<img src="./img/26_language model.PNG">   

- t’s now time to update the old cell state, $C_{t−1}$, into the new cell state $C_t$. The previous steps already decided what to do, we just need to actually do it.
<img src="./img/27_language model.PNG">   

- This is the new candidate values, scaled by how much we decided to update each state value.(위의 식을 의미)

- Finally, we need to decide what we’re going to output.
- - First, we run a sigmoid layer which decides what parts of the cell state we’re going to output.
- - Then, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.
<img src="./img/28_language model.PNG">   

## Variants on Long Short Term Memory
- LSTM은 모델마다 차이점이 있는데 여기서는 주목할 만한 몇개를 살펴보자.
- One popular LSTM variant, introduced by Gers & Schmidhuber (2000), is adding “peephole connections.” This means that we let the gate layers look at the cell state.
<img src="./img/29_language model.PNG">   

- Another variation is to use coupled forget and input gates.
- - We only forget when we’re going to input something in its place. We only input new values to the state when we forget something older.
<img src="./img/30_language model.PNG">   

- A  slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by Cho, et al. (2014). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes.
<img src="./img/31_language model.PNG">   

## Conclusion
- LSTM은 대단한 모델이다.
- LSTM 의 다음 step은 attention이다. (Yes! There is a next step and it’s attention!)
- - The idea is to let every step of an RNN pick information to look at from some larger collection of information.
- generative model 혹은 Grid LSTM 등 attention 뿐만 아니라 RNN도 많이 발젆고 있다.