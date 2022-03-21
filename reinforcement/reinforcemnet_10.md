# 10강 Model-Free Prediction :- Monte Carlo and Temporal Difference Methods(1) : 서울대학교 이정우 교수님
이전까지는 dynamic가 모두 주어져있기 때문에 사실은 강화학습이라고 할 수는 없다.

## Monte Carlo Method
Monte Carlo methods are ways of solving the reinforcement learning problem based on averaging sample returns.
- Monte Carlo : operations involving random component
- Model-free method : 강화학습을 사용하는 이유이다.
- Require only experience $\rightarrow$ still converges to optimal behavior
- Often in practice, easy to generate samples, but difficult to obtain
- Update in an episode-by-episode sense, but not in a step-by-step(online) sense. : 단점 중간중간에 update가 불가능하다.

## Monte Carlo Prediction
- Value of a state is the expected return
- Simply average the returns observed after visits to that state
- - Converges to expected valued with more observed returns by law of large numbers
- First-visit MC method estimates value function $v_\pi$(s) as the average of the returns following first visits to s in each episode : each episode contribute only once for the value of each s.
- Every-visit MC method estimates value function $v_\pi$(s) as the average of the returns following all visits to s in each episode : each episode contribute(possibly) multiple times for the value of each s.
- Both methods converges to $v_\pi$(s) as the number of visits(or first visits) to s goes to infinity

## Monte-Carlo Policy Evaluation
- Goal : learn $v_\pi$(s) from episodes of experience under policy &pi;
$$S_1,A_1,R_2,...,S_T \sim ~ \pi$$
- Recall that the return is the total discounted reward :
$$G_t = R_{t+1}+\gamma R_{t+2}+...+\gamma^{T-t-1}R_T$$
- Recall that the value function is the expected return :
$$v_\pi(s)=E_\pi[G_\pi | S_\pi = s]$$
- Monte - Carlo policy evaluation uses empirical mean return instead of expected return  
$${V_\pi}^{MC}(s)= \frac{1}{N} \sum_{i=1}^{N} {G_t}^{(i)}$$
- where $S_t$=s and ${G_t}^{(i)}$ is the return of the $i$ th episode (or occurrence of s)

## First-Visit Monte-Carlo Policy Evaluation
- To evaluate state s
- The first time-step t that state s is visited in an episode.
- Increment counter N(s) $\leftarrow$ N(s) + 1
- Increment total return S(s) $\leftarrow$ S(s) + $G_t$
- Value is estimated by mean return V(s) = S(s)/N(s)
- By law of large numbers, V(s) $\rightarrow$ $v_\pi(s)$ as N(s) $\rightarrow$ $\infty$

## Advantages of MC over DP
- DP : one-step transition
- MC : transitions all the way to the end of an episode
- Advantages of MC
- - Ability to learn from actual experience(without a model)
- - Ability to learn from simulated experience
- - Estimating the value of a single state is independent of the number of states :  
attractive when one requires the value of only one or a subset of states.


## MC Learning of Action Value Function $Q_\pi$
- Without a model : better to use action-value instead of state value
- - With a model, a policy is obtained from the best combination of reward and next state with one-step look-ahead
- - 최종 목표는 optimal policy를 구하는 것이기 때문에 Q를 구하는 것이 중요하다.
- Both first-visit and every-visit methods can be used to estimate $q_\pi$
- Difficulty with a deterministic policy : observe returns only for one of the actions from each state.
- 이뜻의 의미는 Q(s,a)이기 때문에 episode에 나타나있는 S,A에 대해서만 Q를 계산한다. 즉 안나타난 S,A pair는 계산이 불가능하기 때문에 optimal policy를 못 얻을 수도 있다. 그렇기 때문에 deterministic policy를 쓰면 안좋다. 기본적으로 stochastic policy를 써야한다.
- - Exploration needed
- - Exploration Starts : episodes start in a state-action pair, and every pair has a nonzero probability of being selected as the start(may not be practical)
- - Generalized policy iteration (GPI) based on Exploration Starts (ES) : Convergence not yet proved (open a question)
- - Alternative : use stochastic policy to visit all state-action pairs
- - * On-policy method: improve the policy that is used to make decisions ($\varepsilon$-greedy policy)
- - * off-policy method : improve a policy different from that used to generate the data :  
데이터를 생성할 때에는 stochastic policy를 사용하고 실제로 학습할 때에는 deterministic policy를 사용한다.