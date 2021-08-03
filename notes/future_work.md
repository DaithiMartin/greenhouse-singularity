# Future work ideas

---
## 1. dynamic adjustment of reward function
The work of 2021.08.03 suggests that a well-structured reward function is as crucial to success as a good RL algorithm.
An efficient way to determine the optimal reward function would be very helpful.
It may also be helpful to dynamically scale the reward function as learning progresses. 
This is easier of course when the optimal solution is known.

The obvious path is to let another agent control the reward scaling.
Perhaps using a softmax to highlight the most import of the reward components. 

## 2. bio-mimicry of the CNS
1. error gated plasticity
2. dense network to sparse network, developmental biology