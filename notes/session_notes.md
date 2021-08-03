# Session Notes

---
## 2021.8.03

It would appear that I finally figured out what was going wrong with training pipeline. 
The issue was multifaceted. 
1. Insufficient reward signal.

The reward signal from the environment was not strong enough for the agent to learn. 
It was initially MSE(ideal_temp, internal_temp).
I changed it to include a large reward if ideal_temp == internal_temp.
This provided a much stronger signal for the desired behavior.

**This suggests that a well-structured reward function is almost as critical as a good RL algorithm.**

2. Poor agent hyper parameter choices

The other main problem was the choice of RL agent hyper parameters.
Primarily the epsilon greedy decay with the final stable epsilon set too high.
This forces the agent to perform random behavior even if it knows better.
A sufficiently low final epsilon, or perhaps 0, is required for SARSA type algorithms if precise action is required.

The agent appears to learn the ideal behavior within the simple environment. 
The next step will be to re-incorporate the heat loss and conservation of energy in the gym environment.
