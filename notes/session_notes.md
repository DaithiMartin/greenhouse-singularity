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

Incorporated heat loss and but found more bugs and fixed and number of issues. 

- fixed state update ordering in environment
  - this was causing major learning issues
  
---
## 2021.8.04
- fixed history recording for temp, reward and actions
- changed reward function to incorporate tolerance 

---
## 2021.8.05
- updated env.render() to with tolerances
- fixed U value in temperature update

---
## 2021.8.07
- tweaked model and agent hyper parameters

Under the basic scenario of constant outside time and conductive heat exchange, the agent finds the ideal policy.

The next step is to incorporate diurnal temp swings.

---
## 2021.08.11

Re-incorporating the diurnal temperature swings is proving challenging.

This initially appears to be due to improper representation of state in the observation. 
The observation does not contain enough information or perhaps the wrong information.

old observation: [time, outside_temp, inside_temp, ideal_temp]

For example removing time component from the observation, the model increases' performance.

While state/observation may still be an issue, some other bullshit is going on.

After much investigation, the agent appears to form equilibrium with the environment action.
It does not however push the temp to the ideal range....

It also appears that my coding of the agent is not the issue. I plugged in a reference DQN, and it performed similarly. 
I am done for the day.


---
I am Odin, not Sisyphus.

1. Bug in SARSA algorithm

A major problem was the implementation of the PDQN_Agent. 
The Q value estimates and Q value targets were being calculated incorrectly. 
This fix remarkably improved performance and was likely a major contribution to all previous issues.

2. Epsilon greedy policy slows convergence.

Epsilon greedy policy appears to slow convergence.
When comparing a vanilla DQ with no epsilon greedy policy to PDQ with epsilon greedy policy, the vanilla DQ performs best.
Further, studies with both torch and numpy seeds will need to be done to confirm this.
One idea is a jupiter notebook for side by side analysis. 
TODOs were added to both PDQN_Agent and DQ_Agent for epsilon greedy features.

3. Network size is crucial for optimal solution.

Even with this relatively simple problem, network size is crucial for success.
The DQ_Agent was not able to solve the environment with hidden_layer = 64, but was able to with hidden_layer = 256.
Similar to the epsilon studies proposed above, width vs depth will need to be explored.

4. Reference solutions are very, VERY helpful. 

Lastly, I would not have found the SARSA bug if it were not for a reference implementation.
When possible, check your implementation against a reference.

