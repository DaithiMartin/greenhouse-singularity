# Autonomous control of stratospheric balloons

---

Paper:
https://www.nature.com/articles/s41586-020-2939-8

## interesting components

---
1. reward function
   1. "cliffing" reward just outside the ideal range
   2. scaling the power component rather than adding it to the range component
2. state vector
   1. blending measurements with forecasts using a Gaussian process
   2. centering the air column encoding on the balloon
      1. creates inductive bias for policy
   3. using Gaussian process to augment measurements on the balloon with medium range forecasts
   4. incorporating variance of the posterior for uncertainty at various elevations
3. Generating the training simulation
   1. taking real-world data and applying procedural noise
      1. analogous to data augmentation
4. Training pipeline
   1. 100 parallel agents generating experience tuples
   2. 4 replay buffers that they all share


## insights
1. QR-DQN appears to work better than other versions DQN due to increased learning stability
2. if your agent has little effect on the environment then "data augmentation" of real data can be used to create a simulated environment
3. Learning appears to be stabilized if your observation-state and rewards are close to 1
4. reward landscape has a large effect on behavior