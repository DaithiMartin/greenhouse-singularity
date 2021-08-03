import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from agents.PDQN_Model import Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
# --------------------------------------------------------------------------------------------- #
LEARNING_RATE = 1e-4  # NN learning rate
BUFFER_SIZE = int(1e2)  # replay buffer size, learning does not start until this is full
BATCH_SIZE = 50  # memory batch size
UPDATE_EVERY = 4  # update NN every n steps
GAMMA = 0.99  # reward discounting factor
BETA = 0.0  # prioritized experience replay
TAU = 1e-3  # controls the soft update

# epsilon greedy exploration vs exploitation parameters
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995


# --------------------------------------------------------------------------------------------- #


class Agent:
    """Q Network agent with prioritized replay buffer"""

    def __init__(self, action_size, observation_size, seed):

        """
        Initializes Agent Object
        """
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = Actor(action_size, observation_size, seed).to(device)
        self.qnetwork_target = Actor(action_size, observation_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # epsilon decay
        self.eps = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY

    def step(self, state, action, reward, next_state, done):
        """
        Saves experience in the replay buffer and decides if the network needs to be updated.
        Args:
            state: (tuple) state vector
            action: (tuple) action vector
            reward: (float) reward scalar
            next_state: (tuple) next state vector
            done: (bool) defines if state is terminal
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.1):

        """
        Chooses an action with epsilon greedy policy.
        Args:
            state: (tuple) state vector
            eps: (float) epsilon value
        Returns:
            int: action index
        """

        state = torch.Tensor(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        action = np.argmax(action_values)

        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return int(action)
        else:
            return np.random.randint(0, 11)

    def learn(self, experiences, gamma):
        # type: (tuple, float) -> None
        """
        Up date the model weights and the resulting priorities in the replay buffer.
        Args:
            experiences: (tuple) tuple of np.vstacks, data for supervised learning
            gamma: (float) expected future reward discount factor
        """
        states, actions, rewards, next_states, dones, probabilities = experiences

        sampling_weight = (1 / BUFFER_SIZE * 1 / probabilities) ** BETA / probabilities.squeeze().max(0)[0]

        # get action value for the chosen action
        Q_target_undiscounted = self.qnetwork_target(next_states).detach()

        # choose the best action and shape correctly
        Q_target_undiscounted = Q_target_undiscounted.max(1)[0].unsqueeze(1)

        # discount the target
        Q_target = rewards + (gamma * Q_target_undiscounted * (1 - dones))

        # get q estimate and shape correctly
        Q_estimate = self.qnetwork_local(states).max(1)[0].unsqueeze(1)

        # calculate loss
        loss = F.mse_loss(sampling_weight * Q_estimate, sampling_weight * Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # update the probabilities with new td_errors
        td_error = Q_target - Q_estimate
        self.memory.update(states, actions, rewards, next_states, dones, td_error)

    def soft_update(self, local_model, target_model, tau):
        # type: (torch.nn.Module, torch.nn.Module, float) -> None
        """
        Weighted update of local model parameters.
        θ_target = tau*θ_local + (1 - tau)*θ_target
        Args:
            local_model: (torch.nn.Module) weights will be copied from
            target_model: (torch.nn.Module) weights will be copied to
            tau: (float) interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size prioritized buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, eta=1e-5, prob_temp=0.5, temp_decay=1e-5):
        # type: (int, int, int, int, float, float, float) -> None
        """
        Initialize a ReplayBuffer object.
        Args:
            action_size: (int) dimension of each action
            buffer_size: (int) maximum size of buffer
            batch_size: (int) size of each training batch
            seed: (int) random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = np.random.seed(seed)
        self.eta = eta
        self.prob_temp = prob_temp
        self.temp_decay = temp_decay
        self.indexes = np.zeros(batch_size)

        self.memory = deque(maxlen=buffer_size)
        self.priority_memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "priority"])

    def add(self, state, action, reward, next_state, done):
        # type: (tuple, int, float, tuple, bool) -> None
        """
        Compute priority and add new experience to replay buffer.
        Args:
            state: (tuple) state vector
            action: (int) action index
            reward: (float) reward scalar
            next_state: (tuple) next state vector
            dones: (bool) indicates if state is terminal
            td_error: (float) used to update priority in buffer
        """

        priority = max(self.priority_memory) if len(self.memory) > 0 else 0.1
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)
        self.priority_memory.append(priority)

    def sample(self):
        """
        Select experiences with priority probability.
        Returns:
            states: (torch.tensor) sample state tensor
            actions: (torch.tensor) sample actions tensor
            rewards: (torch.tensor) sample rewards tensor
            next_states: (torch.tensor) sample next state tensor
            dones: (torch.tensor) sample dones tensor
            priority_prob: (torch.tensor) sample priority tensor
        """

        self.prob_temp = max(0, (self.prob_temp - self.temp_decay))
        priority_array = np.array(self.priority_memory)

        priority_probability = (priority_array ** self.prob_temp) / np.sum(priority_array ** self.prob_temp)
        self.indexes = np.random.choice(np.arange(0, len(self.memory)), size=self.batch_size, replace=False,
                                        p=priority_probability)

        experiences = [self.memory[i] for i in self.indexes]
        filtered_probabilities = priority_probability[self.indexes]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        priority_prob = torch.from_numpy(np.vstack([p for p in filtered_probabilities if p is not None])).float().to(
            device)

        return states, actions, rewards, next_states, dones, priority_prob

    def update(self, state, action, reward, next_state, dones, td_error):
        # type: (torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, float) -> None
        """Use TD errors to change values for respective tuples in replay buffer
        Args:
            state: (torch.tensor) state vector
            action: (torch.tensor) action index
            reward: (torch.tensor) reward scalar
            next_state: (torch.tensor) next state vector
            dones: (torch.tensor) indicates if state is terminal
            td_error: (float) used to update priority in buffer
        """
        priority = (torch.abs(reward + td_error) + self.eta)

        for i, index in enumerate(self.indexes):
            self.memory[index] = self.experience(state[i].cpu().numpy(),
                                                 action[i].cpu().numpy(),
                                                 reward[i].cpu().numpy(),
                                                 next_state[i].cpu().numpy(),
                                                 dones[i].cpu().numpy(),
                                                 priority[i].cpu().item(),
                                                 )
            self.priority_memory[index] = priority[i].item()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
