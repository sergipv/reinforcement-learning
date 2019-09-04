# Implements an RL Agent to interact with an environment.
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from ReplayBuffer import ReplayBuffer


Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward', 'done'))

class Agent():
    """
    Implementation of an Agent that interacts with an environment.
    This class implements both the overall system used (Deep Q-Learning) and modelling for the DNN used.
    """

    def __init__(
            self, 
            model, 
            replay_buffer: ReplayBuffer,
            num_actions: int,
            batch_size: int,
            samples_to_train: int,
            device: str,
            gamma=0.95):
        """
        <TO BE UPDATED>
        Initialization of the Agent.
        :param num_actions: number of possible actions the Agent can take.
        :param buffer_size: number of samples used on the Replay Buffer.
        :param seed: seed used for weight initialization.
        :param device: "cpu" or "cuda:0", for torch computation with GPU.
        """
        self._model = model
        self._buffer = replay_buffer
        self._num_actions = num_actions
        self._device = device
        self._batch_size = batch_size
        self._samples_to_train = samples_to_train
        self._gamma = gamma

        # Internal update count used to only train evey samples_to_train.
        self._temp_steps = 0
        self._optimizer = optim.Adam(model.parameters(), lr=5e-4)
        #self._optimizer = optim.RMSprop(model.parameters())

    def next_action(self, state, epsilon: float):
        self._model.eval()

        # with torch.no_grad():
        #     actions = self._model(torch.from_numpy(state).float().unsqueeze(0).to(self._device))

        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        # Epsilon-greedy:
        if random.random() > epsilon:
            with torch.no_grad():
                return self._model(state).max(1)[1].view(1, 1)
                # return np.argmax(self._model(state).cpu().data.numpy())
        # return random.choice(np.arange(self._num_actions))
        return torch.tensor([[random.randrange(self._num_actions)]],
                device=self._device, dtype=torch.long)

    def train(self, state, action, reward, next_state, done):
        self._buffer.push(state, action, reward, next_state, done)
        self._temp_steps = (self._temp_steps + 1) % self._samples_to_train

        if (self._temp_steps != 0) or (len(self._buffer) < self._batch_size):
            return

        experiences = self._buffer.sample(self._batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = experiences

        # Transform the batches into vectors with state, action, reward,
        # next_state and done values for the batch.
        # batch = Transition(*zip(*transitions))

        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward)
        # next_states_batch = torch.cat(batch.next_state)
        # done_batch = torch.cat(batch.done)

        max_action = self._model(next_state_batch).detach().max(1)[0].unsqueeze(1)

        # We multiply by (1-dones) because when reaching the done there is no
        # future action.
        q_curr = reward_batch + (self._gamma * max_action * (1 - done_batch))
        q_next = self._model(state_batch).gather(1,action_batch)
        loss = F.mse_loss(q_curr, q_next)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def save_nn(self, filename: str):
        """
        Saves the weights of the NN into filename.
        """
        torch.save(self._model.state_dict(), filename)

