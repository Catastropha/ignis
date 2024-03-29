import gym
import torch
import torch.nn as nn
from torch.optim import Adam

from ignis import DDPGAgent


env = gym.make('MountainCarContinuous-v0')
action_space = env.action_space.shape[0]
space_size = env.observation_space.shape[0]


class ActorModel(nn.Module):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.fc1 = nn.Linear(space_size, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, action_space)
        self.tanh = nn.Tanh()

    def forward(self, states):
        x = self.relu(self.fc1(states))
        x = self.tanh(self.fc2(x))
        return x


class CriticModel(nn.Module):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(space_size, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4+action_space, 1)
        self.tanh = nn.Tanh()

    def forward(self, state, action):
        x = self.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = self.tanh(self.fc2(x))
        return x


actor = ActorModel()
critic = CriticModel()
actor_optimizer = Adam(actor.parameters(), lr=1e-4)
critic_optimizer = Adam(critic.parameters(), lr=3e-3)

ddpg = DDPGAgent(
    device='cpu',
    actor=actor,
    actor_optimizer=actor_optimizer,
    critic=critic,
    critic_optimizer=critic_optimizer,
    memory_size=int(1e5),
    batch_size=128,
    update_every=4,
    discount=0.999,
    soft_update_tau=1e-3,
)

ddpg.run(env=env, epochs=2000, score_threshold=90, filename='best_model')
