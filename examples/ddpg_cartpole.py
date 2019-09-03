import gym
import torch
import torch.nn as nn
from torch.optim import Adam

from ignis import DDPGAgent


env = gym.make('CartPole-v1')
action_space = env.action_space
space_size = env.observation_space.shape[0]


class ActorModel(nn.Module):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.fc1 = nn.Linear(space_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, action_space)
        self.sigmoid = nn.Sigmoid()

    def forward(self, states):
        x = self.relu(self.fc1(states))
        x = self.sigmoid(self.fc2(x))
        return x


class CriticModel(nn.Module):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(space_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, states, actions):
        x = self.relu(self.fc1(states))
        x = torch.cat((x, actions), dim=1)
        x = self.sigmoid(self.fc2(x))
        return x


actor = ActorModel()
critic = CriticModel()
actor_optimizer = Adam(actor.parameters(), lr=1e-3)
critic_optimizer = Adam(critic.parameters(), lr=3e-3)


ddpg = DDPGAgent(
    device='cpu',
    actor=actor,
    actor_optimizer=actor_optimizer,
    critic=critic,
    critic_optimizer=critic_optimizer,
    memory_size=int(1e5),
    batch_size=256,
    update_every=16,
    discount=0.999,
    soft_update_tau=1e-3,
)

ddpg.run(environment=env, epochs=500)