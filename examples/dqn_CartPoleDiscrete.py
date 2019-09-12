import gym
import torch.nn as nn
from torch.optim import Adam

from ignis import DQNAgent


env = gym.make('CartPole-v1')
action_space = env.action_space.n
space_size = env.observation_space.shape[0]


class ActorModel(nn.Module):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.fc1 = nn.Linear(space_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, action_space)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, states):
        x = self.relu1(self.fc1(states))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


actor = ActorModel()
actor_optimizer = Adam(actor.parameters(), lr=5e-4)

dqn = DQNAgent(
    device='cpu',
    actor=actor,
    actor_optimizer=actor_optimizer,
    memory_size=int(1e5),
    batch_size=64,
    update_every=4,
    discount=0.99,
    soft_update_tau=1e-3,
    eps_start=1,
    eps_end=0.01,
    eps_decay=0.999,
)

dqn.run(env=env, epochs=2000, score_threshold=190, filename='best_model')
