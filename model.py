import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm


class ActorCritic(torch.nn.Module):

    def __init__(self,
                 minimap_channels,
                 screen_channels,
                 info_size,
                 screen_resolution,
                 action_space,
                 enable_lstm=True):
        super(ActorCritic, self).__init__()
        self.enable_lstm = enable_lstm

        self.conv1 = nn.Conv2d(minimap_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        if self.enable_lstm:
            self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

        self.critic_5 = nn.Linear(256, 1)
        self.actor_5 = nn.Linear(256, action_space.n)
        self.actor_6 = nn.Softmax()

        # apply Xavier weights initialization
        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        torch.nn.init.xavier_uniform(self.actor_5.weight)
        torch.nn.init.xavier_uniform(self.critic_5.weight)

        # apply normalized weight
        self.actor_5 = weight_norm(self.actor_5)
        self.actor_5.bias.data.fill_(0)
        self.critic_5 = weight_norm(self.critic_5)
        self.critic_5.bias.data.fill_(0)

        self.train()

    def forward(self, inputs, lstm_hidden_vb=None):
        """Return value, policy, lstm_hidden variables"""
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)

        new_lstm_hidden_vb = None
        if self.enable_lstm:
            hx, cx = self.lstm(x, lstm_hidden_vb)
            new_lstm_hidden_vb = (hx, cx)
            x = hx

        value = self.critic_5(x)
        policy = self.actor_5(x)
        policy = self.actor_6(x)

        return value, policy, new_lstm_hidden_vb
