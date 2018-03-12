import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.utils import weight_norm
from torch.autograd import Variable


def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv2d]:
        init.xavier_uniform(model.weight)
        init.constant(model.bias, 0)
    elif type(model) in [nn.LSTMCell]:
        init.constant(model.bias_ih, 0)
        init.constant(model.bias_hh, 0)


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

        return value, policy, new_lstm_hidden_vb


class FullyConv(torch.nn.Module):

    def __init__(self,
                 minimap_channels,
                 screen_channels,
                 screen_resolution,
                 num_action,
                 enable_lstm=True):
        super(FullyConv, self).__init__()
        self.enable_lstm = enable_lstm

        # apply paddinga as 'same', padding = (kernel - 1)/2
        self.mconv1 = nn.Conv2d(minimap_channels, 16, 5, stride=1, padding=2)  # shape (N, 16, m, m)
        self.mconv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  # shape (N, 32, m, m)
        self.sconv1 = nn.Conv2d(screen_channels, 16, 5, stride=1, padding=2)  # shape (N, 16, s, s)
        self.sconv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  # shape (N, 32, s, s)

        # spatial actor
        state_channels = 32 * 2 + 1  # stacking minimap, screen, info
        self.sa_conv3 = nn.Conv2d(state_channels, 1, 1, stride=1)  # shape (N, 65, s, s)

        # non spatial feature
        self.ns_fc3 = nn.Linear(
            screen_resolution[0] * screen_resolution[1] * state_channels, 256)
        # non spatial actor
        self.nsa_fc4 = nn.Linear(256, num_action)
        # non spatial critic
        self.nsc_fc4 = nn.Linear(256, 1)

        self.apply(init_weights)
        self.train()

    def forward(self, minimap_vb, screen_vb, info_vb, valid_action_vb, lstm_hidden_vb=None):
        """
            Args:
                minimap_vb, (N, # of channel, width, height)
                screen_vb, (N, # of channel, width, height)
                info_vb, (len(info))
                valid_action_vb, (len(observation['available_actions])) same as (num_actions)
            Returns:
                value_vb, (1, 1)
                spatial_policy_vb, (1, s*s)
                non_spatial_policy_vb, (1, num_actions)
                lstm_hidden variables
            TODO: implement lstm
        """
        x_m = F.relu(self.mconv1(minimap_vb))
        x_m = F.relu(self.mconv2(x_m))
        x_s = F.relu(self.sconv1(screen_vb))
        x_s = F.relu(self.sconv2(x_s))

        x_i = Variable(
            info_vb.data.repeat(
                math.ceil(x_s.shape[2] * x_s.shape[3] / info_vb.shape[0])
            ).resize_(
                x_s.shape[2], x_s.shape[3]
            )
        )  # transform info vector into 2d matrix with shape (s, s) filled with repeat values
        x_i = x_i.unsqueeze(0).unsqueeze(0)  # shape (1, 1, s, s)
        x_state = torch.cat((x_m, x_s, x_i), dim=1)  # concat along channel dimension

        x_spatial = self.sa_conv3(x_state)
        x_spatial = x_spatial.view(x_spatial.shape[0], -1)
        spatial_policy_vb = F.softmax(x_spatial, dim=1)
        spatial_policy_log_vb = F.log_softmax(x_spatial, dim=1)

        x_non_spatial = x_state.view(x_state.shape[0], -1)
        x_non_spatial = F.relu(self.ns_fc3(x_non_spatial))

        # NOTE: use torch.log(torch.softmax(Tensor)) will cause result unstable,
        #       use log_softmax instead
        x_non_spatial_policy = self.nsa_fc4(x_non_spatial)
        non_spatial_policy_vb = F.softmax(x_non_spatial_policy, dim=1)
        non_spatial_policy_vb = self._mask_unavailable_actions(
            non_spatial_policy_vb, valid_action_vb)
        non_spatial_policy_log_vb = F.log_softmax(x_non_spatial_policy, dim=1)
        non_spatial_policy_log_vb = self._mask_unavailable_actions_log(
            non_spatial_policy_log_vb, valid_action_vb
        )

        value_vb = self.nsc_fc4(x_non_spatial)

        return value_vb, spatial_policy_vb, spatial_policy_log_vb, non_spatial_policy_vb, non_spatial_policy_log_vb, None

    def _mask_unavailable_actions(self, policy_vb, valid_action_vb):
        """
            Args:
                policy_vb, (1, num_actions)
                valid_action_vb, (num_actions)
            Returns:
                masked_policy_vb, (1, num_actions)
        """
        masked_policy_vb = policy_vb * valid_action_vb
        masked_policy_vb /= masked_policy_vb.sum(1)
        return masked_policy_vb

    def _mask_unavailable_actions_log(self, policy_vb, valid_action_vb):
        """
            Args:
                policy_vb, (1, num_actions)
                valid_action_vb, (num_actions)
            Returns:
                masked_policy_vb, (1, num_actions)
        """
        masked_policy_vb = policy_vb * valid_action_vb
        masked_policy_vb -= torch.log(
            (masked_policy_vb.exp() * valid_action_vb).sum(1)) * valid_action_vb
        return masked_policy_vb
