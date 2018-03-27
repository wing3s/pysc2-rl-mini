import unittest
import numpy as np
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from rl import model


class ModelTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed = 123
        self.dtype = np.float32
        self.num_action = 7
        self.fully_conv = model.FullyConv(
            minimap_channels=2,
            screen_channels=2,
            screen_resolution=4,
            num_action=self.num_action)

    def testMaskUnavailableActions(self):
        logit = Variable(torch.rand(1, self.num_action))
        policy_vb = F.softmax(logit, dim=1)

        available_action = np.zeros(self.num_action, dtype=self.dtype)
        available_action[[0, 1, 6]] = 1
        available_action_vb = Variable(torch.from_numpy(available_action))

        masked_policy_vb = self.fully_conv._mask_unavailable_actions(
            policy_vb, available_action_vb)

        self.assertAlmostEqual(policy_vb.sum().data[0], 1.)
        self.assertAlmostEqual(available_action_vb.sum().data[0], 3.)
        self.assertAlmostEqual(masked_policy_vb.sum().data[0], 1.)
