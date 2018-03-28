import unittest
import yaml
import numpy as np
from os import path

from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions
from rl import envs

sc2_f_path = path.abspath(path.join(path.dirname(__file__), "..", "configs", "sc2_config.yml"))
with open(sc2_f_path, 'r') as ymlfile:
    sc2_cfg = yaml.load(ymlfile)


class EnvTest(unittest.TestCase):
    def setUp(self):
        self.map_name = "CollectMineralShards"
        self.mode = 'test'
        self.screen_resl = sc2_cfg[self.mode]['resl']
        self.env = envs.create_sc2_minigame_env(self.map_name, False, self.mode)

    def testEnvReset(self):
        states = self.env.reset()
        state = states[0]

        self.assertEqual(len(states), 1)
        self.assertEqual(
            state.observation['minimap'].shape,
            (len(features.MINIMAP_FEATURES), self.screen_resl, self.screen_resl))

    def tearDown(self):
        self.env.close()
