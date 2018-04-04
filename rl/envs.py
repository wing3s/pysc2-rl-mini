import yaml
import numpy as np
from os import path
from absl import flags

from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions


sc2_f_path = path.abspath(path.join(path.dirname(__file__), "..", "configs", "sc2_config.yml"))
with open(sc2_f_path, 'r') as ymlfile:
    sc2_cfg = yaml.load(ymlfile)


# TODO: update README.md for adding random seed for game env
def create_sc2_minigame_env(map_name, mode, visualize=False):
    """Create sc2 game env with available actions printer
        Set screen, minimap same resolution and x, y same pixels for simplicity.
    """
    assert mode in ['full', 'lite', 'test']

    # workaround for pysc2 flags
    FLAGS = flags.FLAGS
    FLAGS([__file__])

    env_seed = 3 if mode == 'test' else None
    env = sc2_env.SC2Env(
        map_name=map_name,
        step_mul=sc2_cfg[mode]['step_mul'],
        screen_size_px=(sc2_cfg[mode]['resl'],) * 2,
        minimap_size_px=(sc2_cfg[mode]['resl'],) * 2,
        visualize=visualize,
        random_seed=env_seed)
    return env


# TODO: move preprocess to neuro net embed layer
# TODO: move post process into sc2_env extension
class GameInterfaceHandler(object):
    """Provide game interface info.
        Transform observed game image and available actions into CNN input tensors.

        - Special Categorial 2d image:
            single layer normalized by scalar max
            (no same category overlapping)
        - Categorial 2d image:
            expand to multiple layer
        - Scalar 2d image:
            single layer normalized by scalar max

        NOTE: This class can potentially be a decorator to wrap sc2_env
    """

    def __init__(self, mode):
        assert mode in ['full', 'lite', 'test']
        self.dtype = np.float32

        self.minimap_player_id = features.MINIMAP_FEATURES.player_id.index
        self.screen_player_id = features.SCREEN_FEATURES.player_id.index
        self.screen_unit_type = features.SCREEN_FEATURES.unit_type.index

        self.screen_resolution = sc2_cfg[mode]['resl']
        self.minimap_resolution = sc2_cfg[mode]['resl']

        (self.sub_to_full_acts, self.full_to_sub_acts) = self._get_action_mappings(
            sc2_cfg[mode]['action_list'])
        self.num_action = len(self.sub_to_full_acts)
        self.non_spatial_actions = self._get_non_spatial_actions()

    @property
    def screen_channels(self):
        """Return number of channels for preprocessed screen image"""
        channels = 0
        for i, screen_feature in enumerate(features.SCREEN_FEATURES):
            if i == self.screen_player_id or i == self.screen_unit_type:
                channels += 1
            elif screen_feature.type == features.FeatureType.SCALAR:
                channels += 1
            else:
                channels += screen_feature.scale
        return channels

    def _preprocess_screen(self, screen):
        """Transform screen image into expanded tensor
            Args:
                screen: obs.observation['screen']
            Returns:
                ndarray, shape (len(SCREEN_FEATURES), screen_size_px.y, screen_size_px.x)
        """
        screen = np.array(screen, dtype=self.dtype)
        layers = []
        assert screen.shape[0] == len(features.SCREEN_FEATURES)
        for i, screen_feature in enumerate(features.SCREEN_FEATURES):
            if i == self.screen_player_id or i == self.screen_unit_type:
                layers.append(np.log(screen[i:i + 1] + 1.))
            elif screen_feature.type == features.FeatureType.SCALAR:
                layers.append(np.log(screen[i:i + 1] + 1.))
            else:
                layer = np.zeros(
                    (screen_feature.scale, screen.shape[1], screen.shape[2]),
                    dtype=self.dtype)
                for j in range(screen_feature.scale):
                    indy, indx = (screen[i] == j).nonzero()
                    layer[j, indy, indx] = 1
                layers.append(layer)
        return np.concatenate(layers, axis=0)

    def get_screen(self, observation):
        """Extract screen variable from observation['minimap']
            Args:
                observation: Timestep.obervation
            Returns:
                screen: ndarray, shape (1, len(SCREEN_FEATURES), screen_size_px.y, screen_size_px.x)
        """
        screen = self._preprocess_screen(observation['screen'])
        return np.expand_dims(screen, 0)

    @property
    def minimap_channels(self):
        """Return number of channels for preprocessed minimap image"""
        channels = 0
        for i, minimap_feature in enumerate(features.MINIMAP_FEATURES):
            if i == self.minimap_player_id:
                channels += 1
            elif minimap_feature.type == features.FeatureType.SCALAR:
                channels += 1
            else:
                channels += minimap_feature.scale
        return channels

    def _preprocess_minimap(self, minimap):
        """Transform minimap image into expanded tensor
            Args:
                minimap: obs.observation['minimap']
            Returns:
                ndarray, shape (len(MINIMAP_FEATURES), minimap_size_px.y, minimap_size_px.x)
        """
        minimap = np.array(minimap, dtype=self.dtype)
        layers = []
        assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
        for i, minimap_feature in enumerate(features.MINIMAP_FEATURES):
            if i == self.minimap_player_id:
                layers.append(np.log(minimap[i:i + 1] + 1.))
            elif minimap_feature.type == features.FeatureType.SCALAR:
                layers.append(np.log(minimap[i:i + 1] + 1.))
            else:
                layer = np.zeros(
                    (minimap_feature.scale, minimap.shape[1], minimap.shape[2]),
                    dtype=self.dtype)
                for j in range(minimap_feature.scale):
                    indy, indx = (minimap[i] == j).nonzero()
                    layer[j, indy, indx] = 1
                layers.append(layer)
        return np.concatenate(layers, axis=0)

    def get_minimap(self, observation):
        """Extract minimap variable from observation['minimap']
            Args:
                observation: Timestep.observation
            Returns:
                minimap: ndarray, shape (1, len(MINIMAP_FEATURES), minimap_size_px.y, minimap_size_px.x)
        """
        minimap = self._preprocess_minimap(observation['minimap'])
        return np.expand_dims(minimap, 0)

    def _preprocess_available_actions(self, available_actions):
        """Returns ndarray of available_actions from observed['available_actions']
            shape (num_actions)
        """
        available_actions = np.intersect1d(available_actions, self.sub_to_full_acts)
        a_actions = np.zeros((self.num_action), dtype=self.dtype)
        a_actions[self.full_to_sub_acts[available_actions]] = 1.
        return a_actions

    def get_available_actions(self, observation):
        """
            Args:
                observation: Timestep.observation
            Returns:
                available_action: ndarray, shape(num_actions)
        """
        return self._preprocess_available_actions(
            observation['available_actions'])

    def get_info(self, observation):
        """Extract available actioins as info from state.observation['available_actioins']
            Args:
                observation: Timestep.observation
            Returns:
                info: ndarray, shape (num_actions)
        """
        return self.get_available_actions(observation)

    def postprocess_action(self, non_spatial_action, spatial_action):
        """Transform selected non_spatial and spatial actions into pysc2 FunctionCall
            Args:
                non_spatial_action: ndarray, shape (1, 1)
                spatial_action: ndarray, shape (1, 1)
            Returns:
                FunctionCall as action for pysc2_env
        """
        act_id = self.sub_to_full_acts[non_spatial_action[0][0]]
        target = spatial_action[0][0]
        target_point = [
            int(target % self.screen_resolution),
            int(target // self.screen_resolution)
        ]  # (x, y)

        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append(target_point)
            else:
                act_args.append([0])
        return actions.FunctionCall(act_id, act_args)

    def _get_non_spatial_actions(self):
        non_spatial_actions = [True] * self.num_action
        for func_id, func in enumerate(actions.FUNCTIONS):
            for arg in func.args:
                if arg.name in ('screen', 'minimap', 'screen2'):
                    non_spatial_actions[self.full_to_sub_acts[func_id]] = False
                    break
        return non_spatial_actions

    def is_non_spatial_action(self, action_id):
        return self.non_spatial_actions[self.full_to_sub_acts[action_id]]

    def _get_action_mappings(self, action_list):
        """Fill actioin list if it's empty

            Args:
                action_list: list
            Returns:
                sub_to_full_acts: ndarray
                full_to_sub_acts: ndarray
        """
        if len(action_list) == 0:
            action_list = [i for i in range(len(actions.FUNCTIONS))]

        sub_to_full_acts = action_list
        full_to_sub_acts = [-1] * len(actions.FUNCTIONS)
        for idx, val in enumerate(sub_to_full_acts):
            full_to_sub_acts[val] = idx
        return (np.asarray(sub_to_full_acts, dtype=np.int32), np.asarray(full_to_sub_acts, dtype=np.int32))
