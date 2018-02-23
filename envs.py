import yaml
import numpy as np

from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

with open("sc2_config.yml", 'r') as ymlfile:
    sc2_cfg = yaml.load(ymlfile)


def create_sc2_minigame_env(map_name, visualize=False):
    """Create sc2 game env with available actions printer"""
    env = sc2_env.SC2Env(
        map_name=map_name,
        step_mul=sc2_cfg['step_mul'],
        screen_size_px=(sc2_cfg['screen']['y_res'], sc2_cfg['screen']['x_res']),
        minimap_size_px=(sc2_cfg['minimap']['y_res'], sc2_cfg['minimap']['x_res']),
        visualize=visualize)
    env = available_actions_printer.AvailableActionsPrinter(env)
    return env


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

    def __init__(self):
        self.minimap_player_id = features.MINIMAP_FEATURES.player_id.index
        self.screen_player_id = features.SCREEN_FEATURES.player_id.index
        self.screen_unit_type = features.SCREEN_FEATURES.unit_type.index

        self.num_action = len(actions.FUNCTIONS)
        self.screen_resolution = (sc2_cfg['screen']['y_res'],
                                  sc2_cfg['screen']['x_res'])
        self.minimap_resolution = (sc2_cfg['minimap']['y_res'],
                                   sc2_cfg['minimap']['x_res'])

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
        screen = np.array(screen, dtype=np.float32)
        layers = []
        assert screen.shape[0] == len(features.SCREEN_FEATURES)
        for i, screen_feature in enumerate(features.SCREEN_FEATURES):
            if i == self.screen_player_id or i == self.screen_unit_type:
                layers.append(screen[i:i+1] / screen_feature.scale)
            elif screen_feature.type == features.FeatureType.SCALAR:
                layers.append(screen[i:i+1] / screen_feature.scale)
            else:
                layer = np.zeros(
                    (screen_feature.scale, screen.shape[1], screen.shape[2]),
                    dtype=np.float32)
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
        layers = []
        assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
        for i, minimap_feature in enumerate(features.MINIMAP_FEATURES):
            if i == self.minimap_player_id:
                layers.append(minimap[i:i+1] / minimap_feature.scale)
            elif minimap_feature.type == features.FeatureType.SCALAR:
                layers.append(minimap[i:i+1] / minimap_feature.scale)
            else:
                layer = np.zeros(
                    (minimap_feature.scale, minimap.shape[1], minimap.shape[2]),
                    dtype=np.float32)
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
        a_actions = np.zeros((self.num_action), dtype=np.float32)
        a_actions[available_actions] = 1
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
                info: ndarray, shape (1, num_actions)
        """
        a_actions = self._preprocess_available_actions(observation['available_actions'])
        return np.expand_dims(a_actions, 0)

    def postprocess_action(self, non_spatial_action_ts, spatial_action_ts):
        """Transform selected non_spatial and spatial actions into pysc2 FunctionCall
            Args:
                non_spatial_action_ts: pytorch tensor
                spatial_action_ts: pytorch tensor
            Returns:
                FunctionCall as action for pysc2_env
        """
        act_id = non_spatial_action_ts.numpy()[0]
        target = spatial_action_ts.numpy()[0]
        target_point = [
            int(target // self.screen_resolution[0]),
            int(target % self.screen_resolution[0])
        ]  # (y, x)

        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                # Point: [x, y]
                act_args.append([target_point[1], target_point[0]])
            else:
                act_args.append([0])
        return actions.FunctionCall(act_id, act_args)
