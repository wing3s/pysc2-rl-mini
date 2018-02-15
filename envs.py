import yaml
import numpy as np

from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

with open("sc2_config.yml", 'r') as ymlfile:
    sc2_cfg = yaml.load(ymlfile)


def create_sc2_minigame_env(map_name, visualize=False):
    env = sc2_env.SC2Env(
        map_name=map_name,
        step_mul=sc2_cfg['step_mul'],
        screen_size_px=(sc2_cfg['screen']['y_res'], sc2_cfg['screen']['x_res']),
        minimap_size_px=(sc2_cfg['minimap']['y_res'], sc2_cfg['minimap']['x_res']),
        visualize=visualize)
    env = available_actions_printer.AvailableActionsPrinter(env)
    return env


class GameInterfaceHandler(object):
    """Provide game interface info and transform observed game image into CNN input tensors.
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

    @property
    def action_space(self):
        """Return total number of available actions"""
        return len(actions.FUNCTIONS)

    @property
    def screen_resolution(self):
        """Return (resolution, resolution) for screen"""
        return (sc2_cfg['screen']['y_res'],
                sc2_cfg['screen']['x_res'])

    @property
    def minimap_resolution(self):
        return (sc2_cfg['minimap']['y_res'],
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

    def preprocess_screen(self, screen):
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

    def preprocess_minimap(self, minimap):
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
