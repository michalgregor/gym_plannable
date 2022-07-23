from .base_grid_world import BaseGridWorldEnvMA, make_sa_grid_world_env

from .grid_world import (
    GoalDrape, PositionActor, BackgroundObject,
    DrapeObject, PosObservation, ConstRewardObject
)
from .logging import PathLogger

class MazeEnvMA(BaseGridWorldEnvMA):
    def __init__(self, grid=None, observation_function=None,
                 show_path=False, move_punishment=0,
                 cliff_punishment=-100, goal_reward=100,
                 action_spec=None, show_path_kw=dict(), **kwargs):
        if grid is None:
            grid = """
            XXXX00000G
            000X000000
            000X000XXX
            000X000000
            000X000000
            0000000000
            0000X00000
            0000XXX00X
            0000X00000
            S000X00000
            """
        
        super().__init__(
            grid=grid, observation_function=observation_function,
            show_path=show_path, move_punishment=move_punishment,
            cliff_punishment=cliff_punishment, goal_reward=goal_reward,
            action_spec=action_spec, show_path_kw=show_path_kw, **kwargs
        )

    def setup_env(
        self, grid_ar, observation_function=None,
        show_path=False, move_punishment=0,
        cliff_punishment=-100, goal_reward=100,
        action_spec=None, show_path_kw=dict(), **kwargs
    ):
        cliff_punishment -= move_punishment

        goal = GoalDrape("goal", grid_ar, 'G', facecolor=None,
                         label='G', player_names=["player"],
                         goal_reward=goal_reward)
        cliff = GoalDrape("cliff", grid_ar, 'C', facecolor='lightgray',
                         label='C', player_names=["player"],
                         goal_reward=cliff_punishment)
        punishment = ConstRewardObject("punishment", "player", reward=move_punishment)
        player = PositionActor("player", grid_ar, 'S', 'walls', 
                               action_spec=action_spec,
                               facecolor='blue', edgecolor=None)
        self.action_spec = player.action_spec

        transition_sequence = [
            player,
            punishment,
            cliff,
            goal
        ]
        
        render_sequence = [
            BackgroundObject("background", grid_ar),
            DrapeObject("walls", grid_ar, 'X', facecolor='gray'),
            DrapeObject("starts", grid_ar, 'S', facecolor=None, label='S'),
            goal,
            cliff,
            player
        ]

        if show_path:
            path_logger = PathLogger("player_logger", "player", **show_path_kw)
            transition_sequence.append(path_logger)
            render_sequence.append(path_logger)

        if observation_function is None:
            observation_function = [PosObservation(grid_ar.shape,
                                                   pos_agent_name="player")]

        return transition_sequence, render_sequence, observation_function

MazeEnv = make_sa_grid_world_env(MazeEnvMA)
