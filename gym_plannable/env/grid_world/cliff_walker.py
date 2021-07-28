from .grid_world import (
    GridWorldEnv, parse_grid_str, GoalDrape, PositionActor, BackgroundObject,
    DrapeObject, PosObservation, ConstRewardObject
)
from ...multi_agent import Multi2SingleWrapper
from .logging import PathLogger

class CliffWalkerEnvMA(GridWorldEnv):
    def __init__(self, grid=None, observation_function=None,
                 visualize_path=False, **kwargs):
        if grid is None:
            grid = """
            000000000000
            000000000000
            000000000000
            SCCCCCCCCCCG
            """
        
        grid_ar = parse_grid_str(grid)
        move_punishment = -1
        cliff_punishment = -100 - move_punishment

        goal = GoalDrape("goal", grid_ar, 'G', facecolor=None,
                         label='G', player_names=["player"],
                         goal_reward=0)
        cliff = GoalDrape("cliff", grid_ar, 'C', facecolor=None,
                         label='C', player_names=["player"],
                         goal_reward=cliff_punishment)
        punishment = ConstRewardObject("punishment", "player", reward=move_punishment)
        player = PositionActor("player", grid_ar, 'S', 'walls',
                               facecolor='blue', edgecolor=None)
        
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

        if visualize_path:
            path_logger = PathLogger("player_logger", "player")
            transition_sequence.append(path_logger)
            render_sequence.append(path_logger)

        if observation_function is None:
            observation_function = [PosObservation(grid_ar.shape,
                                                   pos_agent_name="player")]
        
        super().__init__(
            grid_ar.shape,
            transition_sequence=transition_sequence,
            render_sequence=render_sequence,
            observation_function=observation_function,
            **kwargs
        )

class CliffWalkerEnv(Multi2SingleWrapper):
    def __init__(self, grid=None, observation_function=None,
                 visualize_path=False, **kwargs):
        env = CliffWalkerEnvMA(grid=grid,
            observation_function=observation_function, **kwargs)
        super().__init__(env)