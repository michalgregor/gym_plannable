from .grid_world import (
    GridWorldEnv, parse_grid_str, GoalDrape, PositionActor, BackgroundObject,
    DrapeObject, PosObservation
)
from ...multi_agent import Multi2SingleWrapper
from .logging import PathLogger

class MazeEnvMA(GridWorldEnv):
    def __init__(self, grid=None, observation_function=None,
                 visualize_path=False, **kwargs):
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
        
        grid_ar = parse_grid_str(grid)
                
        goal = GoalDrape("goal", grid_ar, 'G', facecolor=None,
                         label='G', player_names=["player"],
                         goal_reward=100)
        player = PositionActor("player", grid_ar, 'S', 'walls',
                               facecolor='blue', edgecolor=None)
                
        transition_sequence = [
            player,
            goal
        ]
        
        render_sequence = [
            BackgroundObject("background", grid_ar),
            DrapeObject("walls", grid_ar, 'X', facecolor='gray'),
            DrapeObject("starts", grid_ar, 'S', facecolor=None, label='S'),
            goal,
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

class MazeEnv(Multi2SingleWrapper):
    def __init__(self, grid=None, observation_function=None,
                 visualize_path=False, **kwargs):
        env = MazeEnvMA(grid=grid, observation_function=observation_function,
                        visualize_path=visualize_path, **kwargs)
        super().__init__(env)