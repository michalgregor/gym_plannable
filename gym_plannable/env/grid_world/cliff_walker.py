from .maze import MazeEnvMA
from ...multi_agent import Multi2SingleWrapper

class CliffWalkerEnvMA(MazeEnvMA):
    def __init__(self, grid=None, observation_function=None,
                 show_path=False, move_punishment=-1,
                 cliff_punishment=-100, goal_reward=0,
                 action_spec=None, show_path_kw=dict(), **kwargs):
        if grid is None:
            grid = """
            000000000000
            000000000000
            000000000000
            SCCCCCCCCCCG
            """
        
        super().__init__(
            grid=grid, observation_function=observation_function,
            show_path=show_path, move_punishment=move_punishment,
            cliff_punishment=cliff_punishment, goal_reward=goal_reward,
            action_spec=action_spec, show_path_kw=show_path_kw, **kwargs
        )

class CliffWalkerEnv(Multi2SingleWrapper):
    def __init__(self, grid=None, observation_function=None,
                 show_path=False, move_punishment=-1,
                 cliff_punishment=-100, goal_reward=0,
                 action_spec=None, show_path_kw=dict(), **kwargs):
        env = CliffWalkerEnvMA(grid=grid,
            observation_function=observation_function,
            show_path=show_path, move_punishment=move_punishment,
            cliff_punishment=cliff_punishment, goal_reward=goal_reward,
            action_spec=action_spec, show_path_kw=show_path_kw, **kwargs)
        super().__init__(env)
