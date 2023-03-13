from .grid_world import GridWorldEnv, parse_grid_str
from ...multi_agent import Multi2SingleWrapper

class BaseGridWorldEnvMA(GridWorldEnv):
    def __init__(self, grid=None, render_mode=None, observation_function=None, **kwargs):       
        grid_ar = parse_grid_str(grid)

        (transition_sequence,
         render_sequence,
         observation_function) = self.setup_env(grid_ar, **kwargs)
        
        super().__init__(
            grid_ar.shape,
            transition_sequence=transition_sequence,
            render_sequence=render_sequence,
            render_mode=render_mode,
            observation_function=observation_function
        )

    def setup_env(self, grid_ar, **kwargs):
        transition_sequence = []
        render_sequence = []
        observation_function = None
        return transition_sequence, render_sequence, observation_function

def make_sa_grid_world_env(env_class):
    class BaseGridWorld(Multi2SingleWrapper):
        metadata = env_class.metadata

        def __init__(self, grid=None, observation_function=None, **kwargs):
            env = env_class(grid=grid, observation_function=observation_function, **kwargs)
            super().__init__(env)

    BaseGridWorld.__doc__ = env_class.__doc__
    return BaseGridWorld
