import unittest
from gym_plannable.env.grid_world import MazeEnv, MazeEnvMA
from plannable_mixins import PlannableInterfaceTestMixin, MultiAgentPlannableEnvTestMixin
from multi_agent_mixins import EnvTestMixin

class PlannableInterfaceTestMazeEnv(PlannableInterfaceTestMixin, unittest.TestCase):
    env_constructor = MazeEnv
    max_next = 5

class PlannableInterfaceTestMazeEnvMA(PlannableInterfaceTestMixin, unittest.TestCase):
    env_constructor = MazeEnvMA
    max_next = 5

class PlannableEnvTestMazeEnvMA(MultiAgentPlannableEnvTestMixin, unittest.TestCase):
    env_constructor = MazeEnvMA

class EnvTestMazeEnvMA(EnvTestMixin, unittest.TestCase):
    env_constructor = MazeEnvMA
