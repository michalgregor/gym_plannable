import unittest
from gym_plannable.env.grid_world import MazeEnv
from plannable_mixins import PlannableInterfaceTestMixin, PlannableEnvTestMixin
from multi_agent_mixins import EnvTestMixin

class PlannableInterfaceTestMazeEnv(PlannableInterfaceTestMixin, unittest.TestCase):
    env_constructor = MazeEnv
    max_next = 5

class PlannableEnvTestMazeEnv(PlannableEnvTestMixin, unittest.TestCase):
    env_constructor = MazeEnv

class EnvTestMazeEnv(EnvTestMixin, unittest.TestCase):
    env_constructor = MazeEnv
