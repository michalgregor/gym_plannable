import unittest
from gym_plannable.env.grid_world import MazeEnv
from plannable_mixins import PlannableInterfaceTestMixin

class PlannableInterfaceTestMazeEnv(PlannableInterfaceTestMixin, unittest.TestCase):
    env_constructor = MazeEnv
    max_next = 5

import collections
from gym_plannable.multi_agent import (
    ErrorMessage, MultiAgentServer, ActionMessage, ResetMessage,
    ObservationMessage, multi_agent_to_single_agent
)

import gc
import weakref
from threading import Thread

from gym_plannable.agent import LegalAgent


class PlannableEnvTestMazeEnv(unittest.TestCase):
    env_constructor = MazeEnv

    def setUp(self):
        self.env = self.env_constructor()
        self.clients = multi_agent_to_single_agent(self.env)

        self.stopped = False
        def stop_callback():
            self.stopped = True

        self.clients[0].server.stop_callback = stop_callback

    def tearDown(self):
        del self.clients
        gc.collect()
        self.assertTrue(self.stopped)

    def testStartStop(self):
        pass

    def testRun(self):
        agents = [LegalAgent(env, num_episodes=1, max_steps=5) for env in self.clients]

        for a in agents:
            a.start()

        for a in agents:
            a.join()
