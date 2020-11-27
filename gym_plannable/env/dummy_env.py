from ..turn_based import TurnBasedEnv
import numpy as np
import gym

class DummyEnv(TurnBasedEnv):
    """
    A dummy environment for testing.
    """
    def __init__(self, size=3):
        super().__init__(2)

        self.observation_space = gym.spaces.Box(
            low=np.full((size, size), -1, dtype=np.int),
            high=np.full((size, size), self.num_agents-1, dtype=np.int),
            dtype=np.int
        )

        self.action_space = gym.spaces.Box(
            low=np.asarray((0, 0), dtype=np.int),
            high=np.asarray((size-1, size-1), dtype=np.int),
            dtype=np.int
        )

        self._agent_turn = 0
        self._step = 0

    def plannable_state(self):
        class State:
            def legal_actions(self):
                return [0, 1]

        return State()

    @property
    def agent_turn(self):
        return self._agent_turn

    def reset(self):
        return self.observation_space.sample()
 
    def step(self, action, agentid=None):
        """
        Performs the action and returns the next observation, reward,
        done flag and info dict.
        """
        rewards = np.full(self.num_agents, 1)
        done = self._step >= 10
        self._step += 1

        self._agent_turn = (self._agent_turn + 1) % self.num_agents
        
        return self.observation_space.sample(), rewards, done, {}
