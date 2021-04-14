import gym
import numpy as np
from gym_plannable.multi_agent import MultiAgentEnv

class DummyEnvTurnBased(MultiAgentEnv):
    def __init__(self, num_agents=2,
                 num_steps=5, num_actions=4,
                 exception_at=None, **kwargs):
        super().__init__(num_agents=num_agents, **kwargs)

        if self.num_agents == 1:
            self.observation_space = gym.spaces.Discrete(num_steps)
            self.action_space = gym.spaces.Discrete(num_actions)
        else:
            self.observation_space = [gym.spaces.Discrete(num_steps)] * self.num_agents
            self.action_space = [gym.spaces.Discrete(num_actions)] * self.num_agents

        self.exception_at = exception_at
        self.num_steps = num_steps

    @property
    def agent_turn(self):
        return [self._step % self.num_agents]

    def reset(self):
        self._step = 0
        obs = [self._step % self.num_agents for i in range(self.num_agents)]
        return self._wrap_outputs(obs)
 
    def step(self, action):
        # there is always exactly 1 agent turning
        action = self._wrap_inputs(action)
        assert len(action) == 1
        action = action[0]
        agentid = self.agent_turn[0]
        
        if not self.action_space[agentid].contains(action):
            raise ValueError("Invalid action: '{}'.".format(action))

        if not self.exception_at is None and self.exception_at >= self._step:
            raise RuntimeError("Raising a planned exception at step {}.".format(self._step))

        obs = [self._step % self.num_agents for i in range(self.num_agents)]
        rewards = np.zeros(self.num_agents)
        rewards[agentid] = action
        info = [{}] * self.num_agents

        if self._step < self.num_steps:
            done = np.zeros(self.num_agents, dtype=np.bool)
        else:
            done = np.ones(self.num_agents, dtype=np.bool)

        self._step += 1

        return self._wrap_outputs(obs, rewards, done, info)
