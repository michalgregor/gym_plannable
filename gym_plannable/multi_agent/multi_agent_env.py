import gymnasium as gym
import abc

class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents, **kwargs):
        """
        A base class for multiagent environments.

        Unlike gym.Env environments, instead of action_space, observation_space
        and reward_range, the environment has:
        * action_spaces;
        * observation_spaces; and
        * reward_ranges,
        which are all sequences with an entry for each agent.

        Similarly, reset() and step() also return sequences: i.e. reset()
        returns a sequence of observations, one for each agent and step()
        returns several such sequences for observations, rewards, terminated,
        truncated and info.
        
        Finally, actions are presented to step() as a sequence: environments
        can be either turn-based or use simultaneous actions, but a sequence
        of actions is used in either case.

        You can use multi_agent_to_single_agent() to turn a MultiAgentEnv into
        several connected single-agent environments with the standard Gym
        interface.
        """
        super().__init__(**kwargs)
        self.num_agents = num_agents
        self.reward_ranges = [self.reward_range] * self.num_agents

    @property
    @abc.abstractmethod
    def agent_turn(self):
        """
        Returns a sequence of numeric indices corresponding to the agents
        which are going to move next.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, actions):
        """
        Applies the specified actions and activates the state transition.

        This returns what a single-agent step method would return except that:
        * a sequence of num_agents observations will be returned;
        * a sequence of num_agents rewards will be returned;
        * a sequence of num_agents terminated flags will be returned;
        * a sequence of num_agents truncated flags will be returned;
        * a sequence of num_agents info dictionaries will be returned;
        
        Also, actions is a sequence (even if only a single agent is turning
        at each step).
                
        Arguments:
          * actions: The action(s) to apply; the order corresponds to the
                order of agents in agent_turn.
        """
        raise NotImplementedError()

    def checked_step(self, actions, agentids):
        """
        Does the same as self.step(actions), but first checks that
        agentids (i.e. the sequence of ids corresponding to the agents that
        selected the actions) matches agent_turn.
        """
        if list(agentids) != list(self._state.agent_turn):
            raise ValueError("It is not the agents' turn: '{}'.".format(agentids))

        return self.step(actions)

class Multi2SingleWrapper(gym.Env):
    """
    Wraps a multi-agent MultiAgentEnv environment with num_agents == 1 as
    a standard single-agent Gym environment.
    """
    def __init__(self, env):
        if not isinstance(env, MultiAgentEnv) or env.num_agents != 1:
            raise ValueError("A MultiAgentEnv object with num_agents == 1 was expected.")
        
        self.env = env
        self.observation_space = self.env.observation_spaces[0]
        self.action_space = self.env.action_spaces[0]
        self.reward_range = self.env.reward_ranges[0]

    def reset(self):
        obs, infos = self.env.reset()
        return obs[0], infos[0]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step([action])
        return obs[0], reward[0], terminated[0], truncated[0], info[0]
        
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self._state)

    def __repr__(self):
        return str(self)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped
