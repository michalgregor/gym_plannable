import gym
import abc
from .common import ClosedEnvSignal
from threading import Event, Barrier
import numpy as np
import itertools

class TurnBasedState:
    @property
    @abc.abstractmethod
    def agent_turn(self):
        """
        Returns the numeric index of the agent which is going to move next.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def num_agents(self):
        """
        Returns the number of agents in the environment.
        """
        raise NotImplementedError()

class TurnBasedEnv(gym.Env):
    def __init__(self, num_agents):
        """
        A base class for multiplayer environments.

        Unlike the base gym.Env, action_space, observation_space
        and reward_range should all be lists: there should be an entry
        for each agent.

        You can use turn_based_to_single_agent() to turn a TurnBasedEnv into
        several connected single-agent environments.
        """
        super().__init__()
        self.num_agents = num_agents
        self.reward_range = [self.reward_range] * self.num_agents

    @property
    @abc.abstractmethod
    def agent_turn(self):
        """
        Returns the numeric index of the agent which is going to move next.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, action, agentid=None):
        """
        Applies the specified action and activates the state transition.
        If agentid is specified and does not match self.agent_turn,
        a ValueError is raised (useful as a consistency check).

        This returns what a single-agent step method would return except that
        instead of a single reward, there will be a list of rewards, one
        for each agent.

        Arguments:
          * action: The action to apply.
          * agentid: The id of the agent that selected the action. If None,
            self.agent_turn will be used instead.
        """
        raise NotImplementedError()

class SharedState:
    def __init__(self, num_agents):
        assert num_agents >= 2
        self.turn_events = [Event() for i in range(num_agents)]
        self.rewards = np.zeros(num_agents)
        self.done_barrier = Barrier(num_agents)
        self._obs_tuple = [None, 0, False, {}]
        self._is_initial_state = True
        self.reset_done = False
        self.reset_counters()
        self._closed_signal = False
        self._handling_done = False

    def reset_counters(self):
        self._reset_counters = np.zeros(len(self.turn_events), dtype=np.bool)
        self.reset_done = False

    def was_reset(self, agentid, value=None):
        if not value is None:
            self._reset_counters[agentid] = value
            if value:
                self.reset_done = True
        return self._reset_counters[agentid]

    def pass_turn(self, from_agentid, to_agentid, pass2next=False):
        """
        Pass the turn to the specified agent (if pass2next is False) or to the
        agent after that one in the list (if pass2next is True).
        """
        self.turn_events[from_agentid].clear()
        if pass2next:
            self.turn_events[(to_agentid+1) % len(self.turn_events)].set()
        else:
            self.turn_events[to_agentid].set()

    def _get_obs_tuple(self):
        return self._obs_tuple

    def _set_obs_tuple(self, obs_tuple):
        if len(obs_tuple) > 1:
            self._obs_tuple = list(obs_tuple)
        else:
            self._obs_tuple = [obs_tuple[0], 0, False, {}]

    obs_tuple = property(_get_obs_tuple, _set_obs_tuple)

    def wait_turn(self, agentid):
        self.turn_events[agentid].wait()
        self.handle_signals()

    def clear_turn(self, agentid):
        self.turn_events[agentid].clear()

    def is_agents_turn(self, agentid):
        return self.turn_events[agentid].is_set()

    def is_done(self):
        return self._obs_tuple[2]

    def handle_signals(self):
        if self._closed_signal: raise ClosedEnvSignal()

    def signal_close(self):
        self._closed_signal = True
        self._unblock()

    def _unblock(self, except_agentid=None):
        if except_agentid is None:
            for event in self.turn_events:
                event.set()
        else:
            for event in itertools.chain(
                self.turn_events[:except_agentid],
                self.turn_events[except_agentid+1:]
            ):
                event.set()

    def signal_done(self, agentid=None):
        if not self._handling_done:
            self._handling_done = True
            # ensure that the state is terminal
            self.obs_tuple[2] = True

            # set up reset
            self.reset_counters()

            # unblock all except agentid (if given)
            self._unblock(agentid)
            self.done_barrier.wait()
            self._handling_done = False
        else:
            self.done_barrier.wait()
            self.clear_turn(agentid)

class SingleAgentEnvTurnBased(gym.Wrapper):
    def __init__(self, turn_based_env, shared_state, agentid):
        super().__init__(turn_based_env)
        self.agentid = agentid
        self.shared_state = shared_state
        self.observation_space = turn_based_env.observation_space[agentid]
        self.action_space = turn_based_env.action_space[agentid]

    def reset(self):
        # wait our turn
        self.shared_state.wait_turn(self.agentid)

        # if was_reset is true, this agent was already reset after a terminal
        # state has last been reached; if requesting a reset in the middle of
        # an episode, the other agents need to be signalled
        if (self.shared_state.was_reset(self.agentid) and 
           not self.shared_state._is_initial_state):
            # augment obs_tuple with info about the episode being interrupted
            self.shared_state.obs_tuple[3]['interrupted'] = True
            self.shared_state.signal_done(self.agentid)

        # if reset of the underlying env not already done, do it now
        if not self.shared_state.reset_done:
            self.shared_state._is_initial_state = True
            self.shared_state.obs_tuple = (self.env.reset(),)
            self.shared_state.rewards *= 0

        # record the reset
        self.shared_state.was_reset(self.agentid, True)

        # pass the turn to the next agent
        self.shared_state.pass_turn(self.agentid, self.env.agent_turn)

        # wait for the turn back and do not pass it on:
        # to unblock step after reset
        self.shared_state.wait_turn(self.agentid)

        # handle a terminal state
        if self.shared_state.is_done():
            self.shared_state.signal_done(self.agentid)

        # we only return the observation
        return self.shared_state.obs_tuple[0]

    def finish(self):
        # if was_reset is true, this agent is in the middle of an episode;
        # the other agents need to be signalled
        if self.shared_state.was_reset(self.agentid):
            # augment obs_tuple with info about the episode being interrupted
            self.shared_state.obs_tuple[3]['interrupted'] = True
            self.shared_state.signal_done(self.agentid)

        self.close()

    def step(self, action):
        # if calling step, this should always already be our turn
        assert self.shared_state.is_agents_turn(self.agentid)

        # check that this env's agent did not forget to call reset
        # after getting into a terminal state
        assert self.shared_state.was_reset(self.agentid)

        # perform the step
        self.shared_state._is_initial_state = False
        self.shared_state.obs_tuple = self.env.step(action, self.agentid)
        # accumulate rewards
        self.shared_state.rewards += self.shared_state.obs_tuple[1]

        # pass the turn on
        self.shared_state.pass_turn(self.agentid, self.env.agent_turn)
        
        # wait to get the turn back
        self.shared_state.wait_turn(self.agentid)
        obs_tuple = self.shared_state.obs_tuple
        # get agentid's reward since the last step; also reset the counter
        reward = self.shared_state.rewards[self.agentid]
        self.shared_state.rewards[self.agentid] = 0

        # handle a terminal state
        if self.shared_state.is_done():
            self.shared_state.signal_done(self.agentid)

        # return the observation
        return tuple(obs_tuple[0:1]) + (reward,) + tuple(obs_tuple[2:])

    def close(self):
        self.env.close()
        self.shared_state.signal_close()

    def __del__(self):
        self.close()

def turn_based_to_single_agent(turn_based_env):
    shared_state = SharedState(turn_based_env.num_agents)
    envs = [SingleAgentEnvTurnBased(turn_based_env, shared_state, agentid)
                for agentid in range(turn_based_env.num_agents)]
    # set the first event to set the chain off
    shared_state.pass_turn(0, 0)
    return envs
