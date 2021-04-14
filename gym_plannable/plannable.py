import abc
from .score_tracker import ScoreTrackerTotal
from .common import State
from .multi_agent import MultiAgentEnv

class SamplePlannableState(State):
    def __init__(self, score_tracker=None):
        if score_tracker is None:
            self.score_tracker = ScoreTrackerTotal()
        else:
            self.score_tracker = score_tracker

    @property
    def scores(self):
        return self.score_tracker.scores

    @staticmethod
    def _make_state(state):
        state.score_tracker.update_scores(state.rewards)
        return state

    def next(self, actions, *args, inplace=False, **kwargs):
        """
        Returns the next state. If stochastic, one random next state is sampled.

        In the background, scores are tracked.

        Arguments:
            * actions: The agents' actions.
            * inplace: If inplace is true, the state should be modified in place
                       and self should be returned.
        """
        return self._make_state(self._next(actions, *args, inplace=inplace, **kwargs))

    @abc.abstractmethod
    def init(self, inplace=False):
        """
        Returns an initial state.

        Arguments:
            * inplace: If inplace is true, the state should be modified in place
                       and self should be returned.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def legal_actions(self):
        """
        Returns the sequence of all actions that are legal in the state for each agent.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _next(self, actions, *args, inplace=False, **kwargs):
        """
        Returns the next state. If stochastic, one random next state is sampled.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_done(self):
        """
        Returns a sequence of boolean values, indicating whether this is a
        terminal state for each of the agents (for a single-agent environment
        this is going to be a sequence with 1 item).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def observation(self):
        """
        Returns a sequence of observations associated with the state:
        one for each agent.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def rewards(self):
        """
        Returns a sequence containing the rewards for all agents (for a
        single-agent environment this is going to be a sequence with 1 item).
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def agent_turn(self):
        """
        Returns a sequence of numeric indices of the agents
        that are going to move next.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def num_agents(self):
        """
        Returns the number of agents in the environment.
        """
        raise NotImplementedError()

class PlannableState(SamplePlannableState):
    def all_next(self, actions, *args, **kwargs):
        """
        Returns a generator of (state, probability) tuples for all possible
        next states.

        In the background, scores are tracked.
        """
        return ((self._make_state(ns), prob)
            for (ns, prob) in self._all_next(actions, *args, **kwargs))

    @abc.abstractmethod
    def all_init(self):
        """
        Returns a generator of (state, probability) tuples for all possible
        initial states.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _all_next(self, actions, *args, **kwargs):
        """
        Returns a generator of (state, probability) tuples for all possible
        next states.
        """
        raise NotImplementedError()

class PlannableStateDeterministic(PlannableState):
    def all_init(self):
        return ((self.init(), 1.0) for i in range(1))

    def _all_next(self, actions, *args, **kwargs):
        return ((self.next(actions, *args, **kwargs), 1.0) for i in range(1))

class SamplePlannableEnv(MultiAgentEnv):
    def __init__(self, num_agents=1, **kwargs):
        super().__init__(num_agents=num_agents, **kwargs)

    @abc.abstractmethod
    def plannable_state(self) -> SamplePlannableState:
        """
        Returns the environment's current state as SamplePlannableState.

        This function must be callable before reset().
        """
        raise NotImplementedError()

class PlannableEnv(MultiAgentEnv):
    def __init__(self, num_agents=1, **kwargs):
        super().__init__(num_agents=num_agents, **kwargs)

    @abc.abstractmethod
    def plannable_state(self) -> PlannableState:
        """
        Returns the environment's current state as PlannableState.

        This function must be callable before reset().
        """
        raise NotImplementedError()
