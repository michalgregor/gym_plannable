import abc
import numbers
import numpy as np
from .score_tracker import ScoreTrackerTotal
from .common import State
from .multi_agent import MultiAgentEnv

class SamplePlannableState(State):
    def __init__(self, score_tracker=None):
        """
        Arguments:
            score_tracker: A ScoreTracker object that is going to track
                           agents' scores.
        """
        if score_tracker is None:
            self.score_tracker = ScoreTrackerTotal()
        else:
            self.score_tracker = score_tracker

    def _select_who(self, seq, who):
        """
        Filters an input sequence that contains one item for each
        agent in the environment.

        Arguments:
            - who ("all", "turn", "single"): If "all", this returns a list with
              an item for each agent in the environment; if "turn", this returns
              a list with an item for each currently turning agent; if "single"
              this asserts that there is a single turning agent and returns
              its corresponding item.
        """
        if who == "all":
            return seq
        elif who == "turn":
            return [seq[a] for a in self.agent_turn]
        elif who == "single":
            assert len(self.agent_turn) == 1
            return seq[self.agent_turn[0]]
        else:
            raise ValueError('Unknown who "{}".'.format(who))

    def scores(self, who='all'):
        """
        Returns the scores associated with the state. A score is a measure that
        summarizes rewards received by an agent. It can implemented in
        different ways, e.g. as the total sum of rewards; by default, it is
        computed using a ScoreTracker from gym_plannable.score_tracker.

        Arguments:
            - who ("all", "turn", "single"): If "all", this returns a list with
              an item for each agent in the environment; if "turn", this returns
              a list with an item for each currently turning agent; if "single"
              this asserts that there is a single turning agent and returns
              its corresponding item.
        """
        scores = self.score_tracker.scores
        if isinstance(scores, numbers.Number):
            scores = np.full(self.num_agents, scores)
        
        return self._select_who(scores, who)

    @staticmethod
    def _make_state(state):
        state.score_tracker.update_scores(state.rewards())
        return state

    @abc.abstractmethod
    def _next(self, actions, *args, inplace=False, **kwargs):
        """
        Returns the next state. If stochastic, one random next state is sampled.
        """
        raise NotImplementedError()

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
    def _init(self, inplace=False):
        """
        Returns an initial state.

        Arguments:
            * inplace: If inplace is true, the state should be modified in place
                       and self should be returned.
        """
        raise NotImplementedError()

    def init(self, inplace=False):
        """
        Returns an initial state.

        Arguments:
            * inplace: If inplace is true, the state should be modified in place
                       and self should be returned.
        """
        return self._init(inplace=inplace)

    @abc.abstractmethod
    def _legal_actions(self):
        """
        Returns the sequence of all actions that are legal in the state for each agent.
        """
        raise NotImplementedError()

    def legal_actions(self, who="all"):
        """
        Returns the sequence of all actions that are legal in the state for each agent.

        Arguments:
            - who ("all", "turn", "single"): If "all", this returns a list with
              an item for each agent in the environment; if "turn", this returns
              a list with an item for each currently turning agent; if "single"
              this asserts that there is a single turning agent and returns
              its corresponding item.
        """
        return self._select_who(self._legal_actions(), who)

    @abc.abstractmethod
    def _is_done(self):
        """
        Returns a sequence of boolean values, indicating whether this is a
        terminal state for each of the agents.
        """
        raise NotImplementedError()

    def is_done(self, who='all'):
        """
        Returns a sequence of boolean values, indicating whether this is a
        terminal state for each of the agents.

        Arguments:
            - who ("all", "turn", "single"): If "all", this returns a list with
                an item for each agent in the environment; if "turn", this returns
                a list with an item for each currently turning agent; if "single"
                this asserts that there is a single turning agent and returns
                its corresponding item.
        """
        return self._select_who(self._is_done(), who)

    @abc.abstractmethod
    def _observations(self):
        """
        Returns a sequence of observations associated with the state:
        one for each agent.
        """
        raise NotImplementedError()

    def observations(self, who='all'):
        """
        Returns a sequence of observations associated with the state:
        one for each agent.

        Arguments:
            - who ("all", "turn", "single"): If "all", this returns a list with
                an item for each agent in the environment; if "turn", this returns
                a list with an item for each currently turning agent; if "single"
                this asserts that there is a single turning agent and returns
                its corresponding item.
        """
        return self._select_who(self._observations(), who)

    @abc.abstractmethod
    def _rewards(self):
        """
        Returns a sequence containing the rewards for all agents (for a
        single-agent environment this is going to be a sequence with 1 item).
        """
        raise NotImplementedError()

    def rewards(self, who='all'):
        """
        Returns a sequence containing the rewards for all agents (for a
        single-agent environment this is going to be a sequence with 1 item).

        Arguments:
            - who ("all", "turn", "single"): If "all", this returns a list with
                an item for each agent in the environment; if "turn", this returns
                a list with an item for each currently turning agent; if "single"
                this asserts that there is a single turning agent and returns
                its corresponding item.
        """
        return self._select_who(self._rewards(), who)

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
    @abc.abstractmethod
    def _all_next(self, actions, *args, **kwargs):
        """
        Returns a generator of (state, probability) tuples for all possible
        next states.
        """
        raise NotImplementedError()
    
    def all_next(self, actions, *args, **kwargs):
        """
        Returns a generator of (state, probability) tuples for all possible
        next states.

        In the background, scores are tracked.
        """
        return ((self._make_state(ns), prob)
            for (ns, prob) in self._all_next(actions, *args, **kwargs))

    @abc.abstractmethod
    def _all_init(self):
        """
        Returns a generator of (state, probability) tuples for all possible
        initial states.
        """
        raise NotImplementedError()

    def all_init(self):
        """
        Returns a generator of (state, probability) tuples for all possible
        initial states.
        """
        return self._all_init()

class PlannableStateDeterministic(PlannableState):
    def _all_init(self):
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
