import abc
import gym
from .score_tracker import ScoreTrackerTotal

class PlannableState:
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

    def next(self, action, *args, **kwargs):
        """
        Returns the next state. If stochastic, one random next state is sampled.

        In the background, scores are tracked.
        """
        return self._make_state(self._next(action, *args, **kwargs))

    def all_next(self, action, *args, **kwargs):
        """
        Returns a generator of (state, probability) tuples for all possible
        next states.

        In the background, scores are tracked.
        """
        return ((self._make_state(ns), prob)
            for (ns, prob) in self._all_next(action, *args, **kwargs))

    @abc.abstractmethod
    def init(self):
        """
        Returns an initial state.
        """

    @abc.abstractmethod
    def all_init(self):
        """
        Returns a generator of (state, probability) tuples for all possible
        initial states.
        """

    @abc.abstractmethod
    def legal_actions(self):
        """
        Returns the sequence of all actions that are legal in the state.
        """

    @abc.abstractmethod
    def _next(self, action, *args, **kwargs):
        """
        Returns the next state. If stochastic, one random next state is sampled.
        """

    @abc.abstractmethod
    def _all_next(self, action, *args, **kwargs):
        """
        Returns a generator of (state, probability) tuples for all possible
        next states.
        """

    @abc.abstractmethod
    def is_done(self):
        """
        Returns whether this is a terminal state or not.
        """

    @abc.abstractproperty
    def rewards(self):
        """
        Returns a sequence containing the rewards for all agents (for a
        single-agent environment this is going to be a sequence with 1 item).
        """

    @abc.abstractmethod
    def observation(self):
        """
        Returns the observation associated with the state.
        """

class PlannableStateDeterministic(PlannableState):
    def all_init(self):
        return ((self.init(), 1.0) for i in range(1))

    def _all_next(self, action, *args, **kwargs):
        return ((self.next(action, *args, **kwargs), 1.0) for i in range(1))

class PlannableEnv(gym.Env):
    @abc.abstractmethod
    def plannable_state(self) -> PlannableState:
        """
        Returns the environment's current state as PlannableState.

        This function must be callable before reset().
        """
        pass
