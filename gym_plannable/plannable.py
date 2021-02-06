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

    def next(self, action, *args, **kwargs):
        next_state = self._next(action, *args, **kwargs)
        next_state.score_tracker.update_scores(next_state.rewards)
        return next_state

    @abc.abstractmethod
    def legal_actions(self):
        pass

    @abc.abstractmethod
    def _next(self, action, *args, **kwargs):
        pass

    @abc.abstractmethod
    def all_next(self, action, *args, **kwargs):
        pass

    @abc.abstractmethod
    def is_done(self):
        pass

    @abc.abstractproperty
    def rewards(self):
        pass

    @abc.abstractmethod
    def observation(self):
        pass

class PlannableStateDeterministic(PlannableState):
    def all_next(self, action, *args, **kwargs):
        return ((self.next(action, *args, **kwargs), 1.0) for i in range(1))

class PlannableEnv(gym.Env):
    @abc.abstractmethod
    def plannable_state(self) -> PlannableState:
        """
        Returns the environment's current state as PlannableState.
        """
        pass
