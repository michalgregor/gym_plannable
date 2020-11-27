import abc

class ScoreTracker:
    @abc.abstractmethod
    def update_scores(self, rewards):
        pass

    @abc.abstractproperty
    def scores(self):
        pass

class ScoreTrackerTotal(ScoreTracker):
    def __init__(self):
        self._scores = 0

    def update_scores(self, rewards):
        self._scores += rewards

    @property
    def scores(self):
        return self._scores
