from .common import ClosedEnvSignal
from .multi_agent import StopServerException
from .plannable import PlannableStateSingleWrapper
from threading import Thread
import random
import time
import copy
import abc

class BaseAgent:
    def __init__(self, env, num_episodes=None, max_steps=None,
                 verbose=False, show_times=False, record_scores=True):
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.verbose = verbose
        self.step = 0
        self.episode = 0
        self.show_times = show_times

        if record_scores:
            self.episode_scores = []
        else:
            self.episode_scores = None

    @property
    def agentid(self):
        return self.env.agentid

    def _get_plannable_state(self):
        return self.env.plannable_state()

    def _handle_exception(self, e):
        # if the agent crashes irretrievably, make sure the
        # env is finished to prevent lock ups in the remaining threads
        self.env.close()
        raise e

    def __call__(self):
        try:
            self.episode = 0

            while self.num_episodes is None or self.episode < self.num_episodes:
                self.env.reset()
                terminated = False
                truncated = False
                self.steps = 0

                while not terminated and not truncated and (
                    self.max_steps is None or
                    self.steps < self.max_steps
                ):
                    self.steps += 1
                    state = self._get_plannable_state()

                    if self.show_times:
                        start = time.perf_counter()

                    action = self.select_action(state)

                    if self.show_times:
                        end = time.perf_counter()
                        print("Action selected in {} s.".format(end-start))

                    _, _, terminated, truncated, info = self.env.step(action)

                    if self.verbose and info.get('interrupted'):
                        print("agent {} interrupted; terminated={}; truncated={};".format(
                            self.env.agentid, terminated, truncated
                        ))

                if not self.episode_scores is None:
                    self.episode_scores.append(copy.deepcopy(state.scores()))

                if self.verbose:
                    print("Episode {} done; scores: {}".format(self.episode, state.scores()))

                self.episode += 1

            self.env.close()

        except ClosedEnvSignal:
            if self.verbose: print("Exited because of a closed environment.")

        except StopServerException:
            return
            
        except BaseException as e:
            self._handle_exception(e)

    @abc.abstractmethod
    def select_action(self, state):
        raise NotImplementedError()

    def start(self):
        """
        Starts the agent in a new thread.
        """
        self.thread = Thread(target=self)
        self.thread.start()
        return self.thread

    def join(self, **kwargs):
        """
        Joins the agent's thread.
        """
        return self.thread.join(**kwargs)

class SingleBaseAgent(BaseAgent):
    def _get_plannable_state(self):
        return PlannableStateSingleWrapper(self.env.plannable_state())

class LegalAgent(SingleBaseAgent):
    def select_action(self, state):
        legals = state.legal_actions()
        action = random.choice(legals)
        return action
