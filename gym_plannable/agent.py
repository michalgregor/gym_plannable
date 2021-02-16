from .common import ClosedEnvSignal
from threading import Thread
import numpy as np
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

    def __call__(self):
        try:
            self.episode = 0

            while self.num_episodes is None or self.episode < self.num_episodes:
                self.env.reset()
                done = False
                self.steps = 0

                while not done and (self.max_steps is None or
                    self. steps < self.max_steps
                ):
                    self.steps += 1
                    state = self.env.plannable_state()

                    if self.show_times:
                        start = time.perf_counter()

                    action = self.select_action(state)

                    if self.show_times:
                        end = time.perf_counter()
                        print("Action selected in {} s.".format(end-start))
                    
                    _, _, done, info = self.env.step(action)

                    if self.verbose and info.get('interrupted'):
                        print("agent {} interrupted; done={}".format(self.env.agentid, done))

                if not self.episode_scores is None:
                    self.episode_scores.append(copy.deepcopy(state.scores))

                if self.verbose:
                    print("Episode {} done; scores: {}".format(self.episode, state.scores))

                self.episode += 1

            self.env.finish()

        except ClosedEnvSignal:
            if self.verbose: print("Exited because of a closed environment.")

        except:
            # if the agent crashes irretrievably, make sure the
            # env is finished to prevent lock ups in the remaining threads
            self.env.finish()
            raise

    @abc.abstractmethod
    def select_action(self, state):
        raise NotImplementedError()

    def start(self):
        """
        Starts the agent in a new thread.
        """
        thread = Thread(target=self)
        thread.start()
        return thread

class LegalAgent(BaseAgent):
    def select_action(self, state):
        legals = state.legal_actions()
        action = random.choice(legals)
        return action
