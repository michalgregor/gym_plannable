from .common import ClosedEnvSignal
import numpy as np
import time
import abc

class BaseAgent:
    def __init__(self, env, num_epochs, max_steps=None,
                 verbose=True, show_times=True):
        self.env = env
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.verbose = verbose
        self.step = 0
        self.episode = 0
        self.show_times = show_times

    @property
    def agentid(self):
        return self.env.agentid

    def __call__(self):
        try:
            self.episode = 0

            for i in range(self.num_epochs):
                self.env.reset()
                done = False
                self.steps = 0

                while not done and (self.max_steps is None or self.steps < self.max_steps):
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

                self.episode += 1

            self.env.finish()

        except ClosedEnvSignal:
            if self.verbose: print("Exited because of a closed environment.")

    @abc.abstractmethod
    def select_action(self, state):
        pass

class LegalAgent(BaseAgent):
    def select_action(self, state):
        legals = state.legal_actions()
        action = legals[np.random.choice(len(legals))]
        return action
