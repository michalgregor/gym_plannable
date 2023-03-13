from queue import Empty
from multiprocessing.managers import RemoteError
from multiprocessing import Manager

class ManagerSingleton:
    def __init__(self):
        self.manager = None

    def __call__(self):
        if self.manager is None:
            self.manager = Manager()
        return self.manager

manager_singleton = ManagerSingleton()

class ClientServerInterface:
    def __init__(self,
        num_agents,
        observation_spaces,
        action_spaces,
        reward_ranges
    ):
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.reward_ranges = reward_ranges

        manager = manager_singleton()
        self.outgoing_messages = [manager.Queue() for _ in range(num_agents)]
        self.incoming_messages = manager.Queue()
        self.started_event = manager.Event()
        self.finished_event = manager.Event()
        self.stop_lock = manager.Lock()

    def _stop(self):
        if self.started_event.is_set() and not self.finished_event.is_set():
            self.incoming_messages.put(StopServerMessage())

            for oq in self.outgoing_messages:
                # clear away any existing message
                try:
                    oq.get_nowait()
                except Empty:
                    pass

                oq.put_nowait(StopServerMessage())

    def stop(self, wait=True):
        """
        Stops the server. This function is thread-safe.

        Arguments:
        - wait: If True, block until the server thread terminates.
        """

        # note: things can fail if the other side has already disconnected
        # from the manager: therefore the exceptions
        try:
            # if false, another thread is already stopping the server
            if self.stop_lock.acquire(blocking=False):
                try:
                    self._stop()
                except BaseException as e:
                    self.stop_lock.release()
                    raise e

            if wait and self.started_event.is_set(): self.finished_event.wait()
        
        except FileNotFoundError:
            pass
        except RemoteError:
            pass

class ResetMessage:
    def __init__(self, agentid):
        self.agentid = agentid
        
class ResetAction:
    pass

class ActionMessage:
    def __init__(self, action, agentid):
        self.action = action
        self.agentid = agentid
        
class ObservationMessage:
    def __init__(self, observation, reward=0, terminated=False, truncated=False, info=None):
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info or {}

    def totuple(self):
        return self.observation, self.reward, self.terminated, self.truncated, self.info

class ErrorMessage:
    def __init__(self, msg):
        self.msg = msg
        
class ErrorException(Exception):
    pass
        
class StopServerMessage:
    pass

class StopServerException(Exception):
    pass
