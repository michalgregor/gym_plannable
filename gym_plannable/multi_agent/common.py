import collections
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
    def __init__(self, observation, reward=0, done=False, info=None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}

    def totuple(self):
        return self.observation, self.reward, self.done, self.info

class ErrorMessage:
    def __init__(self, msg):
        self.msg = msg
        
class ErrorException(Exception):
    pass
        
class StopServerMessage:
    pass

class StopServerException(Exception):
    pass

class ActionCollector:
    def __init__(self, agentids=[]):
        """
        A class that, given a list of agents' ids, manages collecting
        their actions (or requests to interrupt the episode).

        Arguments:
        - agentids: A list containing ids of agents whose turn it is and their
                    actions are to be collected.
        """
        self.reset(agentids)
        
    @property
    def all_collected(self):
        """
        Returns True if all agents' actions have been collected.
        """
        return self._collected == len(self._actions)

    @property
    def agent_turn(self):
        """
        Returns a view of agent ids for agents whose turn it currently is.
        """
        return self._actions.keys()
    
    def collect(self, msg):
        """
        Registers an agent's action (given an ActionMessage) or an agent's
        request for the episode to be interrupted (given a ResetMessage).

        Arguments:
        - msg: An ActionMessage or a ResetMessage to collect.

        Raises:
        - A ValueError if the agent's action has already been collected;
        - A ValueError if it is currently not the agent's turn.
        - A TypeError if msg is neither an ActionMessage, nor a ResetMessage.
        """
        agentid = msg.agentid
        
        try:
            if not self._actions[agentid] is None:
                raise ValueError("An action has already been collected for agent {}.".format(agentid))
        except KeyError:
            raise ValueError("It is currently not agent {}'s turn.".format(agentid))
            
        if isinstance(msg, ActionMessage):
            self._actions[agentid] = msg.action
        elif isinstance(msg, ResetMessage):
            self._actions[agentid] = ResetAction()
            self.interrupted = True
        else:
            raise TypeError("Unexpected message type '{}'.".format(type(msg)))
        
        self._collected += 1
    
    def get_actions(self):
        """
        Returns a view of the collected actions in the order the agentids
        were specified.

        Raises:
        - A RuntimeError if all actions have not yet been collected.
        """
        if not self.all_collected:
            raise RuntimeError("Actions not yet collected for all agentids.")
        else:
            return self._actions
        
    def reset(self, agentids=[]):
        """
        Resets action collection by removing any already collected actions
        and setting up for collecting new actions from the specified agents.

        Arguments:
        - agentids: A list containing ids of agents whose turn it is and their
                    actions are to be collected.
        """
        self._actions = collections.OrderedDict.fromkeys(agentids)
        self._collected = 0
        self.interrupted = False
