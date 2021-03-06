from queue import Queue, Empty
import numpy as np
import collections
import threading
import weakref
import gym
import abc

class Multi2SingleWrapper(gym.Env):
    """
    Wraps a multi-agent MultiAgentEnv environment with num_agents == 1 as
    a standard single-agent Gym environment.
    """
    def __init__(self, env):
        if not isinstance(env, MultiAgentEnv) or env.num_agents != 1:
            raise ValueError("A MultiAgentEnv object with num_agents == 1 was expected.")
        
        self.env = env
        self.observation_space = self.env.observation_spaces[0]
        self.action_space = self.env.action_spaces[0]
        self.reward_range = self.env.reward_ranges[0]

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        return obs[0], reward[0], done[0], info[0]   
        
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self._state)

    def __repr__(self):
        return str(self)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped

class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents, **kwargs):
        """
        A base class for multiagent environments.

        Unlike gym.Env environments, instead of action_space, observation_space
        and reward_range, the environment has:
        * action_spaces;
        * observation_spaces; and
        * reward_ranges,
        which are all sequences with an entry for each agent.

        Similarly, reset() and step() also return sequences: i.e. reset()
        returns a sequence of observations, one for each agent and step()
        returns several such sequences for observations, rewards, done and info.
        
        Finally, actions are presented to step() as a sequence: environments
        can be either turn-based or use simultaneous actions, but a sequence
        of actions is used in either case.

        You can use multi_agent_to_single_agent() to turn a MultiAgentEnv into
        several connected single-agent environments with the standard Gym
        interface.
        """
        super().__init__(**kwargs)
        self.num_agents = num_agents
        self.reward_ranges = [self.reward_range] * self.num_agents

    @property
    @abc.abstractmethod
    def agent_turn(self):
        """
        Returns a sequence of numeric indices corresponding to the agents
        which are going to move next.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, actions):
        """
        Applies the specified actions and activates the state transition.

        This returns what a single-agent step method would return except that:
        * a sequence of num_agents observations will be returned;
        * a sequence of num_agents rewards will be returned;
        * a sequence of num_agents done flags will be returned;
        * a sequence of num_agents info dictionaries will be returned;
        
        Also, actions is a sequence (even if only a single agent is turning
        at each step).
                
        Arguments:
          * actions: The action(s) to apply; the order corresponds to the
                order of agents in agent_turn.
        """
        raise NotImplementedError()

    def checked_step(self, actions, agentids):
        """
        Does the same as self.step(actions), but first checks that
        agentids (i.e. the sequence of ids corresponding to the agents that
        selected the actions) matches agent_turn.
        """
        if list(agentids) != list(self._state.agent_turn):
            raise ValueError("It is not the agents' turn: '{}'.".format(agentids))

        return self.step(actions)

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
        
class MultiAgentServer:
    def __init__(self, multi_agent_env, stop_callback=None):
        """
        A server that manages a multi agent environment, to which several
        clients may connect and present single-agent views of the environment
        to agents that support the OpenAI Gym interface.

        Arguments:
        - multi_agent_env: The multi agent environment (with a MultiAgentEnv
                           interface) that is to be managed by the server.
        """
        self.multi_agent_env = multi_agent_env
        self.obs_msg_buffer = [None for _ in range(self.num_agents)]
        self.outgoing_messages = [Queue() for _ in range(self.num_agents)]
        self.incoming_messages = Queue()
        
        self._reset_expected = np.ones(self.multi_agent_env.num_agents, dtype=np.bool)
        self._reset_requested = np.zeros(self.multi_agent_env.num_agents, dtype=np.bool)
        self._to_stop = False

        self._obs = None
        self._info = None

        self._action_collector = ActionCollector()        
        self._message_handlers = {
            ActionMessage: self._handle_action,
            ResetMessage: self._handle_reset,
            ErrorMessage: self._handle_error,
            StopServerMessage: self._handle_stop,
        }
        
        self._thread = self._create_thread()
        self._stop_callback = [stop_callback]
        self._stop_lock = threading.Lock()

    @property
    def stop_callback(self):
        return self._stop_callback[0]

    @stop_callback.setter
    def stop_callback(self, callback):
        self._stop_callback[0] = callback
    
    def _create_thread(self):
        """
        Creates a thread that runs handle_messages and carries a weak reference
        to self so that it does not prevent deletion of the server.
        """
        return threading.Thread(
            target=type(self).run,
            args=(weakref.proxy(self),)
        )
    
    @property
    def num_agents(self):
        """
        Returns the number of agents in the managed multi agent environment.
        """
        return self.multi_agent_env.num_agents
        
    def is_running(self):
        """
        Returns whether the server is running or not.
        """
        return self._thread.is_alive()

    def start(self):
        """
        Starts the server in a separate thread.

        Raises:
        - RuntimeError if the server thread has not been created yet (it is None).
        - RuntimeError if the server thread is running already.
        """
        if self._thread is None:
            raise RuntimeError("The server thread has not been created yet.")
            
        if self._thread.is_alive():
            raise RuntimeError("The server is already running.")

        self._thread.start()
        
    def run(self):
        """
        The main server loop: meant to be run in a separate thread
        by calling start().
        """
        stop_callback = self._stop_callback

        try:
            while not self._to_stop:
                msg = self.incoming_messages.get()
                
                for msg_type, handler in self._message_handlers.items():
                    if isinstance(msg, msg_type):
                        handler(msg)
                        break
                else:
                    raise ValueError("Unexpected message type '{}'.".format(type(msg)))
        except StopServerException:
            with self.incoming_messages.mutex:
                self.incoming_messages.queue.clear()
        except ReferenceError:
            pass

        if not stop_callback[0] is None:
            stop_callback[0]()

    def _stop(self, wait=True):
        if self._thread.is_alive():
            self._to_stop = True
            self.incoming_messages.put(StopServerMessage())

            for oq in self.outgoing_messages:
                # clear away any existing message
                try:
                    oq.get_nowait()
                except Empty:
                    pass

                oq.put_nowait(StopServerMessage())

            if wait: self._thread.join()

    def stop(self, wait=True):
        """
        Stops the server. This function is thread-safe.

        Arguments:
        - wait: If True, block until the server thread terminates.
        """
        # if false, another thread is already stopping the server
        if self._stop_lock.acquire(blocking=False):
            try:
                self._stop(wait=wait)
            except BaseException as e:
                self._stop_lock.release()
                raise e

            self._stop_lock.release()
        # if we are waiting for the stop, we need to wait until the other
        # thread has succeeded and releases the lock
        elif wait:
            self._stop_lock.acquire()
            self._stop_lock.release()
        
    def _perform_transition(self):
        """
        Gets actions from the action collector and performs a step in the
        underlying environment or a reset (if an interrupt has been requested).
        """
        action_dict = self._action_collector.get_actions()
        actions = list(action_dict.values())
        
        if self._action_collector.interrupted:
            # make sure everybody knows that the episode ended
            for agentid in range(self.num_agents):
                # the agents asking for an interrupt, and the agents who
                # have been done before, already know
                if (
                    isinstance(action_dict.get(agentid, None), ResetMessage) or 
                    self._reset_expected[agentid] or
                    self._reset_requested[agentid]
                ):
                    continue

                # build up an observation message, repeating the last observation
                if self._info is None:
                    info = {'interrupted': True}
                else:
                    info = dict(self._info[agentid], interrupted=True)

                obs_msg = ObservationMessage(self._obs[agentid], 0, True, info)
                self.outgoing_messages[agentid].put_nowait(obs_msg)

            self._perform_reset()
            # if an agent is still waiting for a reset,
            # another reset is not expected
            self._reset_expected[np.where(self._reset_requested)] = False

            # prepare messages for everyone whose turn it is next
            for agentid in self._action_collector.agent_turn:
                obs_msg = ObservationMessage(self._obs[agentid])
                # if agent already requested a reset, send the message now
                if self._reset_requested[agentid]:
                    self.outgoing_messages[agentid].put_nowait(obs_msg)
                    self._reset_requested[agentid] = False
                else: # else buffer the message
                    self.obs_msg_buffer[agentid] = obs_msg
                    
        else:
            try:
                obs, rew, done, info = self.multi_agent_env.step(actions)
                
            except BaseException as e:
                for agentid in self._action_collector.agent_turn:
                    self.outgoing_messages[agentid].put_nowait(ErrorMessage(e))
                    
                self._action_collector.reset(self._action_collector.agent_turn)
                return
                
            self._action_collector.reset(self.multi_agent_env.agent_turn)
            self._obs = obs
            self._info = info

            # signal all newly done agents
            newly_done = np.where(~self._reset_expected & done)[0]

            # communicate observations to the agents who are newly done
            for agentid in newly_done:
                obs_msg = ObservationMessage(obs[agentid], rew[agentid],
                                             done[agentid], info[agentid])
                self.outgoing_messages[agentid].put_nowait(obs_msg)

            # keep track of which agents were done and should reset
            self._reset_expected[np.where(done)] = True

            # communicate observations to the agents who are turning now:
            # unless they are in newly_done (those have been signaled already)

            for agentid in set(self._action_collector.agent_turn).difference(newly_done):
                obs_msg = ObservationMessage(obs[agentid], rew[agentid],
                                             done[agentid], info[agentid])

                if self._reset_requested[agentid]:
                    self._reset_requested[agentid] = False
                elif self._reset_expected[agentid]:
                    self.obs_msg_buffer[agentid] = obs_msg
                    continue

                self.outgoing_messages[agentid].put_nowait(obs_msg)

    def _handle_action(self, msg):
        """
        Collects the action and performs a transition if all the necessary
        actions are collected.

        Errors:
        - A RuntimeError ErrorMessage is sent to the agent if an action is
          supplied before the agent has reset.
        """
        if self._reset_expected[msg.agentid]:
            e = RuntimeError("Agent {} did not call reset at the beginning of a new episode.".format(msg.agentid))
            self.outgoing_messages[msg.agentid].put_nowait(ErrorMessage(e))
            return

        # register the action message
        try:
            self._action_collector.collect(msg)
        except ValueError as e:
            self.outgoing_messages[msg.agentid].put_nowait(ErrorMessage(e))
            return

        # if all actions collected, perform a transition
        if self._action_collector.all_collected:
            self._perform_transition()

    def _perform_reset(self):
        """
        Resets the underlying environment and do the necessary book-keeping.
        """        
        self._obs = self.multi_agent_env.reset()
        self._info = None
        self._action_collector.reset(self.multi_agent_env.agent_turn)
        self._reset_expected[:] = True
        np.asarray(self.obs_msg_buffer)[:] = None

    def _send_buffer_msg(self, agentid):
        self.outgoing_messages[agentid].put_nowait(
            self.obs_msg_buffer[agentid]
        )
        self.obs_msg_buffer[agentid] = None

    def _has_buffer_msg(self, agentid):
        return not self.obs_msg_buffer[agentid] is None

    def _handle_reset(self, msg):
        """
        Handles a reset message and takes appropriate steps such as resetting
        the underlying environment, lodging an interrupt request, etc.
        """

        # if all agents are done, reset the env
        if np.all(self._reset_expected):
            # reset the env
            self._perform_reset()
            self._reset_expected[msg.agentid] = False

            # prepare messages for everyone whose turn it is next
            for agentid in self._action_collector.agent_turn:
                obs_msg = ObservationMessage(self._obs[agentid])
                self.obs_msg_buffer[agentid] = obs_msg

            # send a buffered observation if any
            if self._has_buffer_msg(msg.agentid):
                self._send_buffer_msg(msg.agentid)
            # or record a reset request to be redeemed later
            else:
                self._reset_requested[msg.agentid] = True

        # a regular reset after done
        elif self._reset_expected[msg.agentid]:
            self._reset_expected[msg.agentid] = False

            # if there is already an observation for this agent lodged
            # in the observation buffer, send it now
            if self._has_buffer_msg(msg.agentid):
                self._send_buffer_msg(msg.agentid)

            # if there is not, record the request: it is going to be redeemed
            # as soon as it is the agent's turn
            else:
                self._reset_requested[msg.agentid] = True
                
        # lodge an interrupt request
        else:
            self._reset_requested[msg.agentid] = True
            self._handle_action(msg)

    def _handle_error(self, msg):
        raise ErrorException(msg.msg)
        
    def _handle_stop(self, msg):
        raise StopServerException()
        
    def __del__(self):
        self.stop()

def handle_error_stop(client, error):
    client.server.stop()
    raise error

def handle_error_nostop(client, error):
    raise error

class AgentClientEnv(gym.Wrapper):
    def __init__(self, server, agentid, error_handler=handle_error_stop, **kwargs):
        super().__init__(server.multi_agent_env, **kwargs)
        self.agentid = agentid
        self.server = server
        self.error_handler = error_handler
        
        self.observation_space = server.multi_agent_env.observation_spaces[agentid]
        self.action_space = server.multi_agent_env.action_spaces[agentid]
        self.reward_range = server.multi_agent_env.reward_ranges[agentid]

    def reset(self):
        if not self.server.is_running():
            raise StopServerException("Calling reset and the multi agent server is not running.")

        self.server.incoming_messages.put(ResetMessage(self.agentid))
        obs_msg = self.server.outgoing_messages[self.agentid].get()

        if isinstance(obs_msg, ErrorMessage):
            self.error_handler(self, obs_msg.msg)
        elif isinstance(obs_msg, StopServerMessage):
            raise StopServerException("The server has stopped.")
        elif not isinstance(obs_msg, ObservationMessage):
            self.error_handler(self, RuntimeError("The server returned: {}.".format(obs_msg)))

        return obs_msg.observation

    def step(self, action):
        if not self.server.is_running():
            raise StopServerException("Calling step and the multi agent server is not running.")

        self.server.incoming_messages.put(ActionMessage(action, self.agentid))
        obs_msg = self.server.outgoing_messages[self.agentid].get()

        if isinstance(obs_msg, ErrorMessage):
            self.error_handler(self, obs_msg.msg)
        elif isinstance(obs_msg, StopServerMessage):
            raise StopServerException("The server has stopped.")
        elif not isinstance(obs_msg, ObservationMessage):
            self.error_handler(self, RuntimeError("The server returned: {}.".format(obs_msg)))

        return obs_msg.totuple()

    def close(self):
        self.server.stop()

    def __del__(self):
        self.server.stop()

SingleAgentEnvTurnBased = AgentClientEnv
    
def multi_agent_to_single_agent(multi_agent_env):
    server = MultiAgentServer(multi_agent_env)
    server.start()
    
    clients = [AgentClientEnv(server, agentid)
               for agentid in range(server.num_agents)]

    return clients
