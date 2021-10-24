from queue import Empty
import threading
from .common import (
    ClientServerInterface, ActionCollector, ActionMessage, ResetMessage,
    ErrorMessage, StopServerMessage, StopServerException, ObservationMessage,
    ErrorException
)
import numpy as np

class MultiAgentServer:
    def __init__(self, multi_agent_env):
        """
        A server that manages a multi agent environment, to which several
        clients may connect and present single-agent views of the environment
        to agents that support the OpenAI Gym interface.

        Arguments:
        - multi_agent_env: The multi agent environment (with a MultiAgentEnv
                           interface) that is to be managed by the server.
        """
        self.multi_agent_env = multi_agent_env
        
        self.csi = ClientServerInterface(
            self.num_agents, multi_agent_env.observation_spaces,
            multi_agent_env.action_spaces, multi_agent_env.reward_ranges
        )

        self.obs_msg_buffer = [None for _ in range(self.num_agents)]
        self._reset_expected = np.ones(self.multi_agent_env.num_agents, dtype=np.bool)
        self._reset_requested = np.zeros(self.multi_agent_env.num_agents, dtype=np.bool)

        self._obs = None
        self._info = None

        self._action_collector = ActionCollector()        
        self._message_handlers = {
            ActionMessage: self._handle_action,
            ResetMessage: self._handle_reset,
            ErrorMessage: self._handle_error,
            StopServerMessage: self._handle_stop,
        }

        self._thread = None
        
    def stop(self, wait=True):
        self.csi.stop(wait=wait)

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
        return self.csi.started_event.is_set() and not self.csi.finished_event.is_set()

    def start(self, wait=True):
        """
        Starts the server in a separate thread. Note: each server must
        only be started once; calling start() again while the finished_event
        is still clear will result in undefined behavior.

        Arguments:
        - wait: If true, the call blocks until the server starts up.
        """
        threading.Thread(
            target=type(self).run,
            args=(self,)
        ).start()

        if wait:
            self.csi.started_event.wait()

    def run(self):
        """
        The main server loop: meant to be run in a separate thread
        by calling start().
        """
        self.csi.finished_event.clear()

        # make sure the message queues are clear; incoming
        try:
            while True: self.csi.incoming_messages.get_nowait()
        except Empty:
            pass

        # make sure the message queues are clear; outgoing
        for q in self.csi.outgoing_messages:
            try:
                while True: q.get_nowait()
            except Empty:
                pass

        self.csi.started_event.set()

        try:
            while True:
                msg = self.csi.incoming_messages.get()
                
                for msg_type, handler in self._message_handlers.items():
                    if isinstance(msg, msg_type):
                        handler(msg)
                        break
                else:
                    raise ValueError("Unexpected message type '{}'.".format(type(msg)))
        except StopServerException:
            pass
        except ReferenceError:
            pass

        self.csi.finished_event.set()
        self.csi.started_event.clear()
        self.csi.stop_lock.release()

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
                self.csi.outgoing_messages[agentid].put_nowait(obs_msg)

            self._perform_reset()
            # if an agent is still waiting for a reset,
            # another reset is not expected
            self._reset_expected[np.where(self._reset_requested)] = False

            # prepare messages for everyone whose turn it is next
            for agentid in self._action_collector.agent_turn:
                obs_msg = ObservationMessage(self._obs[agentid])
                # if agent already requested a reset, send the message now
                if self._reset_requested[agentid]:
                    self.csi.outgoing_messages[agentid].put_nowait(obs_msg)
                    self._reset_requested[agentid] = False
                else: # else buffer the message
                    self.obs_msg_buffer[agentid] = obs_msg
                    
        else:
            try:
                obs, rew, done, info = self.multi_agent_env.step(actions)
                
            except BaseException as e:
                for agentid in self._action_collector.agent_turn:
                    self.csi.outgoing_messages[agentid].put_nowait(ErrorMessage(e))
                    
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
                self.csi.outgoing_messages[agentid].put_nowait(obs_msg)

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

                self.csi.outgoing_messages[agentid].put_nowait(obs_msg)

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
            self.csi.outgoing_messages[msg.agentid].put_nowait(ErrorMessage(e))
            return

        # register the action message
        try:
            self._action_collector.collect(msg)
        except ValueError as e:
            self.csi.outgoing_messages[msg.agentid].put_nowait(ErrorMessage(e))
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
        self.csi.outgoing_messages[agentid].put_nowait(
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
