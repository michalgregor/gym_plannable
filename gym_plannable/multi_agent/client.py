import gym
from .common import (
    StopServerException, ResetMessage, ObservationMessage,
    StopServerMessage, ErrorMessage, ActionMessage
)

def handle_error_stop(client, error):
    client.csi.stop()
    raise error

def handle_error_nostop(client, error):
    raise error

class AgentClientEnv(gym.Wrapper):
    def __init__(self, env_proxy,
        client_server_interface, agentid,
        ignore_multiple_reset=False,
        error_handler=handle_error_stop
    ):
        super().__init__(env_proxy)
        self.agentid = agentid
        self.csi = client_server_interface

        self.error_handler = error_handler        
        self.observation_space = self.csi.observation_spaces[agentid]
        self.action_space = self.csi.action_spaces[agentid]
        self.reward_range = self.csi.reward_ranges[agentid]

        self._ignore_multiple_reset = ignore_multiple_reset
        self._action_taken = True
        self._reset_obs = None

    def reset(self):
        if not self.csi.started_event.is_set() or self.csi.finished_event.is_set():
            raise StopServerException("Calling reset and the multi agent server is not running.")

        if self._ignore_multiple_reset and not self._action_taken:
            return self._reset_obs

        self._action_taken = False
        self.csi.incoming_messages.put(ResetMessage(self.agentid))
        obs_msg = self.csi.outgoing_messages[self.agentid].get()

        if isinstance(obs_msg, ErrorMessage):
            self.error_handler(self, obs_msg.msg)
        elif isinstance(obs_msg, StopServerMessage):
            raise StopServerException("The server has stopped.")
        elif not isinstance(obs_msg, ObservationMessage):
            self.error_handler(self, RuntimeError("The server returned: {}.".format(obs_msg)))

        if self._ignore_multiple_reset:
            self._reset_obs = obs_msg.observation

        return obs_msg.observation

    def step(self, action):
        if not self.csi.started_event.is_set() or self.csi.finished_event.is_set():
            raise StopServerException("Calling step and the multi agent server is not running.")

        self.csi.incoming_messages.put(ActionMessage(action, self.agentid))
        obs_msg = self.csi.outgoing_messages[self.agentid].get()

        if isinstance(obs_msg, ErrorMessage):
            self.error_handler(self, obs_msg.msg)
        elif isinstance(obs_msg, StopServerMessage):
            raise StopServerException("The server has stopped.")
        elif not isinstance(obs_msg, ObservationMessage):
            self.error_handler(self, RuntimeError("The server returned: {}.".format(obs_msg)))

        self._action_taken = True
        return obs_msg.totuple()

    def close(self):
        self.csi.stop()

    def __del__(self):
        self.csi.stop()

SingleAgentEnvTurnBased = AgentClientEnv
