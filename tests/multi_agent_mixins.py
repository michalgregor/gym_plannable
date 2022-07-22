from collections.abc import Sequence
from gym_plannable.multi_agent import (
    ErrorMessage, MultiAgentServer, ActionMessage, ResetMessage,
    ObservationMessage, multi_agent_to_single_agent, handle_error_nostop
)

import gc
import weakref
from threading import Thread

class EnvTestMixin:
    env_constructor = None

    def setUp(self):
        self.env = self.env_constructor()

    def tearDown(self):
        self.env.close()

    def test_attributes(self):
        self.assertEqual(
            len(self.env.action_spaces), self.env.num_agents,
                msg="len(action_spaces) != num_agents"
        )

        self.assertEqual(
            len(self.env.observation_spaces), self.env.num_agents,
                msg="len(observation_spaces) != num_agents"
        )

        self.assertEqual(
            len(self.env.reward_ranges), self.env.num_agents,
                msg="len(reward_ranges) != num_agents"
        )

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(len(obs), self.env.num_agents)
        for io, o in enumerate(obs):
            self.assertTrue(
                self.env.observation_spaces[io].contains(o),
                "Returned observation is not contained in the observation space."
            )

        agent_turn = self.env.agent_turn
        self.assertIsInstance(agent_turn, Sequence)

    def test_transition(self):
        self.env.reset()
        agent_turn = self.env.agent_turn       

        actions = [self.env.action_spaces[agentid].sample()
                for agentid in agent_turn]
        
        obs, rewards, done, info = self.env.step(actions)

        self.assertEqual(len(obs), self.env.num_agents)

        for io, o in enumerate(obs):
            self.assertTrue(self.env.observation_spaces[io].contains(o))

class ServerTestMixin:
    timeout = 1
    env_constructor = None
    actions = None

    def setUp(self):
        self.multiagent_env = self.env_constructor()
        self.server = MultiAgentServer(self.multiagent_env)
        self.assertFalse(self.server.csi.finished_event.is_set())
        self.server.start()
        self.assertTrue(self.server.is_running())
        self.assertTrue(self.server.csi.started_event.is_set())

    def tearDown(self):
        self.server.stop()
        self.assertTrue(self.server.csi.finished_event.is_set())
        self.multiagent_env.close()

    def test_start_and_stop(self):
        pass
    
    def test_reset(self):
        self.server.csi.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_action_before_reset(self):
        self.server.csi.incoming_messages.put(ActionMessage(self.actions[0], 0))
        msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)

        self.assertIsInstance(msg, ErrorMessage)
        self.assertIsInstance(msg.msg, RuntimeError)

    def test_transition(self):
        self.server.csi.incoming_messages.put(ResetMessage(0))
        self.server.csi.incoming_messages.put(ResetMessage(1))

        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ActionMessage(self.actions[0], 0))
        obs_msg = self.server.csi.outgoing_messages[1].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_transition2(self):
        self.server.csi.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)
        self.server.csi.incoming_messages.put(ActionMessage(self.actions[0], 0))

        self.server.csi.incoming_messages.put(ResetMessage(1))
        obs_msg = self.server.csi.outgoing_messages[1].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_consecutive_resets(self):        
        self.server.csi.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ResetMessage(1))

        self.server.csi.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ActionMessage(self.actions[0], 0))
        # collect reset message
        obs_msg = self.server.csi.outgoing_messages[1].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_consecutive_resets2(self):
        self.server.csi.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ResetMessage(1))

        self.server.csi.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ActionMessage(self.actions[0], 0))
        obs_msg = self.server.csi.outgoing_messages[1].get(timeout=self.timeout)

        self.server.csi.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_consecutive_resets3(self):
        self.server.csi.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ResetMessage(1))
        self.server.csi.incoming_messages.put(ActionMessage(self.actions[0], 0))
        obs_msg = self.server.csi.outgoing_messages[1].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.csi.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.csi.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

class ServerDeleteTestMixin:
    env_constructor = None
    actions = None

    def setUp(self):
        self.multiagent_env = self.env_constructor()
        self.server = MultiAgentServer(self.multiagent_env)
        self.server.start()
        self.assertTrue(self.server.is_running())

    def test_del_stop(self):
        finished_event = self.server.csi.finished_event
        self.server.stop()
        del self.server
        gc.collect()
        self.assertTrue(finished_event.is_set())

class ClientTestMixin:
    env_constructor = None
    actions = None

    def setUp(self):
        self.multiagent_env = self.env_constructor()
        self.clients, server = multi_agent_to_single_agent(
            self.multiagent_env, return_server=True
        )
        self.finished_event = server.csi.finished_event
        
    def tearDown(self):
        del self.clients
        gc.collect()
        self.assertTrue(self.finished_event.is_set())

    def testStartStop(self):
        pass

    def testRun(self):
        self.agent0_done = False
        self.agent1_done = False

        def agent0():
            env = weakref.proxy(self.clients[0])
            obs = env.reset()
            any_obs_none = obs is None

            for a in self.actions[::2]:
                obs, rew, done, info = env.step(a)
                any_obs_none = any_obs_none or obs is None
                if done: break

            self.assertTrue(done)
            self.assertFalse(any_obs_none)
            self.agent0_done = True

        def agent1():
            env = weakref.proxy(self.clients[1])
            obs = env.reset()
            any_obs_none = obs is None

            for a in self.actions[1::2]:
                obs, rew, done, info = env.step(a)
                any_obs_none = any_obs_none or obs is None
                if done: break

            self.assertTrue(done)
            self.assertFalse(any_obs_none)
            self.agent1_done = True

        thread0 = Thread(target=weakref.proxy(agent0))
        thread1 = Thread(target=weakref.proxy(agent1))

        thread0.start()
        thread1.start()

        thread0.join()
        thread1.join()

        self.assertTrue(self.agent0_done)
        self.assertTrue(self.agent1_done)

class ClientScoreTestMixin(ClientTestMixin):
    scores = None

    def testRun(self):
        super().testRun()
        self.assertEqual(
            list(self.multiagent_env.plannable_state().scores()),
            list(self.scores)
        )

class IllegalActionTestMixin(ClientTestMixin):
    env_constructor = None
    agent0_actions0 = None
    agent0_actions1 = None
    agent1_actions = None

    def testRun(self):
        self.agent0_done = False
        self.agent1_done = False

        def agent0():
            env = weakref.proxy(self.clients[0])
            env.error_handler = handle_error_nostop
            obs = env.reset()

            for a in self.agent0_actions0:
                obs, rew, done, info = env.step(a)

            with self.assertRaises(BaseException):
                a = self.agent0_actions0[-1]
                obs, rew, done, info = env.step(a)

            for a in self.agent0_actions1:
                obs, rew, done, info = env.step(a)
                if done: break

            self.assertTrue(done)
            self.agent0_done = True

        def agent1():
            env = weakref.proxy(self.clients[1])
            obs = env.reset()

            for a in self.agent1_actions:
                obs, rew, done, info = env.step(a)
                if done: break

            self.assertTrue(done)
            self.agent1_done = True

        thread0 = Thread(target=weakref.proxy(agent0))
        thread1 = Thread(target=weakref.proxy(agent1))

        thread0.start()
        thread1.start()

        thread0.join()
        thread1.join()

        self.assertTrue(self.agent0_done)
        self.assertTrue(self.agent1_done)
