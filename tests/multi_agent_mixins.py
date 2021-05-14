import collections
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
        if self.env.num_agents != 1:
            self.assertEqual(
                len(self.env.action_space), self.env.num_agents,
                    msg="len(action_space) != num_agents"
            )

            self.assertEqual(
                len(self.env.observation_space), self.env.num_agents,
                    msg="len(observation_space) != num_agents"
            )

            self.assertEqual(
                len(self.env.reward_range), self.env.num_agents,
                    msg="len(reward_range) != num_agents"
            )

    def test_reset(self):
        obs = self.env.reset()

        if self.env.num_agents != 1:
            self.assertEqual(len(obs), self.env.num_agents)

        if self.env.num_agents != 1:
            for io, o in enumerate(obs):
                self.assertTrue(self.env.observation_space[io].contains(o))
        else:
            self.assertTrue(self.env.observation_space.contains(obs))

        agent_turn = self.env.agent_turn
        self.assertIsInstance(agent_turn, collections.Sequence)

    def test_transition(self):
        self.env.reset()
        agent_turn = self.env.agent_turn

        if self.env.num_agents != 1:
            actions = [self.env.action_space[agentid].sample()
                    for agentid in agent_turn]
        else:
            actions = self.env.action_space.sample()
        
        obs, rewards, done, info = self.env.step(actions)

        if self.env.num_agents != 1:
            self.assertEqual(len(obs), self.env.num_agents)

            for io, o in enumerate(obs):
                self.assertTrue(self.env.observation_space[io].contains(o))
        else:
            self.assertTrue(self.env.observation_space.contains(obs))

class ServerTestMixin:
    timeout = 1
    env_constructor = None
    actions = None

    def setUp(self):
        self.multiagent_env = self.env_constructor()
        self.server = MultiAgentServer(self.multiagent_env)
        self.server.start()
        self.assertTrue(self.server._thread.is_alive())

    def tearDown(self):
        self.server.stop()
        self.multiagent_env.close()
        self.assertFalse(self.server._thread.is_alive())

    def test_start_and_stop(self):
        pass
    
    def test_reset(self):
        self.server.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_action_before_reset(self):
        self.server.incoming_messages.put(ActionMessage(self.actions[0], 0))
        msg = self.server.outgoing_messages[0].get(timeout=self.timeout)

        self.assertIsInstance(msg, ErrorMessage)
        self.assertIsInstance(msg.msg, RuntimeError)

    def test_transition(self):
        self.server.incoming_messages.put(ResetMessage(0))
        self.server.incoming_messages.put(ResetMessage(1))

        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ActionMessage(self.actions[0], 0))
        obs_msg = self.server.outgoing_messages[1].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_transition2(self):
        self.server.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)
        self.server.incoming_messages.put(ActionMessage(self.actions[0], 0))

        self.server.incoming_messages.put(ResetMessage(1))
        obs_msg = self.server.outgoing_messages[1].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_double_start(self):
        with self.assertRaises(RuntimeError):
            self.server.start()
       
    def test_consecutive_resets(self):        
        self.server.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ResetMessage(1))

        self.server.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ActionMessage(self.actions[0], 0))
        # collect reset message
        obs_msg = self.server.outgoing_messages[1].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_consecutive_resets2(self):
        self.server.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ResetMessage(1))

        self.server.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ActionMessage(self.actions[0], 0))
        obs_msg = self.server.outgoing_messages[1].get(timeout=self.timeout)

        self.server.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

    def test_consecutive_resets3(self):
        self.server.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ResetMessage(0))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ResetMessage(1))
        self.server.incoming_messages.put(ActionMessage(self.actions[0], 0))
        obs_msg = self.server.outgoing_messages[1].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

        self.server.incoming_messages.put(ActionMessage(self.actions[1], 1))
        obs_msg = self.server.outgoing_messages[0].get(timeout=self.timeout)
        self.assertIsInstance(obs_msg, ObservationMessage)

class ServerDeleteTestMixin:
    env_constructor = None
    actions = None

    def setUp(self):
        self.multiagent_env = self.env_constructor()
        self.server = MultiAgentServer(self.multiagent_env)
        self.server.start()
        self.assertTrue(self.server._thread.is_alive())

    def test_del_stop(self):
        self.stopped = False
        def stop_callback():
            self.stopped = True

        self.server.stop_callback = stop_callback
        del self.server
        gc.collect()
        self.assertTrue(self.stopped)

class ClientTestMixin:
    env_constructor = None
    actions = None

    def setUp(self):
        self.multiagent_env = self.env_constructor()
        self.clients = multi_agent_to_single_agent(self.multiagent_env)

        self.stopped = False
        def stop_callback():
            self.stopped = True

        self.clients[0].server.stop_callback = stop_callback

    def tearDown(self):
        del self.clients
        gc.collect()
        self.assertTrue(self.stopped)

    def testStartStop(self):
        pass

    def testRun(self):
        self.agent0_done = False
        self.agent1_done = False

        def agent0():
            env = weakref.proxy(self.clients[0])
            obs = env.reset()

            for a in self.actions[::2]:
                obs, rew, done, info = env.step(a)
                if done: break

            self.assertTrue(done)
            self.agent0_done = True

        def agent1():
            env = weakref.proxy(self.clients[1])
            obs = env.reset()

            for a in self.actions[1::2]:
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
