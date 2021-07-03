import gc
import numbers
from gym_plannable.plannable import PlannableState
from gym_plannable.agent import LegalAgent
from gym_plannable.multi_agent import (
    multi_agent_to_single_agent
)

class PlannableInterfaceTestMixin:
    env_constructor = None
    max_next = 5

    def setUp(self):
        self.plannable_env = self.env_constructor()

    def tearDown(self):
        del self.plannable_env

    def testHasPlannableState(self):
        state = self.plannable_env.plannable_state()
        self.assertIsInstance(state, PlannableState)

    def testStateInterface(self):
        state = self.plannable_env.plannable_state()
        num_agents = self.plannable_env.num_agents

        self.assertEqual(state.num_agents, num_agents)
        self.assertTrue(len(state.agent_turn) <= num_agents,
                        msg="len(agent_turn) <= num_agents")

        self.assertEqual(
            len(state.legal_actions()), num_agents,
            msg="len(state.legal_actions()) != num_agents"
        )

        self.assertEqual(
            len(state.observations()), num_agents,
            msg="len(state.observations()) != num_agents"
        )

        self.assertEqual(
            len(state.rewards()), num_agents,
            msg="len(state.rewards()) != num_agents"
        )

        self.assertEqual(
            len(state.is_done()), num_agents,
            msg="len(state.is_done()) != num_agents"
        )       

        self.assertEqual(
            len(state.scores()), num_agents,
            msg="len(state.scores()) != num_agents"
        )

        self.assertEqual(
            len(state.observation_space), state.num_agents,
            msg="The state's observation_space needs to be a sequence with an item for each agent."
        )

        self.assertEqual(
            len(state.action_space), state.num_agents,
            msg="The state's action_space needs to be a sequence with an item for each agent."
        )

        self.assertEqual(
            len(state.reward_range), state.num_agents,
            msg="The state's reward_range needs to be a sequence with an item for each agent."
        )

    def testInit(self):
        state = self.plannable_env.plannable_state()
        state = state.init()
        self.assertIsInstance(state, PlannableState)

        for istate, (state, prob) in enumerate(state.all_init()):
            if istate >= self.max_next: break
            self.assertIsInstance(state, PlannableState)
            self.assertIsInstance(prob, numbers.Number)        
    
    def testNext(self):
        self.plannable_env.reset()
        state = self.plannable_env.plannable_state()
        legals = state.legal_actions()
        actions = [legals[a][0] for a in state.agent_turn]
        next_state = state.next(actions)
        self.assertIsInstance(next_state, PlannableState)

        for istate, (state, prob) in enumerate(state.all_next(actions)):
            if istate >= self.max_next: break
            self.assertIsInstance(state, PlannableState)
            self.assertIsInstance(prob, numbers.Number)

    def testIsDoneNotBool(self):
        # to avoid silent errors, is_done is supposed to return 
        # a sequence that is not convertible to bool
        state = self.plannable_env.plannable_state()
        is_done = state.is_done()

        if self.plannable_env.num_agents > 1:        
            with self.assertRaises(ValueError):
                if is_done:
                    pass

class PlannableEnvTestMixin:
    env_constructor = None

    def setUp(self):
        self.env = self.env_constructor()
        self.clients = multi_agent_to_single_agent(self.env)

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
        agents = [LegalAgent(env, num_episodes=1, max_steps=5) for env in self.clients]

        for a in agents:
            a.start()

        for a in agents:
            a.join()