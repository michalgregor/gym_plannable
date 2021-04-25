import numbers
from gym_plannable.plannable import PlannableState

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