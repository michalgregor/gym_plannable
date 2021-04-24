import abc
import gym
import numpy as np
import itertools
import matplotlib.pyplot as plt
from enum import Enum
import random
import collections
from copy import deepcopy
import textwrap

from .render import MplFigEnv
from ..plannable import PlannableEnv, PlannableState
from ..turn_based import TurnBasedEnv, TurnBasedState

def tt_parse_grid_str(gridstr):
    lines = [list(line) for line in gridstr.splitlines() if len(line.strip())]
    return np.array(lines, dtype=np.unicode_)

class WorldObject:
    def __init__(self, name):
        self.name = name
        self.world = None
    
    @abc.abstractmethod
    def render(self, ax):
        pass
    
    @abc.abstractmethod
    def channel(self):
        pass
    
    @abc.abstractmethod
    def where(self):
        pass
    
    @abc.abstractmethod
    def occupies(self, pos):
        pass
    
class TransitionObject(WorldObject):
    @abc.abstractmethod
    def init(self, **params):
        """
        Initializes world's state; the function can be stochastic if
        params is empty, but given params, it must behave deterministically.
        """
    
    @abc.abstractmethod
    def all_init(self):
        """
        Returns a generator of (params, probability) tuples, for each possible
        initial state and its corresponding probability; the corresponding
        states can be generated by calling init with the specified params.
        """
    
    @abc.abstractmethod
    def next(self, **params):
        """
        Transitions the world into the next state. The function can be stochastic
        if params is empty, but given params, it must behave deterministically.
        """
    
    @abc.abstractmethod
    def all_next(self):
        """
        Returns a generator of (params, probability) tuples, for each possible
        next state and its corresponding probability; the corresponding
        states can be generated by calling next with the specified params.
        """

class Actor(WorldObject):
    """
    Defines an OpenAI Gym action_space as an attribute.
    """
    
    @abc.abstractmethod
    def init(self, **params):
        """
        Initializes world's state; the function can be stochastic if
        params is empty, but given params, it must behave deterministically.
        """
    
    @abc.abstractmethod
    def all_init(self, world):
        """
        Returns a generator of (params, probability) tuples, for each possible
        initial state and its corresponding probability; the corresponding
        states can be generated by calling init with the specified params.
        """
    
    @abc.abstractmethod
    def legal_actions(self):
        """
        Returns a list of legal actions.
        """
    
    @abc.abstractmethod
    def next(self, action, **params):
        """
        Transitions the world into the next state. The function can be stochastic
        if params is empty, but given params, it must behave deterministically.
        """
    
    @abc.abstractmethod
    def all_next(self, action):
        """
        Returns a generator of (params, probability) tuples, for each possible
        next state and its corresponding probability; the corresponding
        states can be generated by calling next with the specified params.
        """

    @property
    @abc.abstractmethod
    def reward(self):
        """
        Returns the agent's reward.
        """
    
    @reward.setter
    @abc.abstractmethod
    def reward(self, val):
        """
        Sets the agent's reward.
        """
        
class RenderObject(WorldObject):
    def __init__(self, name, grid_shape, edgecolor='gray',
                 facecolor='white', tile_size=1,
                 label=None, fontcolor='black',
                 fontsize=10, patch_func=None, **kwargs):
        super().__init__(name, **kwargs)
        self.grid_shape = grid_shape
        self.edgecolor = edgecolor if not edgecolor is None else (0, 0, 0, 0)
        self.facecolor = facecolor if not facecolor is None else (0, 0, 0, 0)
        self.tile_size = tile_size
        self.label = label
        self.fontcolor = fontcolor if not fontcolor is None else (0, 0, 0, 0)
        self.fontsize = fontsize
        self.patch_func = patch_func or RenderObject._rectangle
    
    @staticmethod
    def _rectangle(c, r, tile_size, facecolor, edgecolor):
        return plt.Rectangle(
            [c - tile_size / 2,
             r - tile_size / 2],
            tile_size, tile_size,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
    
    @staticmethod
    def _circle(c, r, tile_size, facecolor, edgecolor):
        return plt.Circle(
            [c, r], tile_size / 4,
            facecolor=facecolor,
            edgecolor=edgecolor
        )
    
    def render(self, ax):    
        for r, c in self.where():
            patch = self.patch_func(
                c, r, self.tile_size,
                self.facecolor, self.edgecolor
            )
            ax.add_patch(patch)
            
            if not self.label is None:
                ax.text(
                    c, r, self.label,
                    color=self.fontcolor,
                    fontsize=self.fontsize,
                    horizontalalignment='center',
                    verticalalignment='center'
                )

class BackgroundObject(RenderObject):
    def __init__(self, name, grid, **kwargs):
        super().__init__(name, grid.shape, **kwargs)
    
    def channel(self):
        return np.ones(self.grid_shape)
    
    def where(self):
        return itertools.product(range(self.grid_shape[0]),
                                 range(self.grid_shape[1]))
    
    def occupies(self, pos):
        True
        
class DrapeObject(RenderObject):
    def __init__(self, name, grid, symbol, **kwargs):
        super().__init__(name, grid.shape, **kwargs)
        self.symbol = symbol
        self._channel = (grid == self.symbol)
    
    def channel(self):
        return self._channel
    
    def where(self):
        return zip(*np.where(self._channel))
    
    def occupies(self, pos):
        return self._channel[pos[0], pos[1]]

class GoalDrape(TransitionObject, DrapeObject):
    def __init__(self, name, grid, symbol='G', player_names=None,
                 goal_reward=0, **kwargs):
        super().__init__(name, grid, symbol, **kwargs)
        self.player_names = player_names or []
        self.goal_reward = goal_reward
            
    def init(self, **params):
        pass
    
    def all_init(self):
        return (({}, 1.0) for i in range(1))
        
    def next(self, **params):
        for pn in self.player_names:
            p = getattr(self.world, pn)
            if self.occupies(p.position):
                p.reward += self.goal_reward
                self.world.done = True
    
    def all_next(self):
        return (({}, 1.0) for i in range(1))

class GWAction(Enum):
    up = 0
    down = 1
    left = 2
    right = 3

class PositionActor(Actor, RenderObject):
    def __init__(self, name, grid, symbol, walls_name=None, **kwargs):
        render_kwargs = dict(patch_func=RenderObject._circle)
        render_kwargs.update(**kwargs)
        super().__init__(name, grid.shape, **render_kwargs)
        
        self._position = None
        self.starting_poses = np.dstack(np.where(grid == symbol))[0]
        if not len(self.starting_poses):
            raise ValueError("No starting position with symbol '{}' found on the grid.".format(symbol))
        
        self.walls_name = walls_name
        self._reward = 0
        self.action_space = gym.spaces.Discrete(4)
        
    @property
    def reward(self):
        """
        Returns the agent's reward.
        """
        return self._reward
    
    @reward.setter
    def reward(self, val):
        """
        Sets the agent's reward.
        """
        self._reward = val
        
    @property
    def position(self):
        return self._position
        
    def init(self, pos=None):        
        self._reward = 0
        if not pos is None:
            self._position = pos
        else:
            self._position = random.choice(self.starting_poses)
        
    def all_init(self):
        prob = 1 / len(self.starting_poses)
        for pos in self.starting_poses:                
            yield (dict(pos=pos), prob)

    def where(self):
        return np.array([self._position])

    def channel(self):
        channel = np.zeros(self.grid_shape, dtype=np.bool)
        channel[self._position[0], self._position[1]] = True
        return channel
    
    def occupies(self, pos):
        return pos == self._position
        
    def legal_actions(self):
        """
        Returns a list of legal actions.
        """
        return [GWAction.up, GWAction.down, GWAction.left, GWAction.right]        
    
    def next(self, action, **params):
        self._reward = 0
        
        if action == GWAction.up:
            newpos = (self._position[0]-1, self._position[1])
        elif action == GWAction.down:
            newpos = (self._position[0]+1, self._position[1])
        elif action == GWAction.left:
            newpos = (self._position[0], self._position[1]-1)
        elif action == GWAction.right:
            newpos = (self._position[0], self._position[1]+1)
        else:
            raise ValueError("Unknown action '{}'.".format(action))
    
        newpos = (
            max(min(newpos[0], self.grid_shape[0]-1), 0),
            max(min(newpos[1], self.grid_shape[1]-1), 0)
        )
        
        if (self.walls_name is None or
            not getattr(self.world, self.walls_name).occupies(newpos)
        ):
            self._position = newpos
    
    def all_next(self, action):
        # agent's movement is deterministic in this implementation
        return (({}, 1.0) for i in range(1))

class StochasticActor(PositionActor):
    def next(self, action, rand_act=None):
        self._reward = 0
        
        rand_act = np.random.randint(0, 4) if rand_act is None else rand_act
        
        if rand_act == 0:
            newpos = (self._position[0]-1, self._position[1])
        elif rand_act == 1:
            newpos = (self._position[0]+1, self._position[1])
        elif rand_act == 2:
            newpos = (self._position[0], self._position[1]-1)
        elif rand_act == 3:
            newpos = (self._position[0], self._position[1]+1)
        else:
            raise ValueError("Unknown action '{}'.".format(rand_act))
    
        newpos = (
            max(min(newpos[0], self.grid_shape[0]-1), 0),
            max(min(newpos[1], self.grid_shape[1]-1), 0)
        )
        
        if (self.walls_name is None or
            not getattr(self.world, self.walls_name).occupies(newpos)
        ):
            self._position = newpos
    
    def all_next(self, action):
        return (({'rand_act': i}, 0.25) for i in range(4))

class ObservationFunction:
    """
    Defines OpenAI Gym observation_space as an attribute.
    """

class PosObservation(ObservationFunction):
    def __init__(self, grid_shape, pos_agent_name="player"):
        self.pos_agent_name = pos_agent_name
        self.observation_space = gym.spaces.Box(
            low=np.zeros(2),
            high=np.asarray(grid_shape),
            dtype=np.int
        )       
        
    def __call__(self, state):
        return getattr(state, self.pos_agent_name).position
        
class MatrixObservation(ObservationFunction):
    def __init__(self, grid_shape, num_layers, observation_sequence=None):
        mat_shape = tuple(grid_shape) + (num_layers,)
        self.observation_sequence = observation_sequence
        self.observation_space = gym.spaces.Box(
            low=np.zeros(mat_shape, dtype=np.bool),
            high=np.ones(mat_shape, dtype=np.bool),
            dtype=np.int
        )
        
    def __call__(self, state):
        seq = self.observation_sequence or state.render_sequence
        obs = np.dstack([obj.channel() for obj in seq])
        return obs

class WorldState(TurnBasedState, PlannableState):
    def __init__(self,
                 grid_shape,
                 transition_sequence,
                 render_sequence,
                 observation_function=None,
                 score_tracker=None, **kwargs):
        """
        Arguments:
        * transition_sequence: A sequence of TransitionObject or Actor
                               objects in the order in which their
                               transition functions (next, all_next are
                               to be called).
        * render_sequence: A sequence of WorldObject objects in the
                           order in which they are to be rendered.
        """
        super().__init__(score_tracker=score_tracker, **kwargs)
        self.transition_sequence = transition_sequence
        self.render_sequence = render_sequence
        self.grid_shape = grid_shape
        self.actors = []
        self.done = False
        self._tr_reset()

        for obj in self.transition_sequence:
            setattr(self, obj.name, obj)
            obj.world = self
            
            if isinstance(obj, Actor):
                self.actors.append(obj)
                
        if not len(self.actors):
            raise ValueError("The transition_sequence must contain at least one actor.")

        for obj in self.render_sequence:
            setattr(self, obj.name, obj)
            obj.world = self
        
        # set up the observation functions
        self.observation_function = observation_function or MatrixObservation(
            self.grid_shape, len(self.render_sequence)
        )
        
        if isinstance(self.observation_function, collections.abc.Sequence):
            if not len(self.observation_function) == len(self.actors):
                raise ValueError("The number of observation functions must equal the number of actors.")
        else:
            self.observation_function = [self.observation_function] * len(self.actors)
    
    def _tr_obj(self):
        """
        Returns the current transition object. The sequence wraps
        around: once the objects are exhausted, iteration starts
        again from the beginning.
        """
        return self.transition_sequence[self.transition_step % len(self.transition_sequence)]
    
    def _next_tr_obj(self):
        """
        Increments the transition step to point to the next object.
        """
        if isinstance(self._tr_obj(), Actor):
            self.actor_step = (self.actor_step + 1) % len(self.actors)
        self.transition_step += 1
        
    def _is_tr_end(self):
        """
        Returns whether this is the last step in the transition sequence.
        """
        return self.transition_step >= len(self.transition_sequence)
    
    def _tr_reset(self):
        self.transition_step = 0
        self.actor_step = 0
    
    @property
    def num_agents(self):
        """
        Returns the number of agents in the environment.
        """
        return len(self.actors)
    
    @property
    def agent_turn(self):
        """
        Returns the numeric index of the agent which is going to move next.
        """
        return self.actor_step

    def copy(self):
        return deepcopy(self)
    
    def render(self, fig):
        ax = fig.gca()
        ax.clear()
        
        for p in self.render_sequence:
            p.render(ax)
            
        # Scale the axes, invert y.
        ax.set_aspect('equal')
        ax.autoscale_view(tight=True)
        if(ax.get_ylim()[0] < ax.get_ylim()[1]): ax.invert_yaxis()

    def init(self, inplace=False):
        """
        Returns an initial state.
        
        Arguments:
        * inplace: If True, the state is not copied; the changes
                   are done in place.
        """
        w = self if inplace else self.copy()
        w._tr_reset()
        
        for obj in w.transition_sequence:
            obj.init()
            
        # also run next until the first
        # actor is encountered
        for obj in w.transition_sequence:
            if isinstance(obj, Actor): break
            obj.next()
            w._next_tr_obj()
            
        w.done = False
        return w
    
    def _next(self, action, inplace=False):
        w = self if inplace else self.copy()
        
        while not isinstance(w._tr_obj(), Actor):
            w._tr_obj().next()
            w._next_tr_obj()
            
        w._tr_obj().next(action)
        w._next_tr_obj()
        
        while not isinstance(w._tr_obj(), Actor):
            w._tr_obj().next()
            w._next_tr_obj()
        
        return w

    @staticmethod
    def _unfold(states, gen_method, transition_method, stop_criterion):
        for state, state_prob in states:
            if stop_criterion(state, 0):
                yield state, state_prob
                continue

            obj = state._tr_obj()
            gen = gen_method(obj)
        
            gen_queue = collections.deque() # (state, params generator, probability)
            gen_queue.append((state, gen, state_prob))
        
            while len(gen_queue):
                root, gen, root_prob = gen_queue[-1]
                
                try:
                    params, prob = next(gen)
                    s = root.copy()
                    transition_method(s._tr_obj(), **params)
                    s._next_tr_obj()
                    
                    if stop_criterion(s, s.transition_step - state.transition_step):
                        yield s, root_prob*prob
                    else:
                        gen = gen_method(s._tr_obj())
                        gen_queue.append((s, gen, root_prob*prob))
                except StopIteration:
                    gen_queue.pop()
    
    def all_init(self):
        """
        Returns a generator of (state, probability) tuples for all possible
        initial states.
        """
        gen_method = lambda obj: obj.all_init()
        transition_method = lambda obj, **params: obj.init(**params)
        stop_criterion = lambda state, steps: state._is_tr_end()
        
        init_gen = self._unfold([(self, 1.0)], gen_method, transition_method, stop_criterion)
        
        gen_method = lambda obj: obj.all_next()
        transition_method = lambda obj, **params: obj.next(**params)
        stop_criterion = lambda state, steps: isinstance(state._tr_obj(), Actor)
        
        next_gen = self._unfold(init_gen, gen_method, transition_method, stop_criterion)
        
        for state in next_gen:
            state.done = False
            yield state

    def _all_next(self, action):
        """
        Returns a generator of (state, probability) tuples for all possible
        next states.
        """
        
        # unfold until an agent is encountered
        gen_method = lambda obj: obj.all_next()
        transition_method = lambda obj, **params: obj.next(**params)
        stop_criterion = lambda state, steps: isinstance(state._tr_obj(), Actor)
        
        pre_agent_gen = self._unfold([(self, 1.0)], gen_method, transition_method, stop_criterion)
        
        # apply the action and unfold on that
        gen_method = lambda obj: obj.all_next(action)
        transition_method = lambda obj, **params: obj.next(action, **params)
        stop_criterion = lambda state, steps: steps > 0
        
        agent_gen = self._unfold(pre_agent_gen, gen_method, transition_method, stop_criterion)
        
        # keep unfolding until the next agent is encountered, then stop
        gen_method = lambda obj: obj.all_next()
        transition_method = lambda obj, **params: obj.next(**params)
        stop_criterion = lambda state, steps: isinstance(state._tr_obj(), Actor)
    
        post_agent_gen = self._unfold(agent_gen, gen_method, transition_method, stop_criterion)
        
        return post_agent_gen

    def legal_actions(self):
        """
        Returns the sequence of all actions that are legal in the state.
        """
        return self.actors[self.actor_step].legal_actions()
        
    def is_done(self):
        """
        Returns whether this is a terminal state or not.
        """
        return self.done

    @property
    def rewards(self):
        """
        Returns a sequence containing the rewards.
        """
        return np.asarray([obj.reward for obj in self.actors])

    def observation(self):
        """
        Returns the observation associated with the state.
        """        
        return self.observation_function[self.actor_step](self)

class GridWorldEnv(PlannableEnv, TurnBasedEnv, MplFigEnv):
    def __init__(
        self,
        grid_shape,
        transition_sequence,
        render_sequence,
        observation_function=None,
        **kwargs
    ):
        self._state = WorldState(
            grid_shape,
            transition_sequence,
            render_sequence,
            observation_function=observation_function
        )
        
        super().__init__(num_agents=self._state.num_agents, **kwargs)
        
        self.observation_space = [
            obs_func.observation_space 
                for obs_func in self._state.observation_function
        ]

        self.action_space = [
            actor.action_space
                for actor in self._state.actors
        ]

    @property
    def agent_turn(self):
        """
        Returns the numeric index of the agent which is going to move next.
        """
        return self._state.agent_turn
    
    def _render(self, fig):
        return self._state.render(fig)
    
    def plannable_state(self):
        return self._state
    
    def reset(self):
        self._state = self._state.init()
        return self._state.observation()
    
    def step(self, action, agentid=None):
        """
        Applies the specified action and activates the state transition.
        If agentid is specified and does not match self.agent_turn,
        a ValueError is raised (useful as a consistency check).

        This returns what a single-agent step method would return except that
        instead of a single reward, there will be a list of rewards, one
        for each agent.

        Arguments:
          * action: The action to apply.
          * agentid: The id of the agent that selected the action. If None,
            self.agent_turn will be used instead.
        """       
        agentid = agentid or self._state.agent_turn
        if agentid != self._state.agent_turn:
            raise ValueError("It is not the agent's turn: '{}'.".format(agentid))
        
        self._state.next(action, inplace=True)
        rewards = self._state.rewards
        if len(rewards) == 1: rewards = rewards[0]

        return self._state.observation(), rewards, self._state.is_done(), {}

class MazeEnv(GridWorldEnv):
    def __init__(self, grid=None, observation_function=None, **kwargs):
        if grid is None:
            grid = textwrap.dedent(
            """
            XXXX00000G
            000X000000
            000X000XXX
            000X000000
            000X000000
            0000000000
            0000X00000
            0000XXX00X
            0000X00000
            S000X00000
            """)
        
        grid_ar = tt_parse_grid_str(grid)
                
        goal = GoalDrape("goal", grid_ar, 'G', facecolor=None,
                         label='G', player_names=["player"],
                         goal_reward=100)
        player = PositionActor("player", grid_ar, 'S', 'walls',
                               facecolor='blue', edgecolor=None)
        
        transition_sequence = [
            player,
            goal
        ]
        
        render_sequence = [
            BackgroundObject("background", grid_ar),
            DrapeObject("walls", grid_ar, 'X', facecolor='gray'),
            DrapeObject("starts", grid_ar, 'S', facecolor=None, label='S'),
            goal,
            player
        ]
        
        if observation_function is None:
            observation_function = [PosObservation(grid_ar.shape,
                                                   pos_agent_name="player")]
        
        super().__init__(
            grid_ar.shape,
            transition_sequence=transition_sequence,
            render_sequence=render_sequence,
            observation_function=observation_function,
            **kwargs
        )