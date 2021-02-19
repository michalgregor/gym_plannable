from ..plannable import PlannableStateDeterministic, PlannableEnv
from ..turn_based import TurnBasedEnv, TurnBasedState
from ..agent import BaseAgent
from copy import deepcopy
import numpy as np
import gym

class TicTacToeState(TurnBasedState, PlannableStateDeterministic):
    def __init__(self, size=3, score_tracker=None, **kwargs):
        super().__init__(score_tracker=score_tracker, **kwargs)
        self._num_agents = 2
        self._agent_turn = 0
        self.agent_turn_prev = None
        self.size = size
        self.board = np.full((self.size, self.size), -1, dtype=np.int)
        self.num_empty = self.size**2
        self.winner = []
        self.winning_seq = []
        self._rewards = np.zeros(self._num_agents)

    @property
    def agent_turn(self):
        """
        Returns the numeric index of the agent which is going to move next.
        """
        return self._agent_turn

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def observation(self):
        return deepcopy(self.board)
    
    @property
    def rewards(self):
        return self._rewards

    def _update_rewards(self):
        if len(self.winner) == 0 or None in self.winner:
            self._rewards = np.zeros(self._num_agents)
        else:
            self._rewards = np.full(self._num_agents, -1)
            for agentid in self.winner:
                self._rewards[agentid] = 1

    def legal_actions(self):
        """
        Returns the space of all actions legal at the current step.
        """
        if self.is_done(): return np.array([])
        else: return np.argwhere(self.board == -1)

    def _next(self, action, agentid=None, in_place=False):
        if not in_place:
            state = deepcopy(self)
        else:
            state = self

        x, y = action

        if not agentid is None and agentid != state._agent_turn:
            raise ValueError("It is not agent {}'s turn.".format(agentid))
        
        if(state.is_done()):
            raise ValueError('Illegal action: the game is over.')

        if(state.board[x, y] != -1):
            raise RuntimeError('Illegal action: cell {}, {} is not empty.'.format(x, y))
        
        # perform the move
        state.board[x, y] = state._agent_turn
        state.num_empty -= 1
        
        # check whether the game is over and if so, who has won
        state._check_done(x, y, state._agent_turn)

        # it's the other player's turn next
        state.agent_turn_prev = state._agent_turn
        state._agent_turn = (state._agent_turn + 1) % state._num_agents

        # update the rewards and the scores
        state._update_rewards()

        return state

    def is_done(self):
        """
        Returns whether the game is over.
        """
        return len(self.winner) > 0
        
    def _check_done(self, x, y, agent_turn):
        # Check row x and column y.
        rowseq = []
        colseq = []
        
        for i in range(self.size):
            if self.board[x, i] == agent_turn: rowseq.append([x, i])
            if self.board[i, y] == agent_turn: colseq.append([i, y])

        if len(rowseq) == self.size:
            self.winner = [agent_turn]
            self.winning_seq = rowseq
            return
            
        if len(colseq) == self.size:
            self.winner = [agent_turn]
            self.winning_seq = colseq
            return
            
        # If x and y is at the diagonal, check it.
        if x == y:
            diagseq = []
            
            for i in range(self.size):
                if self.board[i, i] == agent_turn: diagseq.append([i, i])
            
            if len(diagseq) == self.size:
                self.winner = [agent_turn]
                self.winning_seq = diagseq
                return

        # If x, y is at the antidiagonal, check it.
        if x + y == self.size - 1:
            diagseq = []
            
            for i in range(self.size):
                if self.board[i, self.size - 1 - i] == agent_turn:
                    diagseq.append([i, self.size - 1 - i])
            
            if len(diagseq) == self.size:
                self.winner = [agent_turn]
                self.winning_seq = diagseq
                return
        
        # Are all the squares taken and there is no winner?
        if self.num_empty <= 0:
            self.winner = [None]
            return
    
class TicTacToeEnv(TurnBasedEnv, PlannableEnv):
    def __init__(self, size=3, **kwargs):
        """
        The constructor; the board has dimensions size x size.
        """
        state = TicTacToeState(size)
        super().__init__(num_agents=state.num_agents, **kwargs)
        self._state = state

        self.observation_space = [gym.spaces.Box(
            low=np.full((size, size), -1, dtype=np.int),
            high=np.full((size, size), self.num_agents-1, dtype=np.int),
            dtype=np.int
        )] * state.num_agents

        self.action_space = [gym.spaces.Box(
            low=np.asarray((0, 0), dtype=np.int),
            high=np.asarray((size-1, size-1), dtype=np.int),
            dtype=np.int
        )] * state.num_agents

    def plannable_state(self):
        return self._state

    @property
    def agent_turn(self):
        return self._state.agent_turn

    def reset(self):
        self._state = TicTacToeState(self._state.size)
        return self._state.observation
 
    def step(self, action, agentid=None):
        """
        Performs the action and returns the next observation, reward,
        done flag and info dict.
        """
        agentid = agentid or self._state.agent_turn
        self._state.next(action, agentid=agentid, in_place=True)
        rewards = self._state.rewards
        done = self._state.is_done()
        
        return self._state.observation, rewards, done, {
            'winning_seq': self._state.winning_seq,
            'winner': self._state.winner
        }

class Minimax(BaseAgent):
    def select_action(self, state):
        # query for legal actions
        legals = state.legal_actions()

        # go over all legal actions and find the one
        # that maximizes our player's score
        maxval = -np.inf
        maxa = None

        for a in legals:
            next_state = state.next(a)
            val = self.min_value(next_state)

            if(val > maxval):
                maxval = val
                maxa = a

        return maxa

    def min_value(self, state):
        # Ending the recursion: if the game is over, return the score.
        if state.is_done(): return state.scores[self.agentid]

        # query for legal actions
        legals = state.legal_actions()

        # go over all legal actions and find the one that minimizes
        # our player's score (we assume the opponent will want to do that)
        minval = np.inf

        for a in legals:
            next_state = state.next(a)
            val = self.max_value(next_state)

            if(val < minval):
                minval = val

        return minval

    def max_value(self, state):
        # Ending the recursion: if the game is over, return the score.
        if state.is_done(): return state.scores[self.agentid]

        # query for legal actions
        legals = state.legal_actions()

        # go over all legal actions and find the one
        # that maximizes our player's score
        maxval = -np.inf

        for a in legals:
            next_state = state.next(a)
            val = self.min_value(next_state)

            if(val > maxval):
                maxval = val

        return maxval

try:
    from notebook_invoke import register_callback, remove_callback, jupyter_javascript_routines
    from IPython.display import display, HTML, Javascript
    import uuid

    class TicTacToeAgentJavascript:
        def __init__(self, env):
            self.id = uuid.uuid1().hex
            self.env = env

            register_callback(
                'reset_' + self.id, self.reset
            )

            register_callback(
                'step_' + self.id, self.step
            )

            board_height, board_width = self.env.observation_space.shape
            
            display(HTML("""
            <style>
            .ttt_container .row {
                display:inline-block;
            }
            .ttt_container .cell-wrapper {
                margin: 0.2em;
                border: solid 1px;
                text-align: center;
                width: 5em;
                height: 5em;
                vertical-align: middle;
                display: table;
            }

            .ttt_container .cell {
                vertical-align: middle;
                display: table-cell;
                font-size: 200%;
            }

            .ttt_container, .ttt_container .ttt_grid {
                display: inline-block;
            }

            .ttt_container button {
                display: block;
                width: 100%;
                margin-top: 0.5em;
                height: 2.5em;
            }

            .ttt_winning {
                color: green;
            }

            .ttt_losing {
                color: red;
            }

            .ttt_done {
                opacity: 0.5;
            }
            </style>

            <div id="ttt_container_{{UUID_STR}}"></div>
            """.replace("{{UUID_STR}}", self.id)))
            
            display(Javascript(
                jupyter_javascript_routines + ttc_javascript(self.id,
                    board_height, board_width)
            ))

        def reset(self):
            obs = self.env.reset()
            return {'board': obs.tolist()}

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            return {'board': obs.tolist(),
                    'reward': reward,
                    'done': done,
                    'info': info}

        def __del__(self):
            remove_callback('reset_' + self.id)
            remove_callback('step_' + self.id)

    def ttc_javascript(uuid_str, board_height, board_width):
        return ("""
        class TTCGrid {
            constructor(container, grid_height, grid_width) {
                this.grid_height = grid_height;
                this.grid_width = grid_width;
                this.container = container;
                this.board = this.create_board(container, grid_height, grid_width);
                this.symbols = ['', '&#x25EF;', '&#x2715;'];
                this.reset_game();
            }

            reset_game() {
                var self = this;
                invoke_function('reset_{{UUID_STR}}', [], {}).then(
                    data => {return self.update_board(data['board']);}
                );
            }

            update_board(board_contents) {
                for (var i=0; i < this.board.length; i++) {
                    for (var j=0; j < this.board[i].length; j++) {
                        this.board[i][j].innerHTML = this.symbols[
                            board_contents[i][j]+1
                        ];
                    }
                }
            }

            update_done(winner, winning_seq) {
                this.board.grid.classList.add('ttt_done');
                var cls = '';

                if (winner.includes(0)) {
                    cls = 'ttt_winning';
                } else if(winner.length != 0 && winner.length != 2) {
                    cls = 'ttt_losing';
                }

                for (var i=0; i < winning_seq.length; i++) {
                    this.board[
                        winning_seq[i][0]
                    ][
                        winning_seq[i][1]
                    ].classList.add(cls);
                }
            }

            click(event) {
                if (this.done) return;
                var cellContent = event.target || event.srcElement;
                if (cellContent.cell_pos === undefined) return;
                var self = this;
                
                invoke_function('step_{{UUID_STR}}', [cellContent.cell_pos], {}).then(
                    data => {
                        self.update_board(data['board']);
                        self.done = data['done'];
                        if(self.done) self.update_done(
                            data['info']['winner'],
                            data['info']['winning_seq']
                        );
                    }
                );
            }

            create_board(container, grid_height, grid_width) {
                var board = Array.from(Array(grid_height),
                                        () => new Array(grid_width));

                // remove any existing child elements
                while (container.firstChild) {
                    container.removeChild(container.firstChild);
                }

                // add the ttt_container class
                container.classList.add("ttt_container");

                var grid = document.createElement("div");
                grid.classList.add("ttt_grid");
                container.appendChild(grid);
                board.grid = grid;

                // add the new grid
                for (var i=0; i < grid_height; i++) {
                    var row = document.createElement("div");
                    row.classList.add("row");

                    for (var j=0; j < grid_width; j++) {
                        var cell = document.createElement("div");
                        cell.classList.add("cell-wrapper");
                        var cellContent = document.createElement("span");
                        cellContent.classList.add("cell");
                        cellContent.cell_pos = [j, i];
                        board[j][i] = cellContent;
                        cell.appendChild(cellContent);
                        cell.addEventListener("click", (event) => this.click(event));
                        row.appendChild(cell);
                    }

                    grid.appendChild(row);
                }

                // add controls
                var reset_button = document.createElement("button");
                reset_button.addEventListener("click", (event) => this.reset_event(event));
                reset_button.textContent = "New Game";
                container.appendChild(reset_button);

                // the game is not over yet
                this.done = false;

                return board;
            }

            reset_event(event) {
                this.board = this.create_board(this.container,
                                            this.grid_height,
                                            this.grid_width);
                this.reset_game();
            }
        }

        (function() {
            var container = document.querySelector("#ttt_container_{{UUID_STR}}");
            var ttcGrid = new TTCGrid(container,
                {{board_height}}, {{board_width}});
        })();
        """).replace(
            '{{UUID_STR}}', uuid_str
        ).replace(
            '{{board_height}}', str(board_height)
        ).replace(
            '{{board_width}}', str(board_width)
        )

except ModuleNotFoundError:
    pass
