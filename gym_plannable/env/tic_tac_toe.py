from ..plannable import PlannableStateDeterministic, PlannableEnv
from ..multi_agent import StopServerException, handle_error_nostop, EnvInterface2SingleMixin
from copy import deepcopy
import numpy as np
import traceback
import gym

class TicTacToeState(PlannableStateDeterministic):
    def __init__(self, 
        observation_space, action_space, reward_range,
        size=3, score_tracker=None, **kwargs
    ):
        super().__init__(
            observation_space, action_space, reward_range,
            score_tracker=score_tracker, **kwargs
        )
        self.size = size
        self.init(inplace=True)

    def _init(self, inplace=False):
        if not inplace:
            state = deepcopy(self)
        else:
            state = self

        state._num_agents = 2
        state._agent_turn = 0
        state.agent_turn_prev = None
        state.board = np.full((state.size, state.size), -1, dtype=np.int)
        state.num_empty = state.size**2
        state.winner = []
        state.winning_seq = []
        state._reward_ar = np.zeros(state._num_agents)

        return state

    @property
    def agent_turn(self):
        """
        Returns the numeric index of the agent which is going to move next.
        """
        return [self._agent_turn]

    @property
    def num_agents(self):
        return self._num_agents

    def _observations(self):
        return [deepcopy(self.board)] * self.num_agents
    
    def _rewards(self):
        return self._reward_ar

    def _update_rewards(self):
        if len(self.winner) == 0 or None in self.winner:
            self._reward_ar = np.zeros(self._num_agents)
        else:
            self._reward_ar = np.full(self._num_agents, -1)
            for agentid in self.winner:
                self._reward_ar[agentid] = 1

    def _legal_actions(self):
        """
        Returns the space of all actions legal at the current step for each agent.
        """
        if np.all(self.is_done()): legals = np.array([])
        else: legals = np.argwhere(self.board == -1)
        return [legals] * self.num_agents

    def _next(self, actions, agentid=None, inplace=False):
        if not inplace:
            state = deepcopy(self)
        else:
            state = self

        assert len(actions) == 1
        x, y = actions[0]

        if not agentid is None and agentid != state.agent_turn:
            raise ValueError("It is not agent {}'s turn.".format(agentid))
        
        if(np.all(state.is_done())):
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

    def _is_done(self):
        """
        Returns whether the game is over.
        """
        done = len(self.winner) > 0
        return [done for i in range(self.num_agents)]
        
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
    
class TicTacToeEnv(EnvInterface2SingleMixin, PlannableEnv):
    def __init__(self, size=3, **kwargs):
        """
        The constructor; the board has dimensions size x size.
        """
        num_agents = 2
        super().__init__(num_agents=num_agents, **kwargs)
        
        self.observation_space = [gym.spaces.Box(
            low=np.full((size, size), -1, dtype=np.int),
            high=np.full((size, size), num_agents-1, dtype=np.int),
            dtype=np.int
        )] * num_agents

        self.action_space = [gym.spaces.Box(
            low=np.asarray((0, 0), dtype=np.int),
            high=np.asarray((size-1, size-1), dtype=np.int),
            dtype=np.int
        )] * num_agents

        self._state = TicTacToeState(
            self.observation_space,
            self.action_space,
            self.reward_range,
            size=size
        )

    def plannable_state(self):
        return self._state

    @property
    def agent_turn(self):
        return self._state.agent_turn

    def reset(self):
        self._state = TicTacToeState(
            self.observation_space,
            self.action_space,
            self.reward_range,
            size=self._state.size
        )
        
        return self._state.observations()
 
    def step(self, action):
        """
        Performs the action and returns the next observation, reward,
        done flag and info dict.
        """
        action = self._wrap_inputs(action)
        self._state.next(action, agentid=self.agent_turn, inplace=True)

        obs = self._state.observations()
        rewards = self._state.rewards()
        is_done = self._state.is_done()
        info = [{
            'winning_seq': self._state.winning_seq,
            'winner': self._state.winner
        }] * self.num_agents

        return self._wrap_outputs(obs, rewards, is_done, info)

try:
    from notebook_invoke import register_callback, remove_callback, jupyter_javascript_routines
    from IPython.display import display, HTML, Javascript
    import uuid

    class TicTacToeAgentJavascript:
        def __init__(self, env, error_traceback=False):
            self.id = uuid.uuid1().hex
            self.error_traceback = error_traceback
            env.error_handler = handle_error_nostop
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
            try:
                obs, reward, done, info = self.env.step(action)
                return {'board': obs.tolist(),
                        'reward': reward,
                        'done': done,
                        'info': info}
            except StopServerException as e:
                return None
            except BaseException as e:
                if self.error_traceback:
                    traceback.print_tb(e.__traceback__)
                else:
                    print(e)
                return None

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
                        if (data !== null) {
                            self.update_board(data['board']);
                            self.done = data['done'];
                            if(self.done) self.update_done(
                                data['info']['winner'],
                                data['info']['winning_seq']
                            );
                        }
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
