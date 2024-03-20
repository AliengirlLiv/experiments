import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output
from enum import Enum

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Action(Enum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    NO_OP = 3

def get_forward_pos(agent_pos, agent_dir):
    agent_row, agent_col = agent_pos
    if agent_dir == Direction.UP:
        agent_row -= 1
    elif agent_dir == Direction.RIGHT:
        agent_col += 1
    elif agent_dir == Direction.DOWN:
        agent_row += 1
    elif agent_dir == Direction.LEFT:
        agent_col -= 1
    return (agent_row, agent_col)


class GridGame(gym.Env):
    """
    A grid-based game environment compatible with OpenAI Gym.
    The game involves an agent that needs to move balls to a target zone while avoiding zombies.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self,
                 grid_width:int=10,
                 grid_height:int=10,
                 max_num_moves:int=30,
                 target_start_accept_fn=lambda coord: True,
                 agent_start_accept_fn=lambda coord, dir: True,
                 obs_type='text',
                 ):
        """
        Initialize the game environment.
        
        Parameters:
            grid_width (int): Width of the grid.
            grid_height (int): Height of the grid.
            target_start_accept_fn (function): Function that takes a coordinate tuple (row, col) and returns whether a ball can be placed there.
            agent_start_accept_fn (function): Function that takes a coordinate tuple (row, col) and direction and returns whether the agent can be placed there.
        """
        super(GridGame, self).__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_num_moves = max_num_moves
        self.target_start_accept_fn = target_start_accept_fn
        self.agent_start_accept_fn = agent_start_accept_fn
        
        # Throw an error if the board is too small
        if self.grid_width < 1 or self.grid_height < 1 or self.grid_width * self.grid_height < 2:
            raise ValueError("Board is too small; expected at least 2 squares")
        
        self.target_pos = None  # Position of the target zone (will be initialized on reset)
        self.agent_pos = None  # Position of the agent (will be initialized on reset)
        self.agent_dir = None  # Direction of the agent (will be initialized on reset)
        self.num_moves = 0  # Number of moves taken by the agent
        
        
        
        # Board-ground is a 2D list of strings representing the ground layer of the board
 
        # Define the action space (4 moves: left, right, forward, no-op)
        self.action_space = spaces.Discrete(4)
        
        # Define the observation space
        self.observation_space = None
        self.obs_type = obs_type
        if obs_type == 'text':
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif obs_type == 'dict':
            self.observation_space = spaces.Dict({
                'agent_pos': spaces.Tuple((spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width))),
                'agent_dir': spaces.Discrete(4),
                'target_pos': spaces.Tuple((spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width))),
            })
        elif obs_type == 'arr':
            self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        elif obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_height, self.grid_width, 3), dtype=np.float32)
        elif obs_type == 'textified_state':
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif obs_type == 'short_text':
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif obs_type == 'multi':
            self.observation_space = spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                spaces.Dict({
                    'agent_pos': spaces.Tuple((spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width))),
                    'agent_dir': spaces.Discrete(4),
                    'target_pos': spaces.Tuple((spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width))),
                }),
                spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
                spaces.Box(low=0, high=1, shape=(self.grid_height, self.grid_width, 3), dtype=np.float32),
            ))
        else:
            raise ValueError(f"Invalid observation type: {obs_type}")

        # Gym-specific attributes
        self.seed()
        self.viewer = None
        
        # Call reset to initialize board and agent
        self.reset()

    def seed(self, seed=None):
        """
        Set the seed for this environment's random number generator.
        
        Parameters:
        seed (int or None): The seed to use. If None, a random seed will be used.
        
        Returns:
        list: A list containing the seed used.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the game environment to its initial state.
        
        Returns:
        np.array: The initial observation of the game state.
        """
        # Randomly choose the target zone position
        while True:
            target_row = np.random.randint(0, self.grid_height)
            target_col = np.random.randint(0, self.grid_width)
            if self.target_start_accept_fn((target_row, target_col)):
                break
        # Randomly choose the agent position and direction
        while True:
            agent_row = np.random.randint(0, self.grid_height)
            agent_col = np.random.randint(0, self.grid_width)
            agent_dir = np.random.choice(Direction)
            if self.agent_start_accept_fn((agent_row, agent_col), agent_dir):
                break
        self.target_pos = (target_row, target_col)
        self.agent_pos = (agent_row, agent_col)
        self.agent_dir = agent_dir
        self.num_moves = 0

        # Return the initial observation
        return self.get_observation(self.obs_type)
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Parameters:
        action (int): An action to take, represented as an integer.
        
        Returns:
        tuple: A tuple containing:
            - np.array: The observation after taking the action.
            - float: The reward obtained after taking the action.
            - bool: Whether the game is over after taking the action.
            - dict: Extra information about the step.
        """
        # Map the discrete action to a Direction
        game_action = Action(action)
        self.move_agent(game_action)
        
        # Check if the game is over
        done = self.num_moves >= self.max_num_moves
        
        # Reward is 1 if the agent is in the target zone on the last timestep, 0 otherwise
        reward = 1 if done and self.agent_pos == self.target_pos else 0
        
        # Return step information
        return self.get_observation(self.obs_type), reward, done, {}
    
    def get_observation(self, obs_type):
        """
        Get the current observation of the game state.
        
        Returns:
        np.array: The current observation of the game state.
        """
        # Convert the board state to a binary representation for the observation
        def format_coord(coord):
            return f'({coord[0]}, {coord[1]})'
        dir_str = Direction(self.agent_dir).name.lower()
        
        if obs_type == 'text':
            return f'The agent is at {format_coord(self.agent_pos)}, facing {dir_str}. The target is at {format_coord(self.target_pos)}.'
        elif obs_type == 'short_text':
            return f'Agent at {self.agent_pos[0]}, {self.agent_pos[1]} facing {dir_str}. Target at {self.target_pos[0]}, {self.target_pos[1]}.'
        elif obs_type == 'dict':
            return {'agent_pos': self.agent_pos, 'agent_dir': self.agent_dir, 'target_pos': self.target_pos}
        elif obs_type == 'arr':
            return np.array([self.agent_pos[0], self.agent_pos[1], self.agent_dir.value, self.target_pos[0], self.target_pos[1]])
        elif obs_type == 'image':
            return self.get_image()
        elif obs_type == 'textified_state':
            arr_state = np.array([self.agent_pos[0], self.agent_pos[1], self.agent_dir.value, self.target_pos[0], self.target_pos[1]])
            return ' ' + ' '.join(map(str, arr_state.tolist()))
        else:
            raise ValueError(f"Invalid observation type: {obs_type}")
    
    def get_image(self):
        arr = np.zeros((self.grid_height, self.grid_width, 3)) - 1
        arr[self.target_pos[0], self.target_pos[1], 1] = 4
        arr[self.agent_pos[0], self.agent_pos[1], 2] = self.agent_dir
        return arr
        
    
    def render(self, mode='human'):
        """
        Render the current game state.
        
        Parameters:
        mode (str): The mode to render with ('human' or 'rgb_array').
        """
        if mode == 'rgb_array':
            return self.get_image()
        elif mode == 'human':
            self.print_board_image()
    
    def close(self):
        """
        Perform any necessary cleanup.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None                   

    def move_agent(self, action):
        self.num_moves += 1
        
        agent_row, agent_col = self.agent_pos
        if action == Action.FORWARD:
            agent_row, agent_col = get_forward_pos(self.agent_pos, self.agent_dir)
                
            # Agent cannot go outside of grid
            agent_row = max(0, agent_row)
            agent_row = min(self.grid_height - 1, agent_row)
            agent_col = max(0, agent_col)
            agent_col = min(self.grid_width - 1, agent_col)
            self.agent_pos = (agent_row, agent_col)
            
                
        elif action == Action.RIGHT:
            self.agent_dir = Direction((self.agent_dir.value + 1) % 4)
                
        elif action == Action.LEFT:
            self.agent_dir = Direction((self.agent_dir.value - 1) % 4)


    def print_board(self):
        agent_char = ['^', '>', 'v', '<'][self.agent_dir.value]
        for i in range(self.grid_height):
            row = []
            for j in range(self.grid_width):
                if (i, j) == self.agent_pos:
                    row.append(agent_char)
                elif (i, j) == self.target_pos:
                    row.append('X')
                else:
                    row.append('_')
            print(' '.join(row))

    def play_game(self, print_fn='text'):
        print("Welcome to the Grid Game!")
        if print_fn == 'text':
            self.print_board()
        elif print_fn == 'image':
            self.print_board_image()
        print("Commands: w (move), a (turn left), d (turn right), s (no-op), quit")
        
        while True:
            action = input("Enter your move: ").lower().strip()
            if action in ['w', 'a', 'd', 's']:
                action_int = ['w', 'd', 's', 'a'].index(action)
                obs, reward, done, _ = self.step(action_int)
                if print_fn == 'text':
                    self.print_board()
                elif print_fn == 'image':
                    self.print_board_image()
                else:
                    raise NotImplementedError
                print(obs)
                print(f"Moves taken: {self.num_moves}")
                if done:
                    print(f"Game over! Your score: {reward}")
                    break
            elif action == 'quit':
                print("Thanks for playing!")
                break
    
    def print_board_image(self):
        plt.close()
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(3,3))
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        ax.set_xticks(range(self.grid_width + 1))
        ax.set_yticks(range(self.grid_height + 1))
        ax.grid(which='both')

        # Add a colored patch for the target zone
        ax.add_patch(patches.Rectangle((self.target_pos[1], self.grid_height-self.target_pos[0]-1), 1, 1, facecolor='green'))

        i, j = self.agent_pos
        if self.agent_dir == Direction.UP:
            points = [(j+0.5, self.grid_height-i), (j, self.grid_height-i-0.5), (j+1, self.grid_height-i-0.5)]
        elif self.agent_dir == Direction.DOWN:
            points = [(j+0.5, self.grid_height-i-1), (j, self.grid_height-i-0.5), (j+1, self.grid_height-i-0.5)]
        elif self.agent_dir == Direction.RIGHT:
            points = [(j+1, self.grid_height-i-0.5), (j+0.5, self.grid_height-i), (j+0.5, self.grid_height-i-1)]
        elif self.agent_dir == Direction.LEFT:
            points = [(j, self.grid_height-i-0.5), (j+0.5, self.grid_height-i), (j+0.5, self.grid_height-i-1)]
        triangle = patches.Polygon(points, closed=True, facecolor='blue')
        ax.add_patch(triangle)
        display(plt.gcf())


def oracle(state_dict):
    agent_pos = state_dict['agent_pos']
    agent_dir = state_dict['agent_dir']
    target_pos = state_dict['target_pos']
    
    # If the agent is in the target zone, we can stop
    if agent_pos == target_pos:
        long_reasoning = " We're at the target, so don't move. ACTION = NO_OP"
        short_reasoning = ' At target. ACTION = NO_OP'
        action_only_reasoning = ' ACTION = NO_OP'
        return Action.NO_OP.value, {'long_reasoning': long_reasoning, 'short_reasoning': short_reasoning, 'action_only_reasoning': action_only_reasoning}
    
    reasoning = ' Remember, the coordinates are (row, col), so (y, x). The origin is the top left.'
    short_reasoning = ''
    
    delta_y = target_pos[0] - agent_pos[0]
    delta_x = target_pos[1] - agent_pos[1]
    reasoning += f' We should move {delta_y} in the y direction and {delta_x} in the x direction.'
    short_reasoning += f' Distance to target: {delta_y}, {delta_x}.'
    
    # Figure out which direction moving forward would take us
    if agent_dir == Direction.UP:
        delta = -1
        direction = 'y'
    elif agent_dir == Direction.DOWN:
        delta = 1
        direction = 'y'
    elif agent_dir == Direction.LEFT:
        delta = -1
        direction = 'x'
    elif agent_dir == Direction.RIGHT:
        delta = 1
        direction = 'x'
    reasoning += f' Moving forward takes us {delta} in the {direction} direction.'
    
    # If this is the right direction, take it
    relevant_direction = delta_x if direction == 'x' else delta_y
    if np.sign(relevant_direction) == np.sign(delta):
        reasoning += f" This is the right direction, so let's move forward. ACTION = FORWARD"
        short_reasoning += f' Move forward. ACTION = FORWARD'
        return Action.FORWARD.value, {'long_reasoning': reasoning, 'short_reasoning': short_reasoning, 'action_only_reasoning': ' ACTION = FORWARD'}
    
    # if we're facing exactly the wrong direction, turn around
    not_relevant_direction = delta_x if direction == 'y' else delta_y
    if not_relevant_direction == 0:  # No need to move laterally
        assert np.sign(relevant_direction) == - delta
        reasoning += f' We are facing the wrong direction, so turn around. ACTION = LEFT'
        short_reasoning += f' Turn around. ACTION = LEFT'
        return Action.LEFT.value, {'long_reasoning': reasoning, 'short_reasoning': short_reasoning, 'action_only_reasoning': ' ACTION = LEFT'}
    
    reasoning += f" This is wrong, so we should turn."
    # Otherwise, turn left or right
    if agent_dir == Direction.UP:
        left_dir = '-x'
        right_dir = '+x'
    elif agent_dir == Direction.DOWN:
        left_dir = '+x'
        right_dir = '-x'
    elif agent_dir == Direction.LEFT:
        left_dir = '+y'
        right_dir = '-y'
    elif agent_dir == Direction.RIGHT:
        left_dir = '-y'
        right_dir = '+y'

    reasoning += f' Turning left lets us travel in the {left_dir} direction, and turning right lets us travel in the {right_dir} direction.'
    if 'y' in left_dir:
        dir_str = 'y'
        relevant_delta = delta_y
    else:
        dir_str = 'x'
        relevant_delta = delta_x
    sign_str = '+' if relevant_delta > 0 else '-'
    desired_dir_str = f'{sign_str}{dir_str}'
    go_left = left_dir == desired_dir_str
    reasoning += f' We want to travel in the {desired_dir_str} direction,'
    reasoning += f' so we should turn {"left" if go_left else "right"}. ACTION = {"LEFT" if go_left else "RIGHT"}'
    short_reasoning += f' Head {desired_dir_str}. ACTION = {"LEFT" if go_left else "RIGHT"}.'
    action_only_reasoning = f' ACTION = {"LEFT" if go_left else "RIGHT"}'
    return Action.LEFT.value if go_left else Action.RIGHT.value, {'long_reasoning': reasoning, 'short_reasoning': short_reasoning, 'action_only_reasoning': action_only_reasoning}
    
    
    