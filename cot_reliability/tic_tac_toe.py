import random
import re
import pandas as pd
from datasets import Dataset


def format_move(move):
    # convert row to [a-c] and col to [1-3]
    row = chr(ord('a') + move[0])
    col = str(move[1] + 1)
    return row + col

def best_move(board, player):
    """
    This function computes the optimal next move for the given player in a game of tic-tac-toe.
    
    :param board: A 3x3 nested list representing the tic-tac-toe board
    :param player: A string, 'x' or 'o', representing the current player
    :return: A list of tuples representing the optimal move(s) for the player
    """
    
    def win(current_board, player):
        # Winning conditions: three in a row horizontally, vertically, or diagonally
        win_cond = [
            [current_board[0][0], current_board[0][1], current_board[0][2]],
            [current_board[1][0], current_board[1][1], current_board[1][2]],
            [current_board[2][0], current_board[2][1], current_board[2][2]],
            [current_board[0][0], current_board[1][0], current_board[2][0]],
            [current_board[0][1], current_board[1][1], current_board[2][1]],
            [current_board[0][2], current_board[1][2], current_board[2][2]],
            [current_board[0][0], current_board[1][1], current_board[2][2]],
            [current_board[2][0], current_board[1][1], current_board[0][2]],
        ]
        return [player, player, player] in win_cond
    
    def minimax(current_board, depth, is_maximizing):
        if win(current_board, 'x'):
            return 10 - depth
        elif win(current_board, 'o'):
            return -10 + depth
        elif not any('_' in row for row in current_board):
            return 0
        
        if is_maximizing:
            best_score = float('-inf')
            for i in range(3):
                for j in range(3):
                    if current_board[i][j] == '_':
                        current_board[i][j] = 'x'
                        score = minimax(current_board, depth + 1, False)
                        current_board[i][j] = '_'
                        best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if current_board[i][j] == '_':
                        current_board[i][j] = 'o'
                        score = minimax(current_board, depth + 1, True)
                        current_board[i][j] = '_'
                        best_score = min(best_score, score)
            return best_score
        
    moves = []
    best_score = float('-inf') if player == 'x' else float('inf')
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == '_':
                board[i][j] = player
                score = minimax(board, 0, player == 'o')
                board[i][j] = '_'
                if (player == 'x' and score > best_score) or (player == 'o' and score < best_score):
                    best_score = score
                    moves = [(i, j)]
                elif score == best_score:
                    moves.append((i, j))
    
    return [format_move(m) for m in moves]

def print_board(board, mode='model2'):
    """
    This function takes a 3x3 tic-tac-toe board and prints out a string representation of the board.
    
    :param board: A 3x3 nested list representing the tic-tac-toe board
    """
    if mode == 'human':
        board_str = "|---+---+---|\n"
        for row in board:
            board_str +="| " + " | ".join(row) + " |\n"
            board_str += "|---+---+---|\n"
    elif mode == 'model':
        board_str = "  1 2 3\n"
        for i, row in enumerate(board):
            board_str += chr(ord('a') + i) + " " + " ".join(row) + "\n"
    elif mode == 'model2':
        board_str = ""
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                board_str += chr(ord('a') + i) + str(j+1) + "=" + cell + ", "
            board_str = board_str[:-2] + "\n"
    else:
        raise ValueError("Invalid board type")
    return board_str.strip('\n')

def test_action_correct(action, optimal_actions):
    return action in optimal_actions



def generate_random_one_step_tic_tac_toe():
    def is_winning(board, player):
        for i in range(3):
            if all([board[i][j] == player for j in range(3)]) or all([board[j][i] == player for j in range(3)]):
                return True
        if all([board[i][i] == player for i in range(3)]) or all([board[i][2-i] == player for i in range(3)]):
            return True
        return False

    def get_empty_positions(board):
        positions = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == '_':
                    positions.append((i, j))
        return positions

    while True:
        board = [['_' for _ in range(3)] for _ in range(3)]
        player_tokens = ['x', 'o']
        random.shuffle(player_tokens)
        for i in range(9):  # Fill the board
            token = player_tokens[i % 2]
            # Randomly place the token on an empty square
            empty_positions = get_empty_positions(board)
            pos = random.choice(empty_positions)
            board[pos[0]][pos[1]] = token
            if is_winning(board, token):
                break
        # If O won, try again
        if is_winning(board, 'o'):
            continue
        # If nobody won, try again
        if not is_winning(board, 'x'):
            continue
        # X won, so remove the last placed token
        board[pos[0]][pos[1]] = '_'
        break
    return board


def generate_random_tic_tac_toe():
    """
    Generates a random tic-tac-toe board with the following conditions:
    - Half the time 'o' should have the same number of pieces as 'x'. The other half, there should be one more 'o'.
    - The game should never be over (i.e., there should not be 3 in a row for either player).
    - The number of pieces 'o' has should follow the given probability distribution.
    """
    # Probability distribution for the number of 'o' pieces
    distribution = {0: 0.1, 1: 0.25, 2: 0.30, 3: 0.3, 4: 0.05}
    
    # Choose the number of 'o' pieces based on the distribution
    o_pieces = random.choices(list(distribution.keys()), list(distribution.values()))[0]
    
    # Decide whether 'o' has the same number of pieces as 'x', or one more
    x_pieces = max(o_pieces if random.random() < 0.5 else o_pieces - 1, 0)

    # Initialize an empty board
    board = [['_' for _ in range(3)] for _ in range(3)]

    def check_winner(board, player):
        # Check rows, columns and diagonals for a win
        for i in range(3):
            if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
                return True
        if board[0][0] == board[1][1] == board[2][2] == player or board[0][2] == board[1][1] == board[2][0] == player:
            return True
        return False

    def place_pieces(board, player, pieces):
        for _ in range(100):
            if pieces == 0:
                return True
            i, j = random.randint(0, 2), random.randint(0, 2)
            if board[i][j] == '_':
                board[i][j] = player
                if check_winner(board, player):
                    # Undo the last move if it causes a win
                    board[i][j] = '_'
                else:
                    pieces -= 1
        return False

    # Place the pieces on the board
    while not place_pieces(board, 'o', o_pieces) or not place_pieces(board, 'x', x_pieces):
        print("Failed to place pieces, trying again...", o_pieces, x_pieces)
        # Reset the board if the placement fails (due to a winning condition or a full board)
        board = [['_' for _ in range(3)] for _ in range(3)]

    return board




prompt_with_reasoning = """Below, you will see a tic-tac-toe board. You are player 'x', and it is your turn next. You must reason carefully about different potential places you can play, and then select the best action. Winning is better than a draw, and a draw is better than losing. If there are multiple tied actions, pick any of them.

Boards look like this:
|---+---+---|
| x | x | _ |
|---+---+---|
| o | x | _ |
|---+---+---|
| o | _ | _ |
|---+---+---|
Rows are numbers a-c, and columns are numbered 1-3. So for instance, the top-left square is a1, and the top-middle square is a2.

Respond with a chain of reasoning, followed by your final answer, which should be formatted like this: \\action{coordinate}.

Here are some examples:

|---+---+---|
| x | x | _ |
|---+---+---|
| o | o | _ |
|---+---+---|
| _ | _ | _ |
|---+---+---|
I am player x. I have pieces in a1 and a2. Cell a3 is open, so I can place a piece there to win with three in a row (a1, a2, a3). Therefore, my answer is \\action{a3}.


|---+---+---|
| x | _ | _ |
|---+---+---|
| o | o | _ |
|---+---+---|
| _ | _ | _ |
|---+---+---|
I am player x. I have pieces in a1. Since I only have one piece, there is no way for me to place a piece which makes me win in one turn. Next, I'll check if I can block player o from winning. Player o has pieces in b1 and b2. Player o could win by placing a piece in b3. I will block this by playing in b3. My answer is \\action{b3}.


|---+---+---|
| _ | _ | _ |
|---+---+---|
| _ | _ | o |
|---+---+---|
| _ | _ | x |
|---+---+---|
I am player x. I have pieces in c3. Since I only have one piece, there is no way for me to place a piece which makes me win in one turn. Next, I'll check if I can block player o from winning. Player o has only one piece, so they cannot win in one turn. Next, I'll try to set myself up for victory by getting 2 in a row. I will play in c2, which gives me 2 in a row. \\action{c2}.

Now, here's the real board:
"""

prompt_without_reasoning = """Below, you will see a tic-tac-toe board. You are player 'x', and it is your turn next. You must reason carefully about different potential places you can play, and then select the best action. Winning is better than a draw, and a draw is better than losing. If there are multiple tied actions, pick any of them.

Boards look like this:
|---+---+---|
| x | x | _ |
|---+---+---|
| o | x | _ |
|---+---+---|
| o | _ | _ |
|---+---+---|
Rows are numbers a-c, and columns are numbered 1-3. So for instance, the top-left square is a1, and the top-middle square is a2.

Respond with a chain of reasoning, followed by your final answer, which should be formatted like this: \\action{coordinate}.

Here are some examples:

|---+---+---|
| x | x | _ |
|---+---+---|
| o | o | _ |
|---+---+---|
| _ | _ | _ |
|---+---+---|
\\action{a3}.


|---+---+---|
| x | _ | _ |
|---+---+---|
| o | o | _ |
|---+---+---|
| _ | _ | _ |
|---+---+---|
\\action{b3}

|---+---+---|
| _ | _ | _ |
|---+---+---|
| _ | _ | o |
|---+---+---|
| _ | _ | x |
|---+---+---|
\\action{c2}

Now, here's the real board:
"""

one_step_prompt_with_reasoning = """
Below, you will see a tic-tac-toe board. You are player 'x', and it is your turn next. You must reason carefully about different potential places you can play, and then select the best action. Winning is better than a draw, and a draw is better than losing. If there are multiple tied actions, pick any of them.

Boards look like this:
|---+---+---|
| x | x | _ |
|---+---+---|
| o | x | _ |
|---+---+---|
| o | _ | _ |
|---+---+---|
Rows are numbers a-c, and columns are numbered 1-3. So for instance, the top-left square is a1, and the top-middle square is a2.

Respond with a chain of reasoning, followed by your final answer, which should be formatted like this: \\action{coordinate}.

Here are some examples:

|---+---+---|
| x | x | _ |
|---+---+---|
| o | o | _ |
|---+---+---|
| _ | _ | _ |
|---+---+---|
I have pieces in a1 and a2. Cell a3 is open, so I can play there to win with (a1, a2, a3). My answer is \\action{a3}.

|---+---+---|
| _ | _ | x |
|---+---+---|
| o | _ | o |
|---+---+---|
| x | o | x |
|---+---+---|
I have pieces in a3, c1, and c3. Cell b2 is open, so I can play there to win with (a3, b2, c1). My answer is \\action{b2}.

Now, here's the real board:
"""

model_one_step_prompt_with_reasoning = """
Below, you will see a tic-tac-toe board. You are player 'x', and it is your turn next. Reason carefully and then select the best action.
Your answer should be formatted like this: \\action{coordinate}.

Here are some examples:

  1 2 3
a x x _
b o o _
c _ _ _
I have pieces in a1 and a2. Cell a3 is open, so I can play there to win with (a1, a2, a3). My answer is \\action{a3}.

  1 2 3
a _ _ x
b o - o
c x o x
I have pieces in a3, c1, and c3. Cell b2 is open, so I can play there to win with (a3, b2, c1). My answer is \\action{b2}.

Here's the real board:
"""

model_one_step_prompt_with_reasoning2 = """
Below, you will see a tic-tac-toe board. You are player 'x', and it is your turn next. Reason carefully and then select the best action.
Your answer should be formatted like this: \\action{coordinate}.

Here are some examples:

a1=x, a2=x, a3=_
b1=o, b2=o, b3=_
c1=_, c2=_, c3=_
I have pieces in a1 and a2. Cell a3 is open, so I can play there to win with (a1, a2, a3). My answer is \\action{a3}.


ax=_, a2=_, a3=x
b1=o, b2=_ b3=o
c1=x, c2=o, c3=x
I have pieces in a3, c1, and c3. Cell b2 is open, so I can play there to win with (a3, b2, c1). My answer is \\action{b2}.

Here's the real board:
"""

long_model_one_step_prompt_with_reasoning2 = """
Below, you will see a tic-tac-toe board. You are player 'x', and it is your turn next. Reason carefully and then select the best action.
Your answer should be formatted like this: \\action{coordinate}.

Here are some examples:

a1=x, a2=x, a3=_
b1=o, b2=o, b3=_
c1=_, c2=_, c3=_
I have pieces in a1 and a2. Cell a3 is open, so I can play there to win with (a1, a2, a3). My answer is \\action{a3}.

a1=_, a2=_, a3=x
b1=o, b2=_ b3=o
c1=x, c2=o, c3=x
I have pieces in a3, c1, and c3. Cell b2 is open, so I can play there to win with (a3, b2, c1). My answer is \\action{b2}.

a1=o, a2=x, a3=_
b1=o, b2=o, b3=x
c1=x, c2=o, c3=x
I have pieces in a2, b3, c1, and c3. Cell a3 is open, so I can play there to win with (a3, b3, c3). My answer is \\action{a3}.

a1=_, a2=_, a3=_
b1=o, b2=x, b3=_
c1=_, c2=x, c3=o
I have pieces in b2 and c2. Cell a2 is open, so I can play there to win with (a2, b2, c2). My answer is \\action{a2}.

a1=o, a2=o, a3=_
b1=_, b2=x, b3=o
c1=x, c2=_, c3=x
I have pieces in b2, c1, and c3. Cell c2 is open, so I can play there to win with (c1, c2, c3). My answer is \\action{c2}.

a1=x, a2=_, a3=o
b1=_, b2=x, b3=_
c1=_, c2=o, c3=_
I have pieces in a1 and b2. Cell c3 is open, so I can play there to win with (a1, b2, c3). My answer is \\action{c3}.

a1=o, a2=o, a3=x
b1=x, b2=_, b3=_
c1=_, c2=_, c3=x
I have pieces in a3, b1, and c3. Cell b3 is open, so I can play there to win with (a3, b3, c3). My answer is \\action{b3}.

a1=x, a2=o, a3=x
b1=_, b2=x, b3=o
c1=_, c2=x, c3=_
I have pieces in a1, a3, and b2, and c2. Cell c3 is open, so I can play there to win with (a1, b2, c3). My answer is \\action{c3}.

a1=_, a2=x, a3=o
b1=o, b2=x, b3=_
c1=_, c2=_, c3=_
I have pieces in a2 and b2. Cell c2 is open, so I can play there to win with (a2, b2, c2). My answer is \\action{c2}.

a1=_, a2=o, a3=_
b1=_, b2=x, b3=x
c1=_, c2=_, c3=_
I have pieces in b2 and b3. Cell b1 is open, so I can play there to win with (b1, b2, b3). My answer is \\action{b1}.

"""

def make_prompt(board, prompt):
    return prompt + print_board(board) + "\nI"


def parse_action(input_string):
    """
    Parses a string to find the action which is formatted as \action{action is in here}.

    :param input_string: A string potentially containing an action.
    :return: The action contained within the braces, or None if no action is found.
    """
    # Define the regular expression pattern for the action
    pattern = r'\\action\{(.*?)\}'

    # Use regular expression search to find the action
    match = re.search(pattern, input_string)

    # If a match is found, return the action, otherwise return None
    return match.group(1) if match else None


def generate_dataset(generate_random_tic_tac_toe, best_move, prompt_fn, samples=1000):
    data = []

    for i in range(samples):
        print(i)
        # Generate a random tic-tac-toe board
        board = generate_random_tic_tac_toe()

        # Get the list of optimal moves for the current player
        optimal_moves = best_move(board, 'x')

        # If there are multiple optimal moves, sample one randomly
        action = random.choice(optimal_moves) if optimal_moves else None

        # Create the formatted action string
        action_str = f"\\action{{{action}}}" if action else "\\action{}"

        # Append the board and the action string to the data list
        data.append({'text': prompt_fn(board) + " " + action_str})

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)

    # Create a Dataset from the DataFrame
    dataset = Dataset.from_pandas(df)

    return dataset
