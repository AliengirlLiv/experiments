from itertools import product
import re
import numpy as np
import random
from datasets import Dataset
import pickle as pkl
import torch


X = 'X'
O = 'O'
EMPTY = '_'

def action_str_to_idxs(action_str):
    assert len(action_str) == 2 and action_str[0].isalpha() and action_str[1].isdigit()
    return ord(action_str[0].lower()) - ord('a'), int(action_str[1]) - 1

def idxs_to_action_str(idxs):
    return chr(idxs[0] + ord('a')) + str(idxs[1] + 1)

def get_value(board, action_or_idxs):
    if isinstance(action_or_idxs, str):
        action_or_idxs = action_str_to_idxs(action_or_idxs)
    return board[action_or_idxs[0]][action_or_idxs[1]]

def is_open(board, action_or_idxs):
    if isinstance(action_or_idxs, str):
        action_or_idxs = action_str_to_idxs(action_or_idxs)
    return get_value(board, action_or_idxs) == EMPTY



def evaluate_move(board, action, optimal_actions):
    parsable_move = action is not None
    legal_move = False if not parsable_move else is_open(board, action)
    good_move = False if not parsable_move else action in optimal_actions
    return {
        'parsable_move': parsable_move,
        'legal_move': legal_move,
        'good_move': good_move,
    }


def format_move(move):
    # convert row to [a-c] and col to [1-3]
    row = chr(ord('a') + move[0])
    col = str(move[1] + 1)
    return row + col

def best_move(board, player):
    """
    This function computes the optimal next move for the given player in a game of tic-tac-toe.

    :param board: A k x k nested list representing the tic-tac-toe board
    :param player: A string, 'X' or 'O', representing the current player
    :return: A list of tuples representing the optimal move(s) for the player
    """

    k = len(board)

    def win(current_board, player):
        # Winning conditions: k in a row horizontally, vertically, or diagonally
        win_cond = []
        for i in range(k):
            win_cond.append([current_board[i][j] for j in range(k)])  # rows
            win_cond.append([current_board[j][i] for j in range(k)])  # columns
        win_cond.append([current_board[i][i] for i in range(k)])  # diagonal
        win_cond.append([current_board[i][k - i - 1] for i in range(k)])  # anti-diagonal
        return [player] * k in win_cond

    def minimax(current_board, depth, is_maximizing):
        if win(current_board, X):
            return 10 - depth
        elif win(current_board, O):
            return -10 + depth
        elif not any(EMPTY in row for row in current_board):
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for i in range(k):
                for j in range(k):
                    if current_board[i][j] == EMPTY:
                        current_board[i][j] = X
                        score = minimax(current_board, depth + 1, False)
                        current_board[i][j] = EMPTY
                        best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(k):
                for j in range(k):
                    if current_board[i][j] == EMPTY:
                        current_board[i][j] = O
                        score = minimax(current_board, depth + 1, True)
                        current_board[i][j] = EMPTY
                        best_score = min(best_score, score)
            return best_score

    moves = []
    best_score = float('-inf') if player == X else float('inf')

    for i in range(k):
        for j in range(k):
            if board[i][j] == EMPTY:
                board[i][j] = player
                score = minimax(board, 0, player == O)
                board[i][j] = EMPTY
                if (player == X and score > best_score) or (player == O and score < best_score):
                    best_score = score
                    moves = [(i, j)]
                elif score == best_score:
                    moves.append((i, j))

    return [format_move(m) for m in moves]
    
def check_winner(board, diagonal_win=None, player=None):
    size = int(len(board) ** 0.5)
    lines = []
    
    if not diagonal_win:
        # Rows
        for i in range(size):
            lines.append(board[i*size : (i+1)*size])
        
        # Columns
        for i in range(size):
            lines.append(board[i::size])
    
    # Diagonals
    if diagonal_win or diagonal_win is None:
        lines.append(board[::size+1])
        lines.append(board[size-1:len(board)-1:size-1])
    
    for line in lines:
        if player is None:
            if line in [[X]*size, [O]*size]:
                return True
        else:
            if line == [player]*size:
                return True
    return False

def player_one_move_from_winning(board, player, diagonal_win=None):
    for i in range(len(board)):
        if board[i] == EMPTY:
            board[i] = player
            if check_winner(board, diagonal_win, player):
                board[i] = EMPTY
                return True
            board[i] = EMPTY
    return False

def generate_exhaustive_set(player= X, only_one_step=True, next_step_diag_wins=None, board_size=3, max_count=None):
    other_player = O if player == X else X
    valid_boards = []
    combos = product(f'{EMPTY}{X}{O}', repeat=board_size**2)
    shuffled_combos = np.random.permutation(list(combos))
    for state in shuffled_combos:
        if max_count and len(valid_boards) >= max_count:
            break
        board = list(state)
        if (board.count(other_player) - board.count(player) in [0, 1]) and not check_winner(board, diagonal_win=None) and EMPTY in board:
            one_move_away = player_one_move_from_winning(board, player, next_step_diag_wins)
            if one_move_away == only_one_step:
                    board = np.array(board).reshape(board_size, board_size)
                    valid_boards.append(board)
    best_actions = [best_move(board, player) for board in valid_boards]
    return {'boards': valid_boards, 'best_actions': best_actions}


def generate_random_one_step_tic_tac_toe(k, player):
    opponent = O if player == X else X
    def is_winning(board, player):
        for i in range(k):
            if all([board[i][j] == player for j in range(k)]) or all([board[j][i] == player for j in range(k)]):
                return True
        if all([board[i][i] == player for i in range(k)]) or all([board[i][k-i-1] == player for i in range(k)]):
            return True
        return False

    def get_empty_positions(board):
        positions = []
        for i in range(k):
            for j in range(k):
                if board[i][j] == EMPTY:
                    positions.append((i, j))
        return positions

    while True:
        board = [[EMPTY for _ in range(k)] for _ in range(k)]
        player_tokens = [X, O]
        random.shuffle(player_tokens)
        for i in range(k*k):  # Fill the board
            token = player_tokens[i % 2]
            # Randomly place the token on an empty square
            empty_positions = get_empty_positions(board)
            pos = random.choice(empty_positions)
            board[pos[0]][pos[1]] = token
            if is_winning(board, token):
                break
        # If oponnent won, try again
        if check_winner(np.array(board).flatten().tolist(), player=opponent):
            continue
        # If nobody won, try again
        if not check_winner(np.array(board).flatten().tolist()):
            continue
        # player won, so remove the last placed token
        board[pos[0]][pos[1]] = EMPTY
        break
    best_action = best_move(np.array(board), player)
    return np.array(board), best_action


def board_to_string(board):
    k = len(board)  # Determine the size of the board
    result = []

    # Generate row identifiers
    row_identifiers = [chr(ord('a') + i) for i in range(k)]

    for row_index, row in enumerate(board):
        for col_index, value in enumerate(row):
            # Use the appropriate row identifier
            position = f"{row_identifiers[row_index]}{col_index + 1}"
            result.append(f"{position}={value.lower()}")

    # Join all position-value pairs with a comma and a space
    return ', '.join(result)

def formatting_prompts_func(examples, include_labels=True, include_label_prompt=True, eos=None, description=None):
    formatted_examples = []
    
    for board, actions in zip(examples['boards'], examples['best_actions']):
        board_str = board_to_string(board) + '.'
        first_action = actions[0]
        
        desc_str = f"{description} " if description else ""
        eos_str = f"{eos}" if eos else ""
        label_prompt_str = " Answer:" if include_label_prompt else ""
        action_str = f" MOVE[{first_action}]" if include_labels else ""
        
        formatted_example = f"{desc_str}{board_str}{label_prompt_str}{action_str}{eos_str}"
        formatted_examples.append(formatted_example)
    
    return formatted_examples

def parse_action_from_string(s):
    # Define the regular expression pattern for the action
    pattern = r"MOVE\[(.*?)\]"
    
    # Search for the pattern in the input string
    match = re.search(pattern, s)
    
    # Extract and return the action if found, otherwise return None
    return match.group(1) if match else None


def generate_dataset(val_set_size):  # TODO: custom svae dir
    train_set = generate_exhaustive_set(player=X, only_one_step=True, next_step_diag_wins=False, board_size=3)
    print(f'original train_set_size: {len(train_set["best_actions"])}')

    val_set_iid = generate_exhaustive_set(player=X, only_one_step=True, next_step_diag_wins=False, board_size=3, max_count=val_set_size)
    print(f'finished generating val_set_iid')
    val_set_diag_wins = generate_exhaustive_set(player=X, only_one_step=True, next_step_diag_wins=True, board_size=3, max_count=val_set_size)
    print(f'finished generating val_set_diag_wins')
    val_set_player_o = generate_exhaustive_set(player=O, only_one_step=True, next_step_diag_wins=False, board_size=3, max_count=val_set_size)
    print(f'finished generating val_set_player_o')
    val_set_size_4_list =  [generate_random_one_step_tic_tac_toe(4, X) for _ in range(val_set_size)]
    val_set_size_4 = {'boards': [x[0] for x in val_set_size_4_list], 'best_actions': [x[1] for x in val_set_size_4_list]}
    print(f'finished generating val_set_size_4')
    val_set_not_one_step = generate_exhaustive_set(player=X, only_one_step=False, next_step_diag_wins=None, board_size=3, max_count=val_set_size)
    print(f'finished generating val_set_not_one_step')
    
    train_dataset = Dataset.from_dict(train_set)
    val_dataset_iid = Dataset.from_dict(val_set_iid)
    val_dataset_diag_wins = Dataset.from_dict(val_set_diag_wins)
    val_dataset_not_one_step = Dataset.from_dict(val_set_not_one_step)
    val_dataset_player_o = Dataset.from_dict(val_set_player_o)
    val_dataset_size_4 = Dataset.from_dict(val_set_size_4)
    
    with open('train_dataset.pkl', 'wb') as f:
        pkl.dump(train_dataset, f)

    with open('val_dataset_diag_wins.pkl', 'wb') as f:
        pkl.dump(val_dataset_diag_wins, f)

    with open('val_dataset_iid.pkl', 'wb') as f:
        pkl.dump(val_dataset_iid, f)
        
    with open('val_dataset_not_one_step.pkl', 'wb') as f:
        pkl.dump(val_dataset_not_one_step, f)

    with open('val_dataset_player_o.pkl', 'wb') as f:
        pkl.dump(val_dataset_player_o, f)

    with open('val_dataset_size_4.pkl', 'wb') as f:
        pkl.dump(val_dataset_size_4, f)


def load_dataset(): # TODO: custom load dir
    with open('train_dataset.pkl', 'rb') as f:
        train_dataset = pkl.load(f)

    with open('val_dataset_diag_wins.pkl', 'rb') as f:
        val_dataset_diag_wins = pkl.load(f)

    with open('val_dataset_iid.pkl', 'rb') as f:
        val_dataset_iid = pkl.load(f)

    with open('val_dataset_not_one_step.pkl', 'rb') as f:
        val_dataset_not_one_step = pkl.load(f)

    with open('val_dataset_player_o.pkl', 'rb') as f:
        val_dataset_player_o = pkl.load(f)

    with open('val_dataset_size_4.pkl', 'rb') as f:
        val_dataset_size_4 = pkl.load(f)

    return {
        'train_dataset': train_dataset,
        'val_dataset_diag_wins': val_dataset_diag_wins,
        'val_dataset_iid': val_dataset_iid,
        'val_dataset_not_one_step': val_dataset_not_one_step,
        'val_dataset_player_o': val_dataset_player_o,
        'val_dataset_size_4': val_dataset_size_4
    }

def get_descriptions():
    description_train = "You are X. Pick your move."
    description_val_iid = description_train
    description_val_diag_wins = description_train
    description_val_not_one_step = description_train
    description_val_player_o = "You are O. Pick your move."
    desciprion_val_size_4 = "This is a modified version of tic-tac-toe where the board is 4x4. You win by having 4 in a row, column or diagonal. You are X. Pick your move."
    return {
        'train_dataset': description_train,
        'val_dataset_iid': description_val_iid,
        'val_dataset_diag_wins': description_val_diag_wins,
        'val_dataset_not_one_step': description_val_not_one_step,
        'val_dataset_player_o': description_val_player_o,
        'val_dataset_size_4': desciprion_val_size_4
    }
    

def evaluate(model, tokenizer, dataset, batch_size, generator_max_length, formatting_fn):
    model.eval()  # Set the model to evaluation mode
    
    # Generate completions for the evaluation dataset
    total_count = 0  # Total number of completions
    parsable = 0  # Number of completions that are well-formatted
    legal = 0  # Number of completions that move into an open space
    optimal = 0  # Number of completions that move into an optimal space     
    generations = []

    while total_count < len(dataset):
        batch = dataset[total_count:total_count + batch_size]            
        input_strings = formatting_fn(batch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(input_strings, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad(): 
            output = model.generate(**inputs, max_new_tokens=generator_max_length, pad_token_id=tokenizer.pad_token_id)
        predicted_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        parsed_answers = [parse_action_from_string(s) for s in predicted_text]
        evaluations = [evaluate_move(batch['boards'][i], parsed_answers[i], batch['best_actions'][i]) for i in range(len(batch['boards']))]
        parsable += sum([e['parsable_move'] for e in evaluations])
        legal += sum([e['legal_move'] for e in evaluations])
        optimal += sum([e['good_move'] for e in evaluations])
        total_count += len(batch['boards'])
        generations += predicted_text

    # Calculate the metrics
    metrics = {
        "parsable": parsable / len(dataset),
        "legal": legal / len(dataset),
        "optimal": optimal / len(dataset),
        "generations": generations,
    }
    return metrics
        