import gym
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import pickle
from simple_gridworld import GridGame, oracle
import pathlib
import numpy as np


# Step 1: Collect Demos
def collect_demos(env, num_episodes=10000, save_dir='demos'):
    demos_state = []
    demos_state_only_text = []
    demos_short_text_reasoning = []
    demos_short_text_no_reasoning = []
    demos_text = []
    demos_dict = []
    
    for i in range(num_episodes):
        if i % 100 == 0:
            print(f"Episode {i}")
        arr_state = env.reset()
        done = False
        
        while not done:
            dict_state = env.get_observation('dict')
            text_state = env.get_observation('text')
            textified_state = env.get_observation('textified_state')
            short_text_state = env.get_observation('short_text')
            action, reasoning_dict = oracle(dict_state)
            long_reasoning = reasoning_dict['long_reasoning']
            short_reasoning = reasoning_dict['short_reasoning']
            action_only_reasoning = reasoning_dict['action_only_reasoning']
            next_state, reward, done, _ = env.step(action)
            demos_dict.append((dict_state, action))
            demos_state.append((arr_state, action))
            demos_state_only_text.append((textified_state, action_only_reasoning))
            demos_short_text_reasoning.append((short_text_state, short_reasoning))
            demos_short_text_no_reasoning.append((short_text_state, action_only_reasoning))
            demos_text.append((text_state, long_reasoning))
            arr_state = next_state
        if not reward == 1:
            import pdb; pdb.set_trace()
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / 'demos_state.pkl', 'wb') as f:
        pickle.dump(demos_state, f)
    with open(save_dir / 'demos_text.pkl', 'wb') as f:
        pickle.dump(demos_text, f)
    with open(save_dir / 'demos_dict.pkl', 'wb') as f:
        pickle.dump(demos_dict, f)
    with open(save_dir / 'demos_state_only_text.pkl', 'wb') as f:
        pickle.dump(demos_state_only_text, f)
    with open(save_dir / 'demos_short_text_reasoning.pkl', 'wb') as f:
        pickle.dump(demos_short_text_reasoning, f)
    with open(save_dir / 'demos_short_text_no_reasoning.pkl', 'wb') as f:
        pickle.dump(demos_short_text_no_reasoning, f)
    return demos_state

# Step 2: MLP Policy with 3 Layers
class MLPPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
        super(MLPPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def eval(env, model, val_demos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        val_loss = 0
        for state, action in val_demos:
            state = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            action = torch.tensor([action], dtype=torch.long).to(device)
            output = model(state)
            loss = criterion(output, action)
            val_loss += loss.item()

    # Evaluate agent performance
    total_reward = 0
    num_episodes = 50
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
            action = model(state_tensor).argmax().item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return val_loss, total_reward / num_episodes

# Step 3: Training with Behavioral Cloning
def train(model, optimizer, criterion, train_demos, val_demos, ood_demos, env, epochs=1, batch_size=512, eval_interval=100):
    
    # Initialize wandb
    wandb.init(project="language-agents")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        model = model.to(device)
        
        # shuffle demos
        random.shuffle(train_demos)
        train_loss_list = []
        
        for i in range(0, len(train_demos), batch_size):
            batch = train_demos[i:i+batch_size]
            states, actions = zip(*batch)
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            train_loss_list.append(loss.item())
            optimizer.step()

            total_loss += loss.item()
        
            if i % batch_size * 100 == 99:
                wandb.log({"train_loss": sum(train_loss_list) / len(train_loss_list)})
            print(f"Epoch {epoch} batch {i} loss: {total_loss / len(train_demos)}")

            # Validation
            if epoch % eval_interval == 0:
                model.eval()
                val_loss, val_reward = eval(env, model, val_demos)
                wandb.log({"val_loss": val_loss / len(val_demos)})
                print(f"Epoch {epoch} val loss: {val_loss / len(val_demos)}")
                wandb.log({"val_reward": val_reward})
                print(f"Epoch {epoch} eval reward: {val_reward}")
                
                ood_val_loss, ood_val_reward = eval(ood_env, model, ood_demos)
                wandb.log({"ood_val_loss": ood_val_loss / len(ood_demos)})
                print(f"Epoch {epoch} ood val loss: {ood_val_loss / len(ood_demos)}")
                wandb.log({"ood_eval_reward": ood_val_reward})
                print(f"Epoch {epoch} ood eval reward: {ood_val_reward}")

# Save demos and model
def save_model(model, filename):
    torch.save(model.state_dict(), f"{filename}_model.pth")

# Main
if __name__ == "__main__":
    accept_top_target_fn = lambda coord: coord[0] < 5
    accept_bottom_target_fn = lambda coord: coord[0] >= 5
    
    env = GridGame(obs_type='arr', target_start_accept_fn=accept_top_target_fn)
    

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    model = MLPPolicy(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_demos = collect_demos(env, num_episodes=10000, save_dir='train_demos')
    print('TRAIN DEMOS LEN', len(train_demos))
    random.shuffle(train_demos)
    val_demos = collect_demos(env, num_episodes=100, save_dir='val_demos')
    ood_env = GridGame(obs_type='arr', target_start_accept_fn=accept_bottom_target_fn)
    ood_demos = collect_demos(ood_env, num_episodes=100, save_dir='ood_demos')

    train(model, optimizer, criterion, train_demos, val_demos, ood_demos, env)

    save_model(model, "rl_agent")
