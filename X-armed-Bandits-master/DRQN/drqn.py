import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import math

try:
    import NOMA_Gauss_Markov_DRQN as EnvModule 
    from Qnetwork import QNetwork
    from Replay_Buffer import RecurrentReplayBuffer
except ImportError as e:
    print(f"erreur: {e}")
    sys.exit()

EPISODES = 5000          
MAX_STEPS = 50           
BATCH_SIZE = 10
SEQ_LEN = 10             
GAMMA = 0.99             
EPSILON_START = 1.0      
EPSILON_END = 0.01       
EPSILON_DECAY = 1000     
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64          
CAPACITY = 2000          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entraînement lancé sur : {device}")

env = EnvModule.GaussMarkov(bruit=0.1) 

input_dim = 2   # [Action, Reward]
output_dim = env.K_actions if hasattr(env, 'K_actions') else 10 

policy_net = QNetwork(input_dim, HIDDEN_DIM, output_dim).to(device)
target_net = QNetwork(input_dim, HIDDEN_DIM, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = RecurrentReplayBuffer(CAPACITY, SEQ_LEN)

history_efficiency = []


epsilon = EPSILON_START

for episode in range(EPISODES):
    obs = env.reset() 
    hidden_state = None 
    episode_agent_reward = 0
    episode_oracle_reward = 0
    
    for step in range(MAX_STEPS):
        try:
            reward_oracle = env.get_oracle_reward()
        except AttributeError:
            # Fallback si tu n'as pas mis à jour la classe NOMA
            reward_oracle = 1.0 
            
        if reward_oracle == 0: reward_oracle = 1e-9

        obs_tensor = torch.tensor([obs], dtype=torch.float32).unsqueeze(0).to(device)
       
        if np.random.rand() < epsilon:
            action = np.random.randint(output_dim)
            with torch.no_grad():
                _, hidden_state = policy_net(obs_tensor, hidden_state)
        else:
            with torch.no_grad():
                q_values, hidden_state = policy_net(obs_tensor, hidden_state)
                action = q_values.argmax().item()
   
        next_obs, reward_agent, done = env.step(action)
       
        episode_agent_reward += reward_agent
        episode_oracle_reward += reward_oracle
        
        obs = next_obs
     
    epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)
    #epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * episode / EPSILON_DECAY)
    
    if episode_oracle_reward == 0: episode_oracle_reward = 1e-9
    efficiency = episode_agent_reward / episode_oracle_reward
    history_efficiency.append(efficiency)
    
    if episode % 100 == 0:
        print(f"Episode {episode}/{EPISODES} | Efficacité: {efficiency:.2f} | Epsilon: {epsilon:.2f}")

def plot_results(efficiencies, window=50):
    plt.figure(figsize=(10, 6))
    
    if len(efficiencies) >= window:
        smoothed = np.convolve(efficiencies, np.ones(window)/window, mode='valid')
    else:
        smoothed = efficiencies

    plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Oracle')
    plt.plot(smoothed, color='purple', linewidth=1.5, label='Efficacité (Agent/Oracle)')
    
    plt.title("Performance Normalisée par l'Oracle (Gauss-Markov)")
    plt.xlabel("Episodes")
    plt.ylabel("Efficacité Normalisée")
    plt.ylim(0, 1.2)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()

plot_results(history_efficiency, window=100)
