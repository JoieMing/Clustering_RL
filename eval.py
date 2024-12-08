import os
import warnings
from stable_baselines3 import PPO  # Replace PPO with your algorithm, e.g., A2C, DQN, etc.
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from model import my_model
from src.argument import get_args
from env import MyGraphEnv
# from env_fixed_embeddings import MyGraphEnv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from utils import *
from src.argument import get_args

def evaluate_and_record_average_action(model, env, n_eval_episodes=10, deterministic=True):
    """
    Evaluate a model and compute the average action across episodes.

    Args:
        model: The trained Stable-Baselines3 model.
        env: The VecEnv environment.
        n_eval_episodes (int): Number of episodes to evaluate.
        deterministic (bool): Whether to use deterministic actions.

    Returns:
        mean_reward (float): Average reward over episodes.
        mean_action (float or np.ndarray): Average action across all steps.
    """
    total_rewards = []
    all_actions = []

    for episode in range(n_eval_episodes):
        obs = env.reset()  # VecEnv API
        episode_reward = 0
        episode_actions = []

        done = False
        while not done:
            # Predict the action
            action, _states = model.predict(obs, deterministic=deterministic)

            # Store the action
            if isinstance(action, (np.ndarray, list)):
                episode_actions.append(action)
            else:
                episode_actions.append([action])  # Handle scalar actions

            # Take the action in the environment
            obs, reward, done, _ = env.step(action)
            
            # VecEnv returns a list for `done`; use the first value
            done = done[0]
            reward = reward[0]  # Reward is also a list

            # Accumulate reward
            episode_reward += reward
        # print("action",action)
        # Track rewards and actions
        total_rewards.append(episode_reward)
        all_actions.extend(episode_actions)

    # Calculate mean reward
    mean_reward = np.mean(total_rewards)

    # Flatten actions if needed
    all_actions = np.array(all_actions)
    print(all_actions)
    mean_action = np.mean(all_actions, axis=0) if len(all_actions) > 0 else None

    return mean_reward, mean_action

def get_embedding(args,device,cluster_num,iter_num):
    # load data
    _, true_labels, _ = load_graph_data(args.dataset, show_details=False)
    sm_fea_s = torch.FloatTensor(np.load(args.fea_sm_path, allow_pickle=True))

    # Initialize the model
    model = my_model(dims=[sm_fea_s.shape[1]] + args.dims).to(device)
    # Load the state dictionary
    model.load_state_dict(torch.load(args.dir + args.save_file_name))

    # Set the model to evaluation mode if needed
    model.eval()
    
    z1, z2 = model(sm_fea_s.to(device))
    embedding = (z1 + z2) / 2
    nmi_list = []
    ari_list = []
    for iter in range(iter_num):
        nmi, ari, _, _, _ = clustering(embedding.detach(), true_labels,cluster_num, device = device)   
        nmi_list.append(nmi)
        ari_list.append(ari)

    nmi_array = np.array(nmi_list)
    ari_array = np.array(ari_list)
    
    return nmi_array,ari_array


if __name__ == "__main__": 
    args = get_args()
    warnings.filterwarnings("ignore")
    log_dir = args.dir
    os.makedirs(log_dir, exist_ok=True)

    # Specify the path to the saved model
    model_path = log_dir + "model.zip"

    # Load the trained model
    model = PPO.load(model_path) 
    env = MyGraphEnv(args.dataset, get_args())
    env = DummyVecEnv([lambda: env])  # Wrap with VecEnv

    # Evaluate the model
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    n_eval_episodes = 10
    # print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    mean_reward, mean_action = evaluate_and_record_average_action(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Number of evaluation eposides {n_eval_episodes}")
    print(f"Mean Reward: {mean_reward}")
    print(f"Average Action: {mean_action[0]}")
    print(f"The number of clusters: {np.round(mean_action[0])+2}")

    iter_num = 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nmi, ari = get_embedding(args,device,cluster_num = np.round(mean_action[0]+2).astype(int),iter_num=iter_num)
    print(f"Average over {iter_num} K-means, on ", args.dataset, " Dataset")
    print(f"Mean of NMI: {nmi.mean()}",f"Std of NMI: {nmi.std()}")
    print(f"Mean of ARI: {ari.mean()}",f"Std of ARI: {ari.std()}")