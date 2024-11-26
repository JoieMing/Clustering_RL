import os
import warnings
from stable_baselines3 import PPO  # Replace PPO with your algorithm, e.g., A2C, DQN, etc.
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from model import my_model
from src.argument import get_args
from env import MyGraphEnv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from utils import *
from src.argument import get_args


def get_embedding(args,device,cluster_num):
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
    nmi, ari, _, _, _ = clustering(embedding.detach(), true_labels,cluster_num, device = device)   
    
    return nmi,ari

def evaluate_and_record_metrics(model, env, args, n_eval_episodes=10, deterministic=True):
    """
    Evaluate a model and compute the mean and standard deviation of NMI and ARI over episodes.

    Args:
        model: The trained Stable-Baselines3 model.
        env: The VecEnv environment.
        args: Argument object containing model and dataset configurations.
        n_eval_episodes (int): Number of episodes to evaluate.
        steps_per_episode (int): Number of steps per episode.
        deterministic (bool): Whether to use deterministic actions.

    Returns:
        avg_nmi (float): Average NMI across episodes.
        std_nmi (float): Standard deviation of NMI across episodes.
        avg_ari (float): Average ARI across episodes.
        std_ari (float): Standard deviation of ARI across episodes.
    """
    nmi_values = []
    ari_values = []
    actions = []
    cluster_nums = []

    for episode in range(n_eval_episodes):
        obs = env.reset()  # Reset environment
        done = False
        step_count = 0

        while not done:
            # Predict the action
            action, _states = model.predict(obs, deterministic=deterministic)

            # Take the action in the environment
            obs, reward, done, _ = env.step(action)
            done = step_count >= args.step_num - 1  # End the episode after 20 steps
            step_count += 1
            last_action = action  # Save the last action of the episode

        # Calculate NMI and ARI for this episode using the final action
        cluster_num = int(np.round(last_action[0]) + 2)  # Convert action to cluster number
        nmi, ari = get_embedding(args, device="cuda:0" if torch.cuda.is_available() else "cpu", cluster_num=cluster_num)

        nmi_values.append(nmi)
        ari_values.append(ari)
        actions.append(last_action)
        cluster_nums.append(cluster_num)

    # Calculate mean and std for NMI and ARI
    avg_action = np.mean(actions)
    avg_cluster_num = np.mean(cluster_nums)
    avg_nmi = np.mean(nmi_values)
    std_nmi = np.std(nmi_values)
    avg_ari = np.mean(ari_values)
    std_ari = np.std(ari_values)

    return avg_cluster_num ,avg_action, avg_nmi, std_nmi, avg_ari, std_ari


if __name__ == "__main__": 
    args = get_args()
    warnings.filterwarnings("ignore")
    log_dir = args.dir

    # Specify the path to the saved model
    model_path = log_dir + "model.zip"

    # Load the trained model
    model = PPO.load(model_path)
    env = MyGraphEnv('amap', get_args())
    env = DummyVecEnv([lambda: env])  # Wrap with VecEnv

    # Evaluate the model
    avg_cluster_num ,avg_action, avg_nmi, std_nmi, avg_ari, std_ari = evaluate_and_record_metrics(model, env, args, n_eval_episodes=5)

    print(f"Average NMI: {avg_nmi}, Standard Deviation NMI: {std_nmi}")
    print(f"Average ARI: {avg_ari}, Standard Deviation ARI: {std_ari}")
    print(f"Average Action: {avg_action}")
    print(f"Average Cluster Number: {avg_cluster_num}")
