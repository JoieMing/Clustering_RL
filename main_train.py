import os
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.argument import get_args
from env import MyGraphEnv



# Function to plot rewards with mean and std
def plot_reward_convergence(mean_rewards, std_rewards, save_path=None):
    # Ensure inputs are numpy arrays for safe computation
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)

    # Generate X-axis (episodes)
    episodes = np.arange(1, len(mean_rewards) + 1)

    # Plot using Seaborn and Matplotlib
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")  # Use Seaborn's dark grid style
    sns.lineplot(x=episodes, y=mean_rewards, label="Mean Reward", color="blue")
    plt.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="blue",
        alpha=0.3,
        label="Reward Std Dev",
    )
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.title("Training Reward Convergence", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_average_reward(mean_rewards, save_path):
    """
    绘制平均奖励
    """
    episodes = np.arange(1, len(mean_rewards) + 1)
    plt.plot(episodes, mean_rewards, label='PPO')
    plt.axhline(y = 1, linestyle = '--', color='lightblue', label='Optimal')
    plt.xlim(0, episodes)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend(fontsize=12)
    plt.grid(True)
    if not os.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path)
    plt.show()


def plot_loss(steps, losses, save_path):
    """
    Plot loss against steps using seaborn.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=steps, y=losses, label="Loss", color="red")
    plt.title("Loss vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create and check the custom environment
    args = get_args()
    env = MyGraphEnv('cora', args)

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps = args.step_num * args.episode_num) 

    # Load recorded rewards and plot
    mean_rewards = np.load(os.path.join(log_dir, "mean_rewards.npy"))
    # std_rewards = np.load(os.path.join(log_dir, "std_rewards.npy"))

    # plot_reward_convergence(
    # mean_rewards=mean_rewards,
    # std_rewards=std_rewards,
    # save_path=log_dir + "reward_convergence_plot.png"
    # )
    plot_average_reward(mean_rewards=mean_rewards, save_path=log_dir + "reward_convergence_plot.png")


    loss_MLP = np.load(os.path.join(log_dir, "loss_MLP.npy"))
    steps = np.arange(len(loss_MLP))
    plot_loss(steps, loss_MLP, log_dir + "/loss_plot.png")

    

