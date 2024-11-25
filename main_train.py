import os
import warnings
import numpy as np
import seaborn as sns
import tqdm
# import rich
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
    # print(episodes)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards,'o-', label='PPO')
    # plt.axhline(y = 1, linestyle = '--', color='lightblue', label='Optimal')
    plt.xlim(0, len(episodes))
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend(fontsize=12)
    plt.grid(True)
    # if not os._exists(save_path):
    #     os.makedirs(save_path)
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
    

    # Create and check the custom environment
    args = get_args()
    env = MyGraphEnv('cora', get_args())
    
    warnings.filterwarnings("ignore")
    # log_dir = "./logs/1124/"
    log_dir = args.dir
    os.makedirs(log_dir, exist_ok=True)

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, n_steps = 10)
    # model = PPO("MultiInputPolicy", env, verbose=1, n_steps = 30)
    model.learn(**args.learning_settings) 

    # save the model
    model.save(log_dir + "model")
    # save value in the env
    np.save(os.path.join(log_dir, "loss_MLP.npy"), np.array(env.loss_MLP))
    np.save(os.path.join(log_dir, "mean_rewards.npy"), np.array(env.mean_rewards))
    np.save(os.path.join(log_dir, "action.npy"), np.array(env.actions))
    # print the best_nmi and best_ari and cluster_num
    print("Optimization Finished!")
    print('best_nmi: {}, best_ari: {}, cluster_num: {}'.format(env.best_nmi, env.best_ari, env.best_cluster))
    # open file result.csv and write down the dataset name(first line)
    file_name = "result.csv"
    file = open(file_name, "a+")
    print(args.dataset, file=file)
    # print the value of key for the best cluster, best nmi, best ari in info
    print(env.best_cluster, env.best_nmi, env.best_ari, file=file)
    file.close()

    # close the environment
    env.close()

    # Load recorded rewards and plot
    mean_rewards = np.load(os.path.join(args.dir, "mean_rewards.npy"))
    # std_rewards = np.load(os.path.join(log_dir, "std_rewards.npy"))
    actions = np.load(os.path.join(args.dir, "action.npy"))
    steps = np.arange(len(actions))
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(steps, actions, alpha=0.7, edgecolors='k')
    plt.title("Scatter Plot of Data")
    plt.xlabel("Steps")
    plt.ylabel("Cluster Number Decision")
    plt.grid(True)
    plt.savefig("log.png")

    # plot_reward_convergence(
    # mean_rewards=mean_rewards,
    # std_rewards=std_rewards,
    # save_path=log_dir + "reward_convergence_plot.png"
    # )
    print(mean_rewards)
    plot_average_reward(mean_rewards=mean_rewards, save_path=args.dir + "reward_convergence_plot.png")
    

    loss_MLP = np.load(os.path.join(args.dir, "loss_MLP.npy"))
    steps = np.arange(len(loss_MLP))
    plot_loss(steps, loss_MLP, args.dir + "/loss_plot.png")

    
