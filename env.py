# create custom environment for the agent to interact with by gym
import os
import torch
from torch import optim
import warnings
import torch.nn.functional as F
from torch_scatter import scatter
from utils import *
from tqdm import tqdm
from torch_scatter import scatter
import gymnasium as gym
from gymnasium import spaces
from src.argument import get_args
from model import my_model
# from sklearn.decomposition import PCA
from stable_baselines3.common.env_checker import check_env


# create custom environment class
class MyGraphEnv(gym.Env):
    def __init__(self, dataset, args):
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        super().__init__()
        self.time = 0
        self.dataset = dataset
        self.args = get_args()
        self.loss_MLP = []

        # Load data
        self.features, self.true_labels, self.A = load_graph_data(self.dataset, show_details=False)
        self.adj_1st = np.load(args.adj_path, allow_pickle=True)
        self.sm_fea_s = torch.FloatTensor(np.load(args.fea_sm_path, allow_pickle=True))
        # self.sm_fea_s = torch.FloatTensor(self.sm_fea_s)

        # Gym Spaces
        self.state_space_dim = self.features.shape[1]
        self.action_space_dim = 9  # Assuming the action range is fixed as in the script
        self.action_space = spaces.Discrete(self.action_space_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2719,500), dtype=np.float32)

        # self.observation_space = spaces.Dict({
        #     "continuous": spaces.Box(low= -np.inf, high= np.inf, shape=(2719,500), dtype=np.float32),  # Continuous space
        #     "discrete": spaces.Discrete(self.action_space_dim),  # Discrete space
        # })


        # Initialize model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = my_model([self.sm_fea_s.shape[1]] + self.args.dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.best_nmi = 0
        self.best_ari = 0
        self.best_cluster = 0
        self.records_r = []
        self.mean_rewards = []
        self.actions = []
        self.issave = args.issave
        self.save_path = args.dir+ args.save_file_name

        # self.std_rewards = []

        self.reset()


    def _compute_infoNEC_loss(self,z1, z2, mask, target):
        # Combine embeddings
        z1_z2 = torch.cat([z1, z2], dim=0)
        
        # Cosine similarity matrix
        S = z1_z2 @ z1_z2.T

        # Positive and negative weights
        pos_neg = mask * torch.exp(S)
        pos = torch.cat([torch.diag(S, target.shape[0]), torch.diag(S, -target.shape[0])], dim=0)
        pos = torch.exp(pos)
        neg = (torch.sum(pos_neg, dim=1) - pos)

        # InfoNEC loss
        infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * target.shape[0])
        # Combine state
        state = (z1 + z2) / 2   # state: the embedding

        return infoNEC, state
    
    def _compute_clustering_loss(self,dis): 
        # print(dis)
        # print(dis.sum(-1).reshape(-1, 1))
        new_dis = (dis + 1).pow(-1)
        q = new_dis / (new_dis.sum(-1).reshape(-1, 1))
        # print(q)
        p = q.pow(2) / q.sum(0).reshape(1, -1)
        # print(p.shape)
        p = p / p.sum(-1).reshape(-1, 1)
        # print(p.shape)
        pq_loss = F.kl_div(q.log(), p)
        
        return pq_loss
    
    def _train_MLP(self,target):
        self.model.train()
        self.optimizer.zero_grad()
        z1, z2 = self.model(self.sm_fea_s.to(self.device)) # z1, z2 embedding
        # compute loss function
        mask = torch.ones([target.shape[0] * 2, target.shape[0] * 2]).to(self.device)
        mask -= torch.diag_embed(torch.diag(mask))
        infoNEC, state = self._compute_infoNEC_loss(z1, z2, mask, target)
        # compute the mean feature state of the data points in that cluster.
        cluster_state = scatter(state, torch.tensor(self.predict_labels).to(self.device), dim=0, reduce="mean")
        return state, cluster_state, infoNEC
    
    def _two_view_MLP(self,action,alpha=10):
        target = torch.FloatTensor(self.adj_1st).to(self.device)
        args = self.args
        cur_state, _, infoNEC = self._train_MLP(target)

        args.cluster_num = action + 2       # the number of clusters from 2 to 11
        nmi, ari, self.predict_labels, centers, dis = clustering(cur_state.detach(), self.true_labels, args.cluster_num, device=self.device)   
        dis = (cur_state.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1)

        # compute the clustering guidance loss function (MLP)
        pq_loss = self._compute_clustering_loss(dis)
        loss = infoNEC + alpha * pq_loss     # alpha is 10 in the paper.
        loss.backward()
        self.optimizer.step()

        # take action and observe environment (next state)
        self.model.eval()
        z1, z2 = self.model(self.sm_fea_s.to(self.device))
        next_state = (z1 + z2) / 2
        next_cluster_state = scatter(next_state, torch.tensor(self.predict_labels).to(self.device), dim=0, reduce="mean")
        center_dis = (centers.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1).mean()
        observation_ini = np.concatenate([next_state.cpu().detach().numpy(), next_cluster_state.cpu().detach().numpy()], axis=0)
        rows_to_add = 2719 - observation_ini.shape[0]
        observation = np.pad(observation_ini, pad_width=((0, rows_to_add), (0, 0)), mode='constant', constant_values=-1)
        # observation = {}
        # observation["continuous"] = np.pad(observation_ini, pad_width=((0, rows_to_add), (0, 0)), mode='constant', constant_values=-1)
        # observation["discrete"] = action
        # reward = (center_dis.detach() - torch.min(dis, dim=1).values.mean().detach()).item() - pq_loss.item() * 1000
        reward = center_dis.detach() - torch.min(dis, dim=1).values.mean().detach().item()
        # print(a)
        # reward = -(pq_loss.item()) 
        # print(reward)
        if self.issave:
            torch.save(self.model.state_dict(), self.save_path)

        return observation, reward, nmi, ari, loss.item()


    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        # Initialize clustering
        super().reset(seed=seed)
        args = self.args
        self.time = 0
        args.cluster_num = np.random.randint(0, 9) + 2
        nmi, ari, self.predict_labels, _, _ = clustering(self.sm_fea_s.detach(), self.true_labels, args.cluster_num, device=self.device)
        target = torch.FloatTensor(self.adj_1st).to(self.device)
        state, cluster_state, _ = self._train_MLP(target)
        observation_ini = np.concatenate([state.cpu().detach().numpy(), cluster_state.cpu().detach().numpy()], axis=0)
        rows_to_add = 2719 - observation_ini.shape[0]
        observation = np.pad(observation_ini, pad_width=((0, rows_to_add), (0, 0)), mode='constant', constant_values=-1)
        # observation = {}
        # observation["continuous"] = np.pad(observation_ini, pad_width=((0, rows_to_add), (0, 0)), mode='constant', constant_values=-1)
        # observation["discrete"] = args.cluster_num-2
        if self.best_nmi < nmi:
            self.best_nmi = nmi
            self.best_ari = ari
            self.best_cluster = args.cluster_num
        info = {'nmi': self.best_nmi, 'ari': self.best_ari, 'best_cluster': self.best_cluster}
        self.records_r = []
        return observation, info
    
   
    def step(self, action):
        # Execute one time step within the environment
        # Update the state
        args = self.args
        self.time += 1
        # print("time: ", self.timef)
        args.cluster_num = action + 2
        observation, reward, nmi, ari, loss = self._two_view_MLP(action=action)
    
        # Update best metrics
        if nmi > self.best_nmi:
            self.best_nmi = nmi
            self.best_ari = ari
            self.best_cluster = args.cluster_num
        # update the best_nmi, best_ari in info
        info = {'nmi': self.best_nmi, 'ari': self.best_ari, 'best_cluster': self.best_cluster}

        # store the loss in loss_MLP list
        self.loss_MLP.append(loss)
        self.actions.append(args.cluster_num)
        self.records_r.append(reward)

        if self.time >= args.step_num:
            terminated = True
            print(terminated)
            mean = np.mean(self.records_r)
            self.mean_rewards.append(mean)
            # file_name = "reward_1121_4pm.csv"
            # file = open(file_name, "a+")
            # print(reward, mean, file=file)
            # file.close()
            tqdm.write('episode_reward: {}, mean_reward: {}'.format(reward, mean))
        else:
            terminated = False
        
        # if self.time + self.episodes * args.step_num >= args.max_steps:
        #     # open file result.csv and write down the dataset name(first line)
        #     file_name = "result.csv"
        #     file = open(file_name, "a+")
        #     print(args.dataset, file=file)
        #     # print the value of key for the best cluster, best nmi, best ari in info
        #     print(info['best_cluster'], info['nmi'], info['ari'], file=file)
        #     file.close()
        #     # print the best_nmi and best_ari and cluster_num
        #     tqdm.write("Optimization Finished!")
        #     tqdm.write('best_nmi: {}, best_ari: {}, cluster_num: {}'.format(info['nmi'], info['ari'], info['best_cluster']))

        #     # check the address of the file exists if not create
        #     log_dir = "./logs/1121/"
        #     os.makedirs(log_dir, exist_ok=True)
        #     # save the loss_MLP in the npy file
        #     np.save(os.path.join(log_dir, "loss_MLP.npy"), np.array(self.loss_MLP)) 
        #     np.save(os.path.join(log_dir, "mean_rewards.npy"), np.array(self.mean_rewards))
        #     np.save(os.path.join(log_dir, "action.npy"), np.array(self.actions))

            
        # Check termination
        # if nmi >= 0.99:
        #     terminated = True
        # else:
        #     terminated = False
        
        # terminated = nmi >= 0.99  # Custom termination criterion
        truncated = False

        return observation, reward, terminated, truncated, info
    
    def render(self, close=False):
        pass

    def close (self):
        pass


# check the environment
# warnings.filterwarnings("ignore")
# env = MyGraphEnv('cora', get_args())
# check_env(env)