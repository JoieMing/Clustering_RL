import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
    parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")


    #training parameters
    parser.add_argument('--step_num', type=int, default=40)  # 回合数   
    parser.add_argument('--episode_num', type=int, default=10)  # 回合数
    # parser.add_argument('--max_steps', type=int, default=200)  # 最大步数

    # E net
    parser.add_argument('--E_epochs', type=int, default=200, help='Number of epochs to train E.')
    parser.add_argument('--n_input', type=int, default=1000, help='Number of units in hidden layer 1.')
    parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--issave', type = bool, default = True, help='whether save the MLP model.')
    parser.add_argument('--dir', type=str, default = "./logs/1125_400/",help= "save directory for MLP model.")
    parser.add_argument('--save_file_name', type=str, default="MLP.pth", help='save file name for MLP model.')


    args = parser.parse_args()
    args.cluster_num = 7
    args.gnnlayers = 2
    args.adj_path = "dataset/{}/{}_adj_1st_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    args.fea_sm_path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)

    args.learning_settings = {
        'progress_bar': True,
        'reset_num_timesteps': True,
        'total_timesteps': args.episode_num * args.step_num,
        'log_interval': 10,
    }

    # args.model_settings = {
    #     ## need to be modified for PPO
    #     "n_input": 1000,
    #     "dims": [500],
    #     "lr": 1e-3,
    #     "Q_epochs": 30,
    #     "epsilon": 0.5,
    #     "replay_buffer_size": 50,
    #     "Q_lr": 1e-3,
    # }

    return args

    # n_steps = 40, eposide = 10, step = 50