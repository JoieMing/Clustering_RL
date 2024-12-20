import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
    parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")


    #training parameters
    parser.add_argument('--step_num', type=int, default=20)  # 回合数
    parser.add_argument('--episode_num', type=int, default=100)  # 回合数
    parser.add_argument('--max_steps', type=int, default=2000)  # 最大步数

    # E net
    parser.add_argument('--E_epochs', type=int, default=400, help='Number of epochs to train E.')
    parser.add_argument('--n_input', type=int, default=1000, help='Number of units in hidden layer 1.')
    parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')


    args = parser.parse_args()
    args.cluster_num = 7
    args.gnnlayers = 2
    args.adj_path = "dataset/{}/{}_adj_1st_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    args.fea_sm_path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)

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

