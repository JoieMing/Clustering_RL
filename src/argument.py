import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
    parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")


    #training parameters
    parser.add_argument('--step_num', type=int, default=100)  # 回合数
    parser.add_argument('--episode_num', type=int, default=20)  # 回合数


    args = parser.parse_args()
    args.cluster_num = 7
    args.gnnlayers = 2
    args.lr = 1e-3
    args.n_input = 500
    args.dims = [1500]
    args.epsilon = 0.5
    args.replay_buffer_size = 40
    args.adj_path = "dataset/{}/{}_adj_1st_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    args.fea_sm_path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)

    args.model_settings = {
        ## need to be modified for PPO
        "n_input": 1000,
        "dims": [500],
        "lr": 1e-3,
        "Q_epochs": 30,
        "epsilon": 0.5,
        "replay_buffer_size": 50,
        "Q_lr": 1e-3,
        "E_epochs": 400
    }

    return args

