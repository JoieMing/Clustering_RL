from src.argument import get_args
from sklearn.decomposition import PCA
import numpy as np
import torch
from utils import *
import os

args = get_args()

X, y, A = load_graph_data(args.dataset, show_details=False)
features = X
true_labels = y
adj = sp.csr_matrix(A)
# reduce the feature dimensions to args.n_input
if args.n_input != -1:
    pca = PCA(n_components=args.n_input)
    features = pca.fit_transform(features)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
A = torch.tensor(adj.todense()).float().to(device)

adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
print('Laplacian Smoothing...')
adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
sm_fea_s = sp.csr_matrix(features).toarray()

path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
# if os.path.exists(path):
#     sm_fea_s = np.load(path, allow_pickle=True)
# else:
for a in adj_norm_s:
    sm_fea_s = a.dot(sm_fea_s)
np.save(path, sm_fea_s, allow_pickle=True)

# X
# sm_fea_s = torch.FloatTensor(sm_fea_s)
adj_1st = (adj + sp.eye(adj.shape[0])).toarray()
path = "dataset/{}/{}_adj_1st_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
# if os.path.exists(path):
#     adj_1st = np.load(path, allow_pickle=True)
# else:
for a in adj_norm_s:
    adj_1st = a.dot(adj_1st)
np.save(path, adj_1st, allow_pickle=True)