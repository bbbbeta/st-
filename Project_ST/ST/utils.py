import numba
from munkres import Munkres
from collections import Counter
import sys
import ot
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection
import os
import torch
import random
import numpy as np
import scanpy as sc
from torch.backends import cudnn
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import distance_matrix

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-",Gene3Pattern="mt-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp3 = np.asarray([not str(name).startswith(Gene3Pattern) for name in adata.var_names], dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2,id_tmp3)
    adata._inplace_subset_var(id_tmp)

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)


def refine_nearest_labels(adata, radius=50, key='label'):
    new_type = []
    df = adata.obsm['spatial']
    old_type = adata.obs[key].values
    df = pd.DataFrame(df,index=old_type)
    distances = distance_matrix(df, df)
    distances_df = pd.DataFrame(distances, index=old_type, columns=old_type)

    for index, row in distances_df.iterrows():
        # row[index] = np.inf
        nearest_indices = row.nsmallest(radius).index.tolist()
        # for i in range(1):
        #     nearest_indices.append(index)
        max_type = max(nearest_indices, key=nearest_indices.count)
        new_type.append(max_type)
        # most_common_element, most_common_count = find_most_common_elements(nearest_indices)
        # nearest_labels.append(df.loc[nearest_indices, 'label'].values)

    return [str(i) for i in list(new_type)]

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  #其思想是 按照(row_index, column_index, value)的方式存储每一个非0元素，所以存储的数据结构就应该是一个以三元组为元素的列表List[Tuple[int, int, int]]
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) #from_numpy()用来将数组array转换为张量Tensor vstack（）：按行在下边拼接
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
    sum=0
    for i in range(t1.shape[0]):
        sum+=(t1[i]-t2[i])**2
    return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n=X.shape[0]
    adj=np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j]=euclid_dist(X[i], X[j])
    return adj

def calculate_adj_matrix(adata):
    if "array_row" in adata.obs.columns and "array_col" in adata.obs.columns:
        x = adata.obs["array_row"]
        y = adata.obs["array_col"]
        X=np.array([x, y]).T.astype(np.float32)
    else:
        X=adata.obsm['spatial'].astype(np.float32)
    adj = pairwise_distance(X)
    return adj

def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result

def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)

def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss*-1

def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat

def regularization_loss(emb, adj):
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    loss = torch.mean((mat - adj) ** 2)
    return loss


def self_representation_matrix(emb, temperature=0.1):
    """
    基于自表示（Self-Representation）的相似度矩阵。

    对于嵌入矩阵 Z (N, D)，计算：
        S = softmax(Z @ Z^T / temperature)

    这与 standard cosine similarity + sigmoid 的本质区别在于：
    - cosine_similarity: 每个点与所有点比较，使用全局阈值
    - self_representation: 沿行做 softmax，迫使每行的概率和为1，
      自然形成"软聚类分配"——每行表示该点属于各聚类的概率分布

    自表示矩阵 S 的低秩约束（通过 t-SVD 核范数）会迫使：
    - 三个视图（emb_x, emb_i, z_xi）的自表示矩阵趋向于共享相同的低秩结构
    - 即：三个视图在"哪些点属于同一类"这件事上达成共识

    参数:
        emb: (N, D) 的嵌入矩阵
        temperature: softmax 温度参数，控制分布的锐度
            - 较小值 (0.01-0.1): 分布锐利，稀疏自表示
            - 较大值 (0.5-1.0): 分布平滑，均接近 1/N

    返回:
        S: (N, N) 的自表示矩阵，torch.FloatTensor
    """
    # L2 归一化确保比较的是方向而非量级
    emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
    # 相似度矩阵
    sim = torch.mm(emb_norm, emb_norm.t())  # (N, N)
    # 行-wise softmax，得到自表示矩阵
    S = torch.nn.functional.softmax(sim / temperature, dim=1)
    return S


def tensor_low_rank_loss(emb_x, emb_i, z_xi, temperature=0.1):
    """
    张量低秩正则化损失 —— 多视图聚类一致性约束。

    原理（辩证分析）:

    现有模型已有两种跨视图一致性正则化：
    1. consistency_loss: 在"隐藏维度分布"层面（协方差矩阵）对齐两视图
    2. regularization_loss: 在"样本-样本关系"层面（cos 相似度 vs 邻接矩阵）对齐嵌入空间

    tensor_low_rank_loss 在第三个几何层次提供约束——"聚类结构"层次：

    具体步骤:
    1. 从三个嵌入视图分别构建自表示矩阵 S_x, S_i, S_z
       （每行是 softmax 归一化的相似度分布，表示"该点从谁那里获得信息"）
    2. 堆叠为张量 T = cat(S_x, S_i, S_z, dim=-1)，形状 (N, N, 3)
    3. 沿视图维度做 FFT：T_fft = FFT(T, dim=-1)
       （频域中，不同视图的"信息通道"被分解为谐波分量）
    4. 对每个频域切片做 SVD，计算核范数（奇异值之和）：
       loss = sum_k ||T_fft[:,:,k]||_* / 3
       （核范数 ≈ 矩阵秩的凸近似；最小化核范数 ≈ 强制矩阵低秩）

    效果:
    - 低秩意味着：自表示矩阵只有 k 个主要自由度——恰好对应 k 个聚类
    - 三个视图共享低秩结构 = 三个视图的聚类结构一致
    - 与 consistency_loss 的关系：后者约束协方差（均值+方差），前者约束聚类结构（高阶矩）
    - 与 regularization_loss 的关系：后者约束"嵌入相似度与空间邻接一致"，前者不依赖邻接矩阵，
      从纯嵌入几何出发约束多视图

    为什么用 FFT + 逐切片 SVD（而非直接 3D 展开 SVD）:
    - t-SVD（张量奇异值分解）在频域等价于对每个频域切片独立做 SVD
    - 逐切片 SVD 更稳定、实现更简单，且能捕捉"通道间相位对齐"
    - 实践中与完整 t-SVD 效果相近，但计算更可控

    参数:
        emb_x: (N, D) 基因分支嵌入
        emb_i: (N, D) 图像分支嵌入
        z_xi:  (N, D) 融合分支嵌入
        temperature: 自表示矩阵的 softmax 温度

    返回:
        loss: 标量 torch.Tensor
    """
    # --- 内存安全检查 ---
    n_samples = emb_x.shape[0]
    max_safe_n = 30000  # N×N float32 ≈ 3.6GB，软上限

    if n_samples > max_safe_n:
        # 大数据集策略：降采样到 ~15k 样本
        # 随机选择样本，但保持与原数据一致的随机性
        import numpy as _np
        perm = _np.random.permutation(n_samples)[:max_safe_n]
        emb_x = emb_x[perm]
        emb_i = emb_i[perm]
        z_xi = z_xi[perm]
        n_samples = max_safe_n

    # --- 步骤1: 构建三个视图的自表示矩阵 ---
    S_x = self_representation_matrix(emb_x, temperature=temperature)
    S_i = self_representation_matrix(emb_i, temperature=temperature)
    S_z = self_representation_matrix(z_xi, temperature=temperature)

    # --- 步骤2: 堆叠为 (N, N, 3) 张量 ---
    T = torch.stack([S_x, S_i, S_z], dim=-1)  # (N, N, 3)

    # --- 步骤3: 沿视图维度 FFT ---
    T_fft = torch.fft.fft(T, dim=-1)

    # --- 步骤4: 逐频域切片计算核范数 ---
    n_freq = T_fft.shape[-1]
    total_nuclear = 0.0
    for k in range(n_freq):
        # torch.svd 返回 U, S, V；S 是奇异值向量
        _, s_vals, _ = torch.svd(T_fft[:, :, k])
        total_nuclear += torch.sum(s_vals)

    # 归一化：除以视图数（=3）和频域切片数（=n_freq）
    loss = total_nuclear / (n_freq * 3.0)

    return loss

def refine_label(adata, radius=50, key='cluster'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type

def munkres_newlabel(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return 0,0,0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    print('Counter(new_predict)\n', Counter(new_predict))
    print('Counter(y_true)\n', Counter(y_true))

    return new_predict

def get_neighbors(coords, n_neighbors=6, eps=1e-1):
    k = min(n_neighbors * 2 + 1, len(coords))
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors * 2, len(coords))).fit(coords)
    dists, indices = nbrs.kneighbors(coords)
    dists = dists[:, 1:]; indices = indices[:, 1:]
    neighbors = []
    if len(coords) > 1:
        grid_spacing = np.median(dists[:, 0])
        max_physical_dist = grid_spacing * 1.5
    else:
        max_physical_dist = np.inf

    for i in range(len(coords)):
        min_dist = dists[i].min()
        local_thresh = min_dist * (1 + eps)
        mask = (dists[i] <= local_thresh) & (dists[i] <= max_physical_dist)
        neighbors.append(indices[i][mask])
    return neighbors

def augment_spots_hexagonal(gene_features, image_features, neighbors, coords):
    augmented_gene_list = []
    augmented_image_list = []
    augmented_coords_list = []
    spot_mapping = []
    
    for i in range(len(gene_features)):
        neighbor_indices = neighbors[i]
        n_neighbors = len(neighbor_indices)
        index = 1.0 / 4.0
        for neighbor_idx in neighbor_indices:
            aug_gene = (1-index) * gene_features[i] + index * gene_features[neighbor_idx]
            augmented_gene_list.append(aug_gene)

            # aug_image = image_features[i]
            aug_image = (1-index) * image_features[i] + index * image_features[neighbor_idx]

            augmented_image_list.append(aug_image)
            
            aug_coord = (1-index) * coords[i] + index * coords[neighbor_idx]

            augmented_coords_list.append(aug_coord)
            
            spot_mapping.append(i)
    
    augmented_gene = np.array(augmented_gene_list)
    augmented_image = np.array(augmented_image_list)
    augmented_coords = np.array(augmented_coords_list)
    spot_mapping = np.array(spot_mapping)
    
    augmented_neighbors_count = [len(neighbors[i]) for i in range(len(neighbors))]
    
    return augmented_gene, augmented_image, augmented_coords, spot_mapping, augmented_neighbors_count

def voting_aggregate_clusters(augmented_predictions, spot_mapping, n_original_spots):
    aggregated_labels = np.zeros(n_original_spots, dtype=int)
    
    for i in range(n_original_spots):
        mask = spot_mapping == i
        predictions_for_spot_i = augmented_predictions[mask]
        
        if len(predictions_for_spot_i) > 0:
            counter = Counter(predictions_for_spot_i)
            most_common_label = counter.most_common(1)[0][0]
            aggregated_labels[i] = most_common_label
        else:
            aggregated_labels[i] = 0
    
    return aggregated_labels

def prepare_augmented_data(adata, neighbors, img_key='image_feat'):

    if sp.issparse(adata.X):
        gene_features = adata.X.toarray()
    else:
        gene_features = adata.X.copy()
    
    if img_key in adata.obsm:
        image_features = adata.obsm[img_key]
    else:
        raise ValueError(f"Image features '{img_key}' not found in adata.obsm")
    
    coords = adata.obsm['spatial']
    
    aug_gene, aug_image, aug_coords, spot_mapping, aug_counts = augment_spots_hexagonal(
        gene_features, image_features, neighbors, coords
    )
    
    return {
        'gene_features': aug_gene,
        'image_features': aug_image,
        'coords': aug_coords,
        'spot_mapping': spot_mapping,
        'augmented_counts': aug_counts,
        'n_original_spots': adata.n_obs
    }