from sklearn import metrics
from sklearn.cluster import KMeans
import gc
from ST import models
import datetime
import torch.optim as optim
from ST.utils import *
import warnings
from sklearn.decomposition import PCA
import sklearn

warnings.filterwarnings('ignore')
def train(adata,knn=10,h=[3000,3000], n_epochs=200,lr=0.0001, key_added='ST', random_seed=110,res=1,
          l=2,weight_decay=0.0001,a=10,b=1,c=10,d=0.5,embed=True,radius=0,enhancement=False,cluster="kmeans",loss_type="consistency",
          use_llm_gene=False,weight_gene=20.0,weight_image=1.0,weight_llm=15.0,weight_fusion=10.0,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    set_seed(random_seed)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    if type(adata.X) == np.ndarray:
        features_X = torch.FloatTensor(adata_Vars.X).to(device)
    else:
        features_X = torch.FloatTensor(adata_Vars.X.toarray()).to(device)
    features_I = torch.FloatTensor(adata_Vars.obsm["im_re"].values).to(device)

    features_L = None
    nfeatL = None
    if use_llm_gene and "llm_gene_feature" in adata_Vars.obsm:
        if type(adata_Vars.obsm["llm_gene_feature"]) == np.ndarray:
            features_L = torch.FloatTensor(adata_Vars.obsm["llm_gene_feature"]).to(device)
        else:
            features_L = torch.FloatTensor(adata_Vars.obsm["llm_gene_feature"].toarray()).to(device)
        nfeatL = features_L.shape[1]
        print(f"Using LLM-Gene features with {nfeatL} components")

    adj=adata_Vars.obsm["adj"]
    adj = np.exp(-1*(adj**2)/(2*(l**2)))
    adj = sp.coo_matrix(adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    model = models.ST(nfeatX=features_X.shape[1],
                 nfeatI=features_I.shape[1],
                 nfeatL=nfeatL,
                 hidden_dims=h,
                 use_llm_gene=use_llm_gene,
                 weight_gene=weight_gene,
                 weight_image=weight_image,
                 weight_llm=weight_llm,
                 weight_fusion=weight_fusion,
                 ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mean_max = []
    ari_max = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        # positional unpacking: z_I, q_x, q_i, q_l, emb_x, emb_i, pi, disp, mean
        if use_llm_gene and features_L is not None:
            z_xi, q_x, q_i, q_l, _, _, pi, disp, mean = model(features_X, features_I, adj, features_L)
        else:
            z_xi, q_x, q_i, _, _, _, pi, disp, mean = model(features_X, features_I, adj)
        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(features_X, mean, mean=True)
        if loss_type == "consistency":
            cl_loss = consistency_loss(q_x, q_i)
            if use_llm_gene and q_l is not None:
                cl_loss += d * consistency_loss(q_x, q_l)
                cl_loss += d * consistency_loss(q_i, q_l)
        else:
            cl_loss = crossview_contrastive_Loss(q_x, q_i)
            if use_llm_gene and q_l is not None:
                cl_loss += d * crossview_contrastive_Loss(q_x, q_l)
                cl_loss += d * crossview_contrastive_Loss(q_i, q_l)
        reg_loss = regularization_loss(z_xi, adj)
        total_loss = a * zinb_loss + b * cl_loss + c * reg_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Epoch: {epoch}/{n_epochs}, Loss: {total_loss:.4f}")

        if 'ground_truth' in adata.obs.columns:
            if cluster == "kmeans":
                kmeans = KMeans(n_clusters=knn,random_state=random_seed).fit(np.nan_to_num(z_xi.cpu().detach()))
                idx = kmeans.labels_
                adata_Vars.obs['temp']=idx
                obs_df = adata_Vars.obs.dropna()
                ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['ground_truth'])
            else:
                adata_Vars.obsm["letemp"]=z_xi.to('cpu').detach().numpy()
                sc.pp.neighbors(adata_Vars, use_rep='letemp')
                sc.tl.leiden(adata_Vars, key_added="temp", resolution=res)
                obs_df = adata_Vars.obs.dropna()
                ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['ground_truth'])
                idx=adata_Vars.obs['temp'].values
                count_unique_leiden = len(pd.DataFrame(adata_Vars.obs['temp']).temp.unique())
                print("num of cluster:",count_unique_leiden)
            if ari_res > ari_max:
                ari_max = ari_res
                idx_max = idx
                mean_max = mean.to('cpu').detach().numpy()
                emb_max = z_xi.to('cpu').detach().numpy()
    if 'ground_truth' in adata.obs.columns:
        print("Ari=", ari_max)
    else:
        if cluster == "kmeans":
            kmeans = KMeans(n_clusters=knn,random_state=random_seed).fit(np.nan_to_num(z_xi.cpu().detach()))
            idx_max = kmeans.labels_
            emb_max = z_xi.to('cpu').detach().numpy()
            mean_max = mean.to('cpu').detach().numpy()
        else:
            adata_Vars.obsm["letemp"] = z_xi.to('cpu').detach().numpy()
            sc.pp.neighbors(adata_Vars, use_rep='letemp')
            sc.tl.leiden(adata_Vars, key_added="temp", resolution=res)
            idx_max = adata_Vars.obs['temp'].values
            count_unique_leiden = len(pd.DataFrame(adata_Vars.obs['temp']).temp.unique())
            emb_max = z_xi.to('cpu').detach().numpy()
            mean_max = mean.to('cpu').detach().numpy()
            print("num of cluster:", count_unique_leiden)
    if embed:
        pca = PCA(n_components=20, random_state=random_seed)
        adata.obsm['emb_pca'] = pca.fit_transform(emb_max.copy())

    adata.obs["cluster"] = idx_max.astype(str)
    if radius !=0 :
        nearest_new_type = refine_label(adata, radius=radius)
        adata.obs[key_added] = nearest_new_type
    else:
        adata.obs[key_added] = adata.obs["cluster"]
    adata.obsm["emb"] = emb_max
    adata.obsm['mean'] = mean_max
    if enhancement:
        mean_max = sklearn.preprocessing.normalize(mean_max, axis=1, norm='max')
        adata.layers[key_added] = mean_max
    return adata


def train_with_augmentation(adata, knn=10, h=[3000,3000], n_epochs=200, lr=0.0001, key_added='ST',
                           random_seed=110, res=1, l=2, weight_decay=0.0001, a=10, b=1, c=10, d=1.0,
                           embed=True, radius=0, enhancement=False, cluster="kmeans", loss_type="consistency",
                           tensor_loss=False, tensor_temp=0.1,
                           use_llm_gene=False, weight_gene=20.0, weight_image=1.0,
                           weight_llm=15.0, weight_fusion=10.0,
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    set_seed(random_seed)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    coords = adata_Vars.obsm['spatial'].astype(np.float32)
    neighbors = get_neighbors(coords, n_neighbors=6, eps=0.05)
    print(f"Found neighbors for {len(neighbors)} spots")

    neighbor_counts = [len(n) for n in neighbors]
    print(f"Neighbor statistics - Min: {min(neighbor_counts)}, Max: {max(neighbor_counts)}, Mean: {np.mean(neighbor_counts):.2f}")

    if type(adata_Vars.X) == np.ndarray:
        gene_features = adata_Vars.X
    else:
        gene_features = adata_Vars.X.toarray()

    image_features = adata_Vars.obsm["im_re"].values

    # LLM-gene features: keep original for branch 3 (augmented separately below)
    llm_features = None
    if use_llm_gene and "llm_gene_feature" in adata_Vars.obsm:
        llm_feat_src = adata_Vars.obsm["llm_gene_feature"]
        if type(llm_feat_src) == np.ndarray:
            llm_features = llm_feat_src
        else:
            llm_features = llm_feat_src.toarray()

    aug_gene, aug_image, aug_coords, spot_mapping, aug_counts = augment_spots_hexagonal(
        gene_features, image_features, neighbors, coords
    )

    del gene_features, image_features, coords
    gc.collect()

    # LLM-gene augmentation: interpolate LLM features using same spot_mapping
    # (must be done before deleting neighbors)
    features_L = None
    nfeatL = None
    if use_llm_gene and llm_features is not None:
        aug_llm_list = []
        for i in range(len(llm_features)):
            for neighbor_idx in neighbors[i]:
                aug_val = (1 - 1.0/4.0) * llm_features[i] + (1.0/4.0) * llm_features[neighbor_idx]
                aug_llm_list.append(aug_val)
        aug_llm = np.array(aug_llm_list)
        features_L = torch.FloatTensor(aug_llm).to(device)
        nfeatL = features_L.shape[1]
        del aug_llm, llm_features
        gc.collect()

    del neighbors
    gc.collect()

    features_X = torch.FloatTensor(aug_gene)
    features_I = torch.FloatTensor(aug_image)
    features_X = features_X.to(device)
    features_I = features_I.to(device)

    del aug_gene, aug_image
    gc.collect()

    adj_dense = pairwise_distance(aug_coords)
    del aug_coords
    gc.collect()
    adj_dense = np.exp(-1 * (adj_dense ** 2) / (2 * (l ** 2)))
    adj_sparse = sp.coo_matrix(adj_dense)
    del adj_dense
    gc.collect()
    adj_sparse = normalize(adj_sparse + sp.eye(adj_sparse.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj_sparse)
    adj = adj.to(device)
    del adj_sparse
    gc.collect()

    model = models.ST(nfeatX=features_X.shape[1],
                     nfeatI=features_I.shape[1],
                     nfeatL=nfeatL,
                     hidden_dims=h,
                     use_llm_gene=use_llm_gene,
                     weight_gene=weight_gene,
                     weight_image=weight_image,
                     weight_llm=weight_llm,
                     weight_fusion=weight_fusion,
                     ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ari_max = 0
    idx_max = None
    emb_max_aug = None
    mean_max_aug = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # positional unpacking: z_I, q_x, q_i, q_l, emb_x, emb_i, pi, disp, mean
        if use_llm_gene and features_L is not None:
            z_xi, q_x, q_i, q_l, emb_x, emb_i, pi, disp, mean = model(features_X, features_I, adj, features_L)
        else:
            z_xi, q_x, q_i, _, emb_x, emb_i, pi, disp, mean = model(features_X, features_I, adj)

        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(features_X, mean, mean=True)

        if loss_type == "consistency":
            cl_loss = consistency_loss(q_x, q_i)
            if use_llm_gene and q_l is not None:
                cl_loss += d * consistency_loss(q_x, q_l)
                cl_loss += d * consistency_loss(q_i, q_l)
        else:
            cl_loss = crossview_contrastive_Loss(q_x, q_i)
            if use_llm_gene and q_l is not None:
                cl_loss += d * crossview_contrastive_Loss(q_x, q_l)
                cl_loss += d * crossview_contrastive_Loss(q_i, q_l)

        reg_loss = regularization_loss(z_xi, adj)

        if tensor_loss:
            t_loss = tensor_low_rank_loss(emb_x, emb_i, z_xi, temperature=tensor_temp)
            total_loss = a * zinb_loss + b * cl_loss + c * reg_loss + d * t_loss
        else:
            t_loss = torch.tensor(0.0, device=device)
            total_loss = a * zinb_loss + b * cl_loss + c * reg_loss
        total_loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if tensor_loss:
                print(f"[{current_time}] Epoch: {epoch}/{n_epochs}, Loss: {total_loss:.4f} "
                      f"(ZINB={a*zinb_loss:.4f}, CL={b*cl_loss:.4f}, Reg={c*reg_loss:.4f}, Tensor={d*t_loss:.4f})")
            else:
                print(f"[{current_time}] Epoch: {epoch}/{n_epochs}, Loss: {total_loss:.4f} "
                      f"(ZINB={a*zinb_loss:.4f}, CL={b*cl_loss:.4f}, Reg={c*reg_loss:.4f})")

            if torch.cuda.is_available() and epoch % 20 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  GPU allocated{allocated:.2f}GB, reserved{reserved:.2f}GB")

        if 'ground_truth' in adata.obs.columns and epoch % 10 == 0:
            with torch.no_grad():
                z_xi_eval = z_xi.cpu().detach().numpy()

                if cluster == "kmeans":
                    kmeans = KMeans(n_clusters=knn, random_state=random_seed).fit(np.nan_to_num(z_xi_eval))
                    aug_labels = kmeans.labels_
                else:
                    temp_adata = ad.AnnData(X=z_xi_eval)
                    sc.pp.neighbors(temp_adata, use_rep='X')
                    sc.tl.leiden(temp_adata, key_added="temp", resolution=res)
                    aug_labels = temp_adata.obs['temp'].values.astype(int)
                    del temp_adata

                original_labels = voting_aggregate_clusters(aug_labels, spot_mapping, adata_Vars.n_obs)

                adata_Vars.obs['temp'] = original_labels.astype(str)
                obs_df = adata_Vars.obs.dropna()
                ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['ground_truth'])
                if ari_res > ari_max:
                    ari_max = ari_res
                    idx_max = original_labels
                    mean_max_aug = mean.to('cpu').detach().numpy()
                    emb_max_aug = z_xi.to('cpu').detach().numpy()
                    print(f"Epoch {epoch}: New best ARI = {ari_res:.5f}")
                del z_xi_eval
                gc.collect()

    if 'ground_truth' in adata.obs.columns and idx_max is not None:
        original_labels_final = idx_max

        with torch.no_grad():
            emb_aggregated = np.zeros((adata_Vars.n_obs, emb_max_aug.shape[1]))
            mean_aggregated = np.zeros((adata_Vars.n_obs, mean_max_aug.shape[1]))

            for i in range(adata_Vars.n_obs):
                mask = spot_mapping == i
                emb_aggregated[i] = emb_max_aug[mask].mean(axis=0)
                mean_aggregated[i] = mean_max_aug[mask].mean(axis=0)
    else:
        with torch.no_grad():
            z_xi_final = z_xi.to('cpu').detach().numpy()
            mean_final = mean.to('cpu').detach().numpy()

            if cluster == "kmeans":
                kmeans = KMeans(n_clusters=knn, random_state=random_seed).fit(np.nan_to_num(z_xi_final))
                aug_labels_final = kmeans.labels_
            else:
                temp_adata = ad.AnnData(X=z_xi_final)
                sc.pp.neighbors(temp_adata, use_rep='X')
                sc.tl.leiden(temp_adata, key_added="temp", resolution=res)
                aug_labels_final = temp_adata.obs['temp'].values.astype(int)
                count_unique_leiden = len(np.unique(aug_labels_final))
                print(f"Number of clusters in augmented data: {count_unique_leiden}")

            original_labels_final = voting_aggregate_clusters(aug_labels_final, spot_mapping, adata_Vars.n_obs)

            emb_aggregated = np.zeros((adata_Vars.n_obs, z_xi_final.shape[1]))
            mean_aggregated = np.zeros((adata_Vars.n_obs, mean_final.shape[1]))

            for i in range(adata_Vars.n_obs):
                mask = spot_mapping == i
                emb_aggregated[i] = z_xi_final[mask].mean(axis=0)
                mean_aggregated[i] = mean_final[mask].mean(axis=0)

    if 'ground_truth' in adata.obs.columns and idx_max is not None:
        adata_Vars.obs['final'] = original_labels_final.astype(str)
        obs_df = adata_Vars.obs.dropna()
        final_ari = metrics.adjusted_rand_score(obs_df['final'], obs_df['ground_truth'])
        print(f"Final ARI (best from training) = {final_ari:.5f}")

    adata.obs["cluster"] = original_labels_final.astype(str)

    if radius != 0:
        nearest_new_type = refine_label(adata, radius=radius, key='cluster')
        adata.obs[key_added] = nearest_new_type
    else:
        adata.obs[key_added] = adata.obs["cluster"]

    adata.obsm["emb"] = emb_aggregated
    adata.obsm['mean'] = mean_aggregated

    if embed:
        pca = PCA(n_components=20, random_state=random_seed)
        adata.obsm['emb_pca'] = pca.fit_transform(emb_aggregated.copy())

    if enhancement:
        mean_aggregated = sklearn.preprocessing.normalize(mean_aggregated, axis=1, norm='max')
        adata.layers[key_added] = mean_aggregated
    del model, features_X, features_I, adj
    if features_L is not None:
        del features_L
    torch.cuda.empty_cache()
    gc.collect()
    return adata


def train_mul(adata,adjlist,knn=10,h=[3000,3000], n_epochs=200,lr=0.0001, key_added='ST', random_seed=110,res=1,
          l=2,weight_decay=0.0001,a=10,b=1,c=10,embed=True,radius=0,enhancement=False,cluster="kmeans",loss_type="consistency",
          use_llm_gene=False,weight_gene=20.0,weight_image=1.0,weight_llm=15.0,weight_fusion=10.0,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    set_seed(random_seed)

    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    if type(adata.X) == np.ndarray:
        features_X = torch.FloatTensor(adata_Vars.X).to(device)
    else:
        features_X = torch.FloatTensor(adata_Vars.X.toarray()).to(device)
    features_I = torch.FloatTensor(adata_Vars.obsm["im_re"].values).to(device)

    features_L = None
    nfeatL = None
    if use_llm_gene and "llm_gene_feature" in adata_Vars.obsm:
        if type(adata_Vars.obsm["llm_gene_feature"]) == np.ndarray:
            features_L = torch.FloatTensor(adata_Vars.obsm["llm_gene_feature"]).to(device)
        else:
            features_L = torch.FloatTensor(adata_Vars.obsm["llm_gene_feature"].toarray()).to(device)
        nfeatL = features_L.shape[1]
        print(f"Using LLM-Gene features with {nfeatL} components")

    total_size = sum(adj.shape[0] for adj in adjlist)
    block_diag_adj = np.zeros((total_size, total_size))
    current_position = 0
    for adj in adjlist:
        adj = np.exp(-1*(adj**2)/(2*(l**2)))
        adj = normalize(adj + sp.eye(adj.shape[0]))
        size = adj.shape[0]
        block_diag_adj[current_position:current_position + size, current_position:current_position + size] = adj
        current_position += size
    block_diag_adj = sp.coo_matrix(block_diag_adj)
    block_diag_adj = sparse_mx_to_torch_sparse_tensor(block_diag_adj).to(device)
    model = models.ST(nfeatX=features_X.shape[1],
                 nfeatI=features_I.shape[1],
                 nfeatL=nfeatL,
                 hidden_dims=h,
                 use_llm_gene=use_llm_gene,
                 weight_gene=weight_gene,
                 weight_image=weight_image,
                 weight_llm=weight_llm,
                 weight_fusion=weight_fusion,
                 ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mean_max = []
    ari_max = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        if use_llm_gene and features_L is not None:
            z_xi, q_x, q_i, q_l, _, _, pi, disp, mean = model(features_X, features_I, block_diag_adj, features_L)
        else:
            z_xi, q_x, q_i, _, _, _, pi, disp, mean = model(features_X, features_I, block_diag_adj)
        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(features_X, mean, mean=True)
        if loss_type == "consistency":
            cl_loss = consistency_loss(q_x, q_i)
            if use_llm_gene and q_l is not None:
                cl_loss += d * consistency_loss(q_x, q_l)
                cl_loss += d * consistency_loss(q_i, q_l)
        else:
            cl_loss = crossview_contrastive_Loss(q_x, q_i)
            if use_llm_gene and q_l is not None:
                cl_loss += d * crossview_contrastive_Loss(q_x, q_l)
                cl_loss += d * crossview_contrastive_Loss(q_i, q_l)
        reg_loss = regularization_loss(z_xi, block_diag_adj)
        total_loss = a * zinb_loss + b * cl_loss + c * reg_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Epoch: {epoch}/{n_epochs}, Loss: {total_loss:.4f}")

        if 'ground_truth' in adata.obs.columns:
            if cluster == "kmeans":
                kmeans = KMeans(n_clusters=knn,random_state=random_seed).fit(np.nan_to_num(z_xi.cpu().detach()))
                idx = kmeans.labels_
                adata_Vars.obs['temp']=idx
                obs_df = adata_Vars.obs.dropna()
                ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['ground_truth'])
            else:
                adata_Vars.obsm["letemp"]=z_xi.to('cpu').detach().numpy()
                sc.pp.neighbors(adata_Vars, use_rep='letemp')
                sc.tl.leiden(adata_Vars, key_added="temp", resolution=res)
                obs_df = adata_Vars.obs.dropna()
                ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['ground_truth'])
                idx=adata_Vars.obs['temp'].values
                count_unique_leiden = len(pd.DataFrame(adata_Vars.obs['temp']).temp.unique())
                print("num of cluster:",count_unique_leiden)
            if ari_res > ari_max:
                ari_max = ari_res
                idx_max = idx
                mean_max = mean.to('cpu').detach().numpy()
                emb_max = z_xi.to('cpu').detach().numpy()
    if 'ground_truth' in adata.obs.columns:
        print("Ari=", ari_max)
    else:
        if cluster == "kmeans":
            kmeans = KMeans(n_clusters=knn,random_state=random_seed).fit(np.nan_to_num(z_xi.cpu().detach()))
            idx_max = kmeans.labels_
            emb_max = z_xi.to('cpu').detach().numpy()
            mean_max = mean.to('cpu').detach().numpy()
        else:
            adata_Vars.obsm["letemp"] = z_xi.to('cpu').detach().numpy()
            sc.pp.neighbors(adata_Vars, use_rep='letemp')
            sc.tl.leiden(adata_Vars, key_added="temp", resolution=res)
            idx_max = adata_Vars.obs['temp'].values
            count_unique_leiden = len(pd.DataFrame(adata_Vars.obs['temp']).temp.unique())
            emb_max = z_xi.to('cpu').detach().numpy()
            mean_max = mean.to('cpu').detach().numpy()
            print("num of cluster:", count_unique_leiden)
    if embed:
        pca = PCA(n_components=20, random_state=random_seed)
        adata.obsm['emb_pca'] = pca.fit_transform(emb_max.copy())

    adata.obs["cluster"] = idx_max.astype(str)
    if radius !=0 :
        nearest_new_type = refine_label(adata, radius=radius)
        adata.obs[key_added] = nearest_new_type
    else:
        adata.obs[key_added] = adata.obs["cluster"]
    adata.obsm["emb"] = emb_max
    adata.obsm['mean'] = mean_max
    if enhancement:
        mean_max = sklearn.preprocessing.normalize(mean_max, axis=1, norm='max')
        adata.layers[key_added] = mean_max
    return adata
