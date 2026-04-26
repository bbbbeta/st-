# 空间转录组学
---

## 概述

### 创新点
1. **引入邻居投票，利用空间连续性先验知识，通过邻居节点的数据增强和投票聚合来提升边界识别的平滑性和准确性。
2. *张量低秩正则化（Tensor Low-Rank Regularization）：在多视图聚类结构层面引入额外的跨视图一致性约束，强制基因分支、图像分支和融合分支的聚类关系矩阵趋于低秩一致。
3. ***LLM辅助基因选择（LLM-Assisted Gene Selection）：利用大语言模型对基因功能知识的理解，从海量基因中筛选出与目标组织高度相关的基因，为模型提供更具生物学意义的高质量特征。实现基因降维

### 用到的库
需要在Linux环境中配置环境。
| 组件 | 技术/库 | 用途 |
|------|---------|------|
| 数据结构 | Scanpy / AnnData | 空间转录组数据管理 |
| 深度学习 | PyTorch | 神经网络模型训练 |
| 聚类 | sklearn (KMeans), scanpy (Leiden) | 空间域聚类 |
| 图计算 | Numba JIT | 高性能距离矩阵计算 |
| 空间分析 | POT (Optimal Transport) | 空间标签精炼 |
| 标签匹配 | munkres (Hungarian) | 聚类标签对齐 |
| 可视化 | matplotlib, scanpy | 结果可视化 |

### 测试用数据集

目前测试中用到的数据集如下。其中，10x Visium探针空间分布为正六边形，PDAC为正四边形。

| 数据集 | 技术平台 | Spot 数量 | Cluster 数 | Ground Truth |
|--------|----------|-----------|------------|-------------|
| DLPFC | 10x Visium | ~4,000/section | 7 | Layer 1-6 + WM |
| PDAC | ST 1.0 | ~427 | 4 | 肿瘤区域标注 |
| Mouse Brain | 10x Visium | ~2,500 | 52 | Brain Atlas 标注 |
| Breast Cancer | 10x Visium | ~3,800 | 20 | 注释分区 |
| Chicken Heart | 10x Visium | ~2,700 | 时间序列 | D4/D7/D10/D14 |
| NanoString Lung | NanoString | ~2,000 (20 FOVs) | 91 | 空间分区 |

---

## 二、邻居投票机制

### 问题分析及解决

针对边界区域的判定，利用空间连续性先验——生物组织中，相邻 spot 通常属于同一组织域。通过邻居投票机制，每个 spot 综合其邻域信息进行判断，从而抑制噪声、平滑边界。
目前，每个样本中的探针空间分布是均匀的正六边/四边形，将一个点和它的相邻点进行综合（目前是原点2/3+相邻点1/3），这样可以获得一个N倍(4/6/?)于原本数据的数据集,将他们视为独立的点聚类，最后得到的结果再进行6->1的投票。

### 代码实现

#### 邻居识别

为每个 spot 找到其空间邻域中的真实邻居。（`ST/utils.py`中`get_neighbors` 函数）：

```python
def get_neighbors(coords, n_neighbors , eps=0.05):
    # 使用 KNN 找到候选邻居
    nbrs = NearestNeighbors(n_neighbors=n_neighbors * 2).fit(coords)
    dists, indices = nbrs.kneighbors(coords)
    dists = dists[:, 1:]  # 去掉自身
    indices = indices[:, 1:]

    # 自适应阈值筛选真实邻居
    grid_spacing = np.median(dists[:, 0])
    max_physical_dist = grid_spacing * 1.5  # 最大物理距离

    for i in range(len(coords)):
        min_dist = dists[i].min()
        local_thresh = min_dist * (1 + eps)  # 局部自适应阈值
        mask = (dists[i] <= local_thresh) & (dists[i] <= max_physical_dist)
        neighbors.append(indices[i][mask])
```

**设计要点**：
- `eps=0.05`：允许 5% 的距离容差，适应布局中计算的误差
- `max_physical_dist`：排除过远的假邻居
- 自适应阈值：每个 spot 根据其最近邻距离动态调整

#### 数据增强

生成包含邻域信息的虚拟样本。（`ST/utils.py`中`augment_spots_hexagonal` 函数）：

```python
def augment_spots_hexagonal(gene_features, image_features, neighbors, coords):
    # 混合比例: 2/3 原始 + 1/3 邻居
    index = 1.0 / 3.0  

    for i in range(len(gene_features)):
        for neighbor_idx in neighbors[i]:
            # 三模态同步增强
            aug_gene = (1 - index) * gene_features[i] + index * gene_features[neighbor_idx]
            aug_image = (1 - index) * image_features[i] + index * image_features[neighbor_idx]
            aug_coord = (1 - index) * coords[i] + index * coords[neighbor_idx]

            augmented_gene_list.append(aug_gene)
            augmented_image_list.append(aug_image)
            augmented_coords_list.append(aug_coord)
            spot_mapping.append(i)  # 记录该增强样本属于原始 spot i
```

- 三模态同步增强：基因表达、图像特征、空间坐标三者都被增强，保证一致性。这里1/3，2/3的比例经过测试是最佳的(相较于1/2、1/4、1/5)。

**数据规模变化**：
```
原始数据: N spots × 1 = N 个样本
增强后:   N spots × ~6 neighbors = ~6N 个样本（每个 spot 平均约 6 个增强样本）
spot_mapping: [0,0,0,0,0,0, 1,1,1,1,1,1, ..., N-1,N-1,N-1,N-1,N-1,N-1]
               ↑ spot 0 的 6 个增强样本        ↑ spot 1 的 6 个增强样本
```

#### 模型训练

在增强数据集上训练多模态 GCN，学习融合邻域信息的空间表示。（`ST/train.py` 第 190-242 行，`train_with_augmentation` 函数）：

```
增强数据集
    ↓
多模态 GCN 模型 (ST 类)
    ├─ 基因特征分支: MultiGCN → emb_x
    └─ 图像特征分支: MultiGCN → emb_i
    ↓
自注意力聚合: att_emb_x, att_emb_i
    ↓
MLP 融合: q_x, q_i
    ↓
加权融合: z_I = 20*att_emb_x + 1*att_emb_i + 10*z_xi
    ↓
ZINB 解码: 重构基因表达
    ↓
多任务损失: Total = 10*ZINB_Loss + 1*Contrastive_Loss + 10*Reg_Loss
    ↓
聚类 (KMeans): 获取每个增强样本的预测标签
```

**多任务损失函数**（`ST/train.py`）：

```python
# ZINB 重构损失: 建模基因表达的零膨胀负二项分布
zinb_loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(features_X, mean, mean=True)

# 对比损失: 跨视图一致性（基因分支 vs 图像分支）
cl_loss = consistency_loss(q_x, q_i)  # 或 crossview_contrastive_Loss

# 图正则化损失: 使嵌入空间保持空间邻近性
reg_loss = regularization_loss(z_xi, adj)

# 总损失: 权重可调（无 tensor 正则化时）
total_loss = a * zinb_loss + b * cl_loss + c * reg_loss

# 总损失: 含 tensor 低秩正则化时（tensor_loss=True）
# 额外增加: d * tensor_low_rank_loss(emb_x, emb_i, z_xi, temperature=tensor_temp)
total_loss = a * zinb_loss + b * cl_loss + c * reg_loss + d * tensor_loss
```

#### 张量低秩正则化

张量低秩正则化的工作流程：

```
1. 从三个嵌入视图分别构建自表示矩阵:
   S_x = softmax(emb_x @ emb_x^T / τ)  → 基因分支
   S_i = softmax(emb_i @ emb_i^T / τ)  → 图像分支
   S_z = softmax(z_xi @ z_xi^T / τ)    → 融合分支

2. 堆叠为 (N, N, 3) 张量: T = [S_x | S_i | S_z]

3. 沿视图维度做 FFT: T_fft = FFT(T, dim=-1)

4. 对每个频域切片做 SVD，计算核范数（奇异值之和）:
   loss = Σ_k ||T_fft[:,:,k]||_* / 3

效果: 迫使三个视图在"哪些点属于同一类"这件事上达成共识。
```

#### 投票聚合

将增强样本的预测聚合回原始 spot，通过多数投票确定最终标签。（`ST/utils.py` 第 343-357 行，`voting_aggregate_clusters` 函数）：

```python
def voting_aggregate_clusters(augmented_predictions, spot_mapping, n_original_spots):
    aggregated_labels = np.zeros(n_original_spots, dtype=int)

    for i in range(n_original_spots):
        # 找到所有属于原始 spot i 的增强样本
        mask = spot_mapping == i
        predictions_for_spot_i = augmented_predictions[mask]

        # 多数投票
        counter = Counter(predictions_for_spot_i)
        most_common_label = counter.most_common(1)[0][0]
        aggregated_labels[i] = most_common_label

    return aggregated_labels
```
---

### 调用关系图

```
DLPFC.py / PDAC.py / compare.py
    │
    ├─ 1. 加载数据（scanpy）
    ├─ 2. 预处理（过滤、归一化、HVG选择）
    ├─ 3. 加载图像特征 → adata.obsm["im_re"]
    ├─ 4. 计算邻接矩阵 → adata.obsm["adj"]
    │
    ▼
train_with_augmentation (train.py:120)
    │
    ├─ get_neighbors (utils.py:290)
    │       ↓
    ├─ augment_spots_hexagonal (utils.py:309)
    │       ├─ gene_features, image_features, coords
    │       ├─ spot_mapping (N → ~6N)
    │       ↓
    ├─ pairwise_distance (utils.py:95) [增强数据邻接矩阵]
    │       ↓
    ├─ models.ST (models.py:263)
    │       ├─ MultiGCN → emb_x, emb_i
    │       ├─ SelfAttention → att_emb_x, att_emb_i
    │       ├─ MLP → q_x, q_i
    │       ├─ 加权融合: z_I = 20*att_emb_x + 1*att_emb_i + 10*z_xi
    │       └─ ZINBdecoder → pi, disp, mean
    │       ↓
    ├─ 多任务损失: 10*ZINB + 1*Contrastive + 10*Reg
    │       （当 tensor_loss=True 时额外增加: d * tensor_low_rank_loss）
    │       ↓
    ├─ KMeans 聚类 → augmented_predictions (~6N,)
    │       ↓
    ├─ voting_aggregate_clusters (utils.py:343)
    │       ├─ augmented_predictions, spot_mapping
    │       ↓
    └─ original_labels (N,) → adata.obs['ST']
```

---

### 张量正则化调用关系（当 `tensor_loss=True` 时）

```
models.ST.forward
    ├─ emb_x (基因分支 GCN 输出)
    ├─ emb_i (图像分支 GCN 输出)
    └─ z_xi (融合分支输出)
            ↓
    self_representation_matrix (utils.py:220)
            ├─ S_x = softmax(emb_x @ emb_x^T / τ)
            ├─ S_i = softmax(emb_i @ emb_i^T / τ)
            └─ S_z = softmax(z_xi @ z_xi^T / τ)
            ↓
    张量堆叠: T ∈ R^(N×N×3)
            ↓
    FFT: T_fft = FFT(T, dim=-1)
            ↓
    逐切片 SVD: loss = Σ_k ||T_fft[:,:,k]||_* / 3
```

---

## 三、LLM辅助基因选择

### 核心思想

baseline依赖HVG来筛选特征基因，但 HVG 仅反映基因表达的变化幅度，无法捕获基因在特定组织中生物学功能相关性。引入LLM，通过自然语言理解能力从基因知识库中筛选出与目标组织结构、功能或细胞类型高度相关的基因，为模型提供更具生物学意义的第三分支特征。

### 技术路线

```
基因知识库（CSV，含 gene_summary / target_description）
        ↓
候选基因列表（全部 / 按表达量过滤）
        ↓
LLMGeneSelector（分批 API 调用）
        ↓
生物相关基因集（G ∈ ℝ^{|G| × d_llm}）
        ↓
PCA 降维 → llm_gene_feature（20 维）
        ↓
三模态 GCN 分支 → 与基因分支、图像分支做一致性约束
```

### 特征构建策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `exclusive`（排他） | LLM 基因从 HVG 中移除，标准分支与 LLM 分支互不重叠 | LLM 基因数量充足且质量高时效果更好 |
| `auxiliary`（辅助） | LLM 基因经 PCA 降维后作为第三分支，保留全部 HVG | LLM 基因较少时，保证信息不丢失 |

### 代码实现

#### LLM 基因选择器（`ST/gene_selector.py`）

`LLMGeneSelector` 负责调用 LLM API 进行批量基因筛选：

```python
gene_selector = LLMGeneSelector(model="gpt-5", base_url=TUZI_API_BASE_URL)
llm_genes, reasoning = gene_selector.select_genes_with_llm(
    tissue_type="Human Pancreatic Ductal Adenocarcinoma",
    gene_list=candidate_genes,
    knowledge_path="Data/knowledge/PDAC_knowledge_table.csv",
    tokens_per_gene_est=80,
    max_context_tokens=12000,
    max_retries=3
)
```

其中，基因的先验知识通过MyGene.info以及NIH获取，保存在Data/Genes中。
#### 三分支模型架构（`ST/models.py`）

当 `use_llm_gene=True` 时，`ST` 模型流程如下：

```
基因特征 X (HVG)  → MultiGCN → emb_x → SelfAttention → att_emb_x
                                               ↓
LLM基因特征 L    → MultiGCN → emb_l → SelfAttention → att_emb_l
                                               ↓
图像特征 I       → MultiGCN → emb_i → SelfAttention → att_emb_i
                                               ↓
                  MLP (mlpx, mlpi, mlpl) → q_x, q_i, q_l
                                               ↓
                  拼接: [q_x | q_i | q_l] → fc → z_xi
                                               ↓
加权融合: z_I = w_gene*att_emb_x + w_image*att_emb_i + w_llm*att_emb_l + w_fusion*z_xi
                                               ↓
                  ZINB 解码 → 重构基因表达
```

权重参数（默认）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weight_gene` | 20.0 | 基因分支融合权重 |
| `weight_image` | 1.0 | 图像分支融合权重 |
| `weight_llm` | 15.0（排他）/ 8.0（辅助） | LLM 分支融合权重 |
| `weight_fusion` | 10.0 | MLP 融合权重 |
| `d` | 0.5（排他）/ 0.3（辅助） | LLM 一致性损失权重 |


### API 配置

API 配置通过tuzi api配置，使用gpt-5模型

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_BASE_URL"] = "https://api.tu-zi.com/v1"

# 或直接传入
gene_selector = LLMGeneSelector(
    api_key="your-api-key",
    model="gpt-5",
    base_url="https://api.tu-zi.com/v1"
)
```

---

## 测试

```bash
cd ST_neighborVote

python compare.py        # PDAC: Base vs Aug
python compare.py         # 将 compare_tensor=False → True 可启用 Aug+Tensor

python PDAC.py           # PDAC 单独对比 Base vs Aug
python DLPFC.py          # DLPFC (151673, k=7) with 邻居投票

# PDAC
python run_PDAC_with_LLM.py --use_llm false
python run_PDAC_with_LLM.py --use_llm true --strategy auxiliary

# DLPFC
python run_DLPFC_with_LLM.py --use_llm false
python run_DLPFC_with_LLM.py --use_llm true --strategy auxiliary --section_id 151673 --k 7

# Mouse Brain
python run_Mouse_Brain_with_LLM.py --use_llm false
python run_Mouse_Brain_with_LLM.py --use_llm true --strategy auxiliary
```

```python
results = run_experiment(
    adata,
    compare_tensor=True,   # 启用 Aug+Tensor 实验
    tensor_temp=0.1,       # 温度参数
    d=1.0                   # tensor loss 权重
)
```

**输出**：
- 控制台打印各方法 ARI 及相对提升
- 生成对比图 `figures/PDAC_base_vs_aug_vs_tensor.pdf/png`

PDAC 数据集 — `PDAC.py`

仅对比标准训练和邻居投票增强训练：

```bash
python PDAC.py
```

### DLPFC 数据集测试 — `DLPFC.py`

```bash
python DLPFC.py --section_id 151673 --k 7
```

**输出**：
- 控制台打印 ARI
- 生成空间分布图（如果 `sc.pl.spatial` 成功）

---

## 遇到的问题
- Neighbor部分
### GPU 显存不足（OOM）

以PDAC为例，增强后的数据集规模约为原始数据的 6 倍（如 4000 spots → 24000 augmented spots），邻接矩阵从 `4000×4000` → `24000×24000`，使用 float32 也需要约 2.3 GB 显存，加上模型训练，总显存需求远接近12GB，其它数据集需求更高，导致 OOM。

可以通过使用显存较大的云服务器(24GB以上)，实现分块距离矩阵计算(需求仍较大)，采用Spatial-MGCN作为Baseline(相较于STMMR少了图像特征的维度，所需显存显著少，但是ARI相应表现不佳)来解决。

### 训练时间较长
由于数据规模成倍增加，导致训练时间相应提升，每次训练耗时较长。

### 改进效果和数据集关系大
在PDAC这类数据点少、数据边界不清晰的数据集上，改进效果较好，ARI 38%->45%，但是在DLPFC这类数据点很多、边界也较为清晰的情况，反而会导致边界处的模糊误判，ARI提升效果不好或降低。需要对投票处理的逻辑进一步改进。

- LLM部分
### 大模型Token花费高
Data中，每个数据集的adata中都可以找到这个数据集中的所有Gene及其表达量，将这些Gene的summary提取出来(可以使用Mygene.info的api来获取，生信数据库，获取内容质量较好)，如果将这些summary(50字截断)送入LLM筛选，token消耗量也很高(基本上一个数据集需要50元token成本)，可以考虑用便宜点的模型、不让LLM输出选择理由、做好缓存等来控制成本。
### LLM输出时间长
由于输入内容过多，为了不超出上下文，需要分批送入，等待输出的时间较长，而且分批送入就需要每一批都要选取一部分Gene作为输出，很难控制Gene输出的准确性/数量，需要对Gene做随机或者多次输入。
### 提示词优化
提示词需要进一步优化，现有提示词输出的Gene虽然对这个组织的影响确实很大，但是很多都是一个作用，会有共相关的问题，输出的结果不太好，需要避免这一点。

---

目前，邻居投票的结果在部分数据集效果较好(Mouse/PDAC)，最多可以提升5%的ARI，张量低秩正则化对最终结果影响较小，而使用LLM的效果比较差，逻辑还需要进一步改进。现在LLM提取出的基因存放在knowledge下，但是可能需要重新更改提示词后再更新。
此外，训练时，所需显存资源和时间的处理也需要再进行优化，可以考虑使用Spacial-MGCN作为baseline来尽量减少显存和时间上的需求。

---