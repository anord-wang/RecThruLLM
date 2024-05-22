import os
import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import load_npz
from tqdm import tqdm
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from torch_geometric.utils import to_dense_adj


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)


def read_and_combine_matrices(data_root):
    # 读取矩阵文件
    train_matrix = load_npz(os.path.join(data_root, 'train_matrix.npz'))
    # print(train_matrix.shape)
    val_matrix = load_npz(os.path.join(data_root, 'val_matrix.npz'))
    # print(val_matrix.shape)
    test_matrix = load_npz(os.path.join(data_root, 'test_matrix.npz'))
    # print(test_matrix.shape)

    # 合并矩阵
    combined_matrix = train_matrix + val_matrix + test_matrix

    # 确保合并后的矩阵仍然是二元的（1 表示交互，0 表示无交互）
    combined_matrix[combined_matrix > 1] = 1

    return combined_matrix


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def preprocess_features(features):
    features = features.cpu()  # 添加这行来确保Tensor在CPU上
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.FloatTensor(np.array(features))


def create_edge_index(adj):
    """
    Create an edge index tensor for PyTorch Geometric from a user-item interaction matrix.

    Parameters:
    - adj: User-item interaction matrix in sparse format.

    Returns:
    - edge_index: Edge index tensor for graph representation in PyTorch Geometric.
    """
    N, M = adj.shape  # N users, M items

    # 创建用户-用户和物品-物品之间的0矩阵
    zero_user_user = sp.csr_matrix((N, N))
    zero_item_item = sp.csr_matrix((M, M))

    # 创建上半部分和下半部分的矩阵
    upper_half = sp.hstack([zero_user_user, adj])  # 上半部分
    lower_half = sp.hstack([adj.T, zero_item_item])  # 下半部分

    # 合并上半部分和下半部分以创建完整的邻接矩阵
    adj_matrix = sp.vstack([upper_half, lower_half])

    # 转换为COO格式
    adj_matrix_coo = adj_matrix.tocoo()

    # 创建edge_index
    edge_index = torch.tensor([adj_matrix_coo.row, adj_matrix_coo.col], dtype=torch.long)

    return edge_index


def graph_autoencoder_loss(z, edge_index, num_nodes):
    # 从edge_index计算稠密的邻接矩阵
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

    # 使用解码器预测邻接矩阵
    adj_pred = torch.sigmoid(torch.matmul(z, z.t()))

    # 计算损失
    loss = F.binary_cross_entropy(adj_pred, adj.to(device))
    return loss


def main():
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    server_root = "/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent"
    
    # dataset = 'beauty'
    dataset = 'luxury'

    data_root = os.path.join(server_root, "dataset", dataset)

    combined_matrix = read_and_combine_matrices(data_root)

    pretrained_user_emb_path = "/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent/model/luxury/collaborative/user_embeddings_1_0.06833057105541229_75.pt"
    pretrained_item_emb_path = "/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent/model/luxury/collaborative/item_embeddings_1_0.06833057105541229_75.pt"
    user_embeddings = torch.load(pretrained_user_emb_path, map_location=device)
    item_embeddings = torch.load(pretrained_item_emb_path, map_location=device)
    print(user_embeddings.keys())

    user_embeddings = user_embeddings['weight']
    item_embeddings = item_embeddings['weight']

    adj = combined_matrix  # Assuming combined_matrix is the adjacency matri
    features = torch.cat([user_embeddings, item_embeddings], dim=0)
    features = preprocess_features(features)

    num_nodes = len(features)

    edge_index = create_edge_index(adj)

    data = Data(x=features, edge_index=edge_index).to(device)
    # Define model
    print('features.shape[1]', features.shape[1])
    # model = GCN(in_channels=features.shape[1], hidden_channels=64, out_channels=16).to(device)
    model = GCN(in_channels=features.shape[1], hidden_channels=features.shape[1], out_channels=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    for epoch in tqdm(range(200), desc='Training Progress'):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)  # GCN模型的前向传播
        loss = graph_autoencoder_loss(z, data.edge_index, num_nodes)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Use the model to generate embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()

    # Cluster the embeddings
    centers = []
    labels = []
    for i in range(3):
        print('index', i)
        kmeans = KMeans(n_clusters=10**(i+1), random_state=0).fit(embeddings)
        # 获取聚类中心的表示向量
        predict_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        centers.append(cluster_centers)
        labels.append(predict_labels)

    # predict_labels 包含了每个节点所属的聚类中心的索引
    # cluster_centers 包含了每个聚类中心的表示向量

    print('labels.shape', len(labels))
    print('labels[0].shape', len(labels[0]))
    print('labels[1].shape', len(labels[1]))
    print('labels[2].shape', len(labels[2]))
    print('labels[1][1].shape', labels[1][1])

    print('centers.shape', len(centers))
    print('centers[1].shape', centers[0].shape)
    print('centers[2].shape', centers[1].shape)
    print('centers[3].shape', centers[2].shape)
    print('centers.shape[1][0]', centers[1][0].shape)

    with open('/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent/dataset/luxury/kmeans_cluster.pkl', 'wb') as f:
        pickle.dump({
        'Labels': labels,
        'Centers': centers
    }, f)

    print("聚类结果已保存为Pickle文件。")

    # # 打印聚类中心的表示向量
    # print("聚类中心的表示向量：")
    # print(cluster_centers)
    #
    # # 如果您想知道每个节点分别属于哪个聚类中心
    # # predict_labels 已经包含了这个信息，每个节点对应的值就是它所属的聚类中心的索引
    #
    # # 例如，打印前10个节点的聚类中心索引
    # print("前10个节点所属的聚类中心索引：")
    # print(predict_labels[:10])
    #
    # # 如果您还想将每个节点与其对应的聚类中心的表示向量直接关联起来
    # # 可以通过以下方式实现
    # node_to_cluster_center = cluster_centers[predict_labels]
    #
    # # 打印前10个节点对应的聚类中心表示向量
    # print("前10个节点对应的聚类中心表示向量：")
    # print(node_to_cluster_center[:10])
    # # predict_labels contains the cluster assignment for each node


if __name__ == '__main__':
    main()
