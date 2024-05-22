import torch
from transformers import GraphormerModel
import os
import numpy as np
from scipy.sparse import load_npz, coo_matrix
from tqdm import tqdm

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

# 假设您的数据根目录为以下路径
# dataset = 'beauty'
# dataset = 'sports'
# dataset = 'toys'
# dataset = 'yelp'
# dataset = 'scientific'
# dataset = 'arts'
# dataset = 'instruments'
# dataset = 'office'
# dataset = 'pantry'
dataset = 'luxury'
# dataset = 'music'
# dataset = 'garden'
# dataset = 'food'
server_root = "/home/local/ASURITE/xwang735/LLM4REC/LLM4Rec"
# server_root = "/home/wxy/LLM4Rec"
gpt2_server_root = server_root
data_root = os.path.join(gpt2_server_root, "dataset", dataset)

combined_matrix = read_and_combine_matrices(data_root)

# 假设local_root, args, device等变量已经定义
pretrained_user_emb_path = "/home/local/ASURITE/xwang735/LLM4REC/LLM4Rec/model/luxury/collaborative/user_embeddings_1_0.06833057105541229_75.pt"
pretrained_item_emb_path = "/home/local/ASURITE/xwang735/LLM4REC/LLM4Rec/model/luxury/collaborative/item_embeddings_1_0.06833057105541229_75.pt"
user_embeddings = torch.load(pretrained_user_emb_path, map_location=device)
item_embeddings = torch.load(pretrained_item_emb_path, map_location=device)
print(user_embeddings.keys())

user_embeddings = user_embeddings['weight']
item_embeddings = item_embeddings['weight']

# 合并用户和物品节点特征
node_features = torch.cat([user_embeddings, item_embeddings], dim=0)
node_features = torch.unsqueeze(node_features, 0)
print('node_features.shape', node_features.shape)

combined_coo = coo_matrix(combined_matrix)
num_users, num_items = combined_coo.shape
input_edges = torch.tensor(np.vstack([combined_coo.row, combined_coo.col + num_users]), dtype=torch.long)
print('input_edges.shape', input_edges.shape)

num_heads = 1  # 假设注意力头数为1，根据你的模型实际情况调整
num_nodes = combined_matrix.shape[0] + combined_matrix.shape[1]  # 总节点数
attn_bias = torch.zeros((num_heads, num_nodes, num_nodes))
print('attn_bias.shape', attn_bias.shape)

degrees = torch.zeros(num_users + num_items, dtype=torch.float)
degrees.index_add_(0, input_edges[0], torch.ones(input_edges.size(1), dtype=torch.float))
degrees.index_add_(0, input_edges[1], torch.ones(input_edges.size(1), dtype=torch.float))
in_degree = degrees
out_degree = degrees
print('in_degree.shape', in_degree.shape)
print('out_degree.shape', out_degree.shape)

# 生成一个从0到num_nodes-1的整数序列，作为spatial_pos
spatial_pos = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0)  # 增加批次维度
print('spatial_pos.shape', spatial_pos.shape)

attn_edge_type = torch.zeros(input_edges.size(1), dtype=torch.long)
print('attn_edge_type.shape', attn_edge_type.shape)

# # 获取用户和物品的数量
# len_user, len_item = user_embeddings.shape[0], item_embeddings.shape[0]
#
# # 更新边索引以反映合并后的节点特征矩阵
# edges = np.transpose(np.nonzero(combined_matrix))
# # 对物品节点的索引进行调整
# edges[:, 1] += len_user  # 增加len_user的量，因为物品索引在合并矩阵中紧随用户索引之后
# edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 加载Graphormer模型
model = GraphormerModel.from_pretrained("clefourrier/graphormer-base-pcqm4mv2").to(device)

# 准备输入数据
inputs = {
    'input_nodes': node_features,  # 已经准备好的节点特征
    'input_edges': input_edges,
    'attn_bias': attn_bias,
    'in_degree': in_degree.unsqueeze(0),  # 增加批次维度
    'out_degree': out_degree.unsqueeze(0),  # 增加批次维度
    'spatial_pos': spatial_pos,
    'attn_edge_type': attn_edge_type,
}


with tqdm(total=1, desc="Processing") as pbar:
    outputs = model(**inputs)
    updated_node_features = outputs.last_hidden_state
    # 在这里更新进度条
    pbar.update(1)


# 分离更新后的用户和物品节点特征
updated_user_features = updated_node_features[:len_user, :]
updated_item_features = updated_node_features[len_user:, :]

# 假设你已经有了 updated_user_features 和 updated_item_features

# 定义保存路径
save_path = "/home/local/ASURITE/xwang735/LLM4REC/LLM4Rec/model/beauty/collaborative/"  # 更改为你的保存目录
user_features_path = os.path.join(save_path, "updated_user_features.pt")
item_features_path = os.path.join(save_path, "updated_item_features.pt")

# 保存更新后的用户和物品特征
torch.save(updated_user_features, user_features_path)
torch.save(updated_item_features, item_features_path)

# 输出相关信息
print(f"Updated user features saved to: {user_features_path}")
print(f"Updated item features saved to: {item_features_path}")
print(f"Updated user features shape: {updated_user_features.shape}")
print(f"Updated item features shape: {updated_item_features.shape}")

