import json
import torch
import pickle
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, HeteroData
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from typing import Dict, List, Tuple
import networkx as nx
from sklearn.decomposition import PCA
import faiss

class GraphBuilder:
    def __init__(self, json_file_path: str, embedding_model: str = 'bge-base-en-v1.5', similarity_threshold: float = 0.7):
        """
        初始化图构建器
        
        Args:
            json_file_path: train_questions.json文件路径
            embedding_model: 用于计算嵌入的模型名称
            similarity_threshold: 相似度阈值，超过此值的节点之间会连边
        """
        self.json_file_path = json_file_path
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        
        # 初始化嵌入模型
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # 数据存储
        self.profiles_data = []  # 存储所有profile数据
        self.all_items = []      # 存储所有item（论文）数据
        self.item_embeddings = None  # 存储所有item的嵌入
        
        # 索引映射
        self.item_id_to_idx = {}     # item_id -> 在图中的节点索引
        self.profile_id_to_items = {}  # profile_id -> item_ids列表
        self.item_to_profile = {}    # item_id -> profile_id
        
        # 新增：用户ID映射
        self.original_user_id_to_profile_id = {}  # 原始user_id -> profile_id
        self.profile_id_to_original_user_id = {}  # profile_id -> 原始user_id
        self.original_user_id_to_node_indices = {}  # 原始user_id -> 该用户的所有节点索引列表
        
    def load_data(self):
        """加载JSON数据"""
        print("Loading data from JSON file...")
        
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理每个profile
        for profile_idx, profile in enumerate(tqdm(data, desc="Processing profiles")):
            profile_id = f"profile_{profile_idx}"
            
            # 提取原始user_id (新增)
            original_user_id = str(profile.get('user_id', f"unknown_{profile_idx}"))
            
            # 存储用户ID映射关系 (新增)
            self.original_user_id_to_profile_id[original_user_id] = profile_id
            self.profile_id_to_original_user_id[profile_id] = original_user_id
            self.original_user_id_to_node_indices[original_user_id] = []
            
            self.profiles_data.append({
                'profile_id': profile_id,
                'original_user_id': original_user_id,  # 新增
                'profile': profile['profile']
            })
            
            # 记录profile包含的items
            item_ids = []
            for item in profile['profile']:
                item_id = item['id']
                item_ids.append(item_id)
                
                # 存储item数据
                self.all_items.append({
                    'id': item_id,
                    'title': item['title'],
                    'abstract': item['abstract'],
                    'date': item.get('date', None),
                    'profile_id': profile_id,
                    'original_user_id': original_user_id  # 新增：记录文档所属的原始用户ID
                })
                
                # 建立映射关系
                self.item_to_profile[item_id] = profile_id
            
            self.profile_id_to_items[profile_id] = item_ids
        
        # 创建item_id到索引的映射
        for idx, item in enumerate(self.all_items):
            item_id = item['id']
            self.item_id_to_idx[item_id] = idx
            
            # 更新用户节点索引映射 (新增)
            original_user_id = item['original_user_id']
            self.original_user_id_to_node_indices[original_user_id].append(idx)
        
        print(f"Loaded {len(self.profiles_data)} profiles with {len(self.all_items)} total items")
        print(f"Found {len(self.original_user_id_to_profile_id)} unique original user IDs")
    
    def compute_embeddings(self, batch_size: int = 128, target_dim: int = 128):
        """计算所有item的嵌入，并降维到指定维度"""
        print("Computing embeddings for all items...")
        
        # 准备文本数据
        texts = []
        for item in self.all_items:
            # 组合title和abstract作为文本表示
            text = f"{item['title']} {item['abstract']}"
            texts.append(text)
        
        # 批量计算原始嵌入（768维）
        original_embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        print(f"Original embeddings shape: {original_embeddings.shape}")
        
        # PCA降维
        if target_dim < original_embeddings.shape[1]:
            print(f"Reducing dimension from {original_embeddings.shape[1]} to {target_dim} using PCA...")
            
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(original_embeddings.cpu().numpy())
            
            # 转换回torch tensor并归一化
            self.item_embeddings = torch.tensor(reduced_embeddings, dtype=torch.float32)
            self.item_embeddings = torch.nn.functional.normalize(self.item_embeddings, dim=1)
            
            # 保存PCA模型以便后续使用
            self.pca_model = pca
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        else:
            self.item_embeddings = original_embeddings
        
        print(f"Final embeddings shape: {self.item_embeddings.shape}")

        output_dir="./"
        if output_dir:
            embedding_save_path = os.path.join(output_dir, "embeddings.pt")
            embedding_data = {
                'item_embeddings': self.item_embeddings,
                'all_items': self.all_items,
                'item_id_to_idx': self.item_id_to_idx,
                'embedding_model_name': self.embedding_model_name,
                'embedding_dim': self.item_embeddings.shape[1],
                'pca_model': getattr(self, 'pca_model', None),
            }
            torch.save(embedding_data, embedding_save_path)
            print(f"Embeddings saved to: {embedding_save_path}")

        return self.item_embeddings
    
    def build_edges_faiss(self, k_neighbors=50):
        """使用FAISS进行高效的近似最近邻搜索"""
        print(f"Building edges with FAISS (k={k_neighbors})...")
        
        edge_list = []
        edge_types = []
        
        # 1. 同一profile内的边（保持不变）
        print("Adding same-profile edges...")
        for profile_id, item_ids in tqdm(self.profile_id_to_items.items(), desc="Same-profile edges"):
            item_indices = [self.item_id_to_idx[item_id] for item_id in item_ids if item_id in self.item_id_to_idx]
            
            for i in range(len(item_indices)):
                for j in range(i + 1, len(item_indices)):
                    edge_list.append([item_indices[i], item_indices[j]])
                    edge_list.append([item_indices[j], item_indices[i]])
                    edge_types.extend([0, 0])
        
        # 2. 使用FAISS构建高效索引
        embeddings_np = self.item_embeddings.cpu().numpy().astype('float32')
        dimension = embeddings_np.shape[1]
        n_vectors = embeddings_np.shape[0]
        
        # 创建IVF索引（倒排索引）
        nlist = min(4096, int(np.sqrt(n_vectors)))  # 聚类数量
        quantizer = faiss.IndexFlatIP(dimension)    # 内积索引
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        print(f"Training FAISS index with {nlist} clusters...")
        index.train(embeddings_np)
        index.add(embeddings_np)
        
        # 设置搜索参数
        index.nprobe = min(64, nlist)  # 搜索多少个聚类
        
        # 3. 搜索最近邻
        print("Searching for nearest neighbors...")
        distances, indices = index.search(embeddings_np, k_neighbors + 1)  # +1因为包含自己
        
        # 4. 处理搜索结果
        print("Processing neighbor results...")
        for i in tqdm(range(len(self.all_items)), desc="Adding similarity edges"):
            for j in range(1, len(distances[i])):  # 跳过自己
                neighbor_idx = indices[i][j]
                similarity = distances[i][j]  # 内积值
                
                if similarity > self.similarity_threshold and neighbor_idx != -1:
                    # 检查是否为不同profile
                    item_i_profile = self.item_to_profile[self.all_items[i]['id']]
                    item_j_profile = self.item_to_profile[self.all_items[neighbor_idx]['id']]
                    
                    if item_i_profile != item_j_profile:
                        # 避免重复添加边
                        if i < neighbor_idx:
                            edge_list.extend([[i, neighbor_idx], [neighbor_idx, i]])
                            edge_types.extend([1, 1])
        
        print(f"Total edges created: {len(edge_list)}")
        return torch.tensor(edge_list, dtype=torch.long).t(), torch.tensor(edge_types, dtype=torch.long)
    
    def build_graph(self):
        """构建同质图（所有节点都是item）"""
        print("Building homogeneous graph...")
        
        # 构建边
        edge_index, edge_types = self.build_edges_faiss()
        
        # 创建图数据
        graph_data = Data(
            x=self.item_embeddings,  # 节点特征（嵌入）
            edge_index=edge_index,   # 边索引
            edge_attr=edge_types     # 边类型
        )
        
        # 添加元数据
        graph_data.num_nodes = len(self.all_items)
        graph_data.item_data = self.all_items
        graph_data.item_id_to_idx = self.item_id_to_idx
        graph_data.profile_id_to_items = self.profile_id_to_items
        graph_data.item_to_profile = self.item_to_profile
        
        return graph_data
    
    def save_graph(self, graph_data, save_path: str):
        """保存图数据"""
        print(f"Saving graph data to {save_path}")
        
        save_data = {
            'graph_data': graph_data,
            'item_embeddings': self.item_embeddings,
            'all_items': self.all_items,
            'profiles_data': self.profiles_data,
            'item_id_to_idx': self.item_id_to_idx,
            'profile_id_to_items': self.profile_id_to_items,
            'item_to_profile': self.item_to_profile,
            'similarity_threshold': self.similarity_threshold,
            'embedding_model_name': self.embedding_model_name,
            'pca_model': getattr(self, 'pca_model', None),  # 保存PCA模型
            'embedding_dim': self.item_embeddings.shape[1],  # 实际维度
            
            # 新增：保存用户ID映射
            'original_user_id_to_profile_id': self.original_user_id_to_profile_id,
            'profile_id_to_original_user_id': self.profile_id_to_original_user_id,
            'original_user_id_to_node_indices': self.original_user_id_to_node_indices
        }
        
        torch.save(save_data, save_path)
        print(f"Graph saved successfully!")
        

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="Build graph from profile data for GNN retrieval")
    
    # 输入/输出参数
    parser.add_argument("--json_file_path", type=str, default="train_questions.json",
                        help="Path to the input JSON file containing profile data")
    parser.add_argument("--output_dir", type=str, default="gnn_retrieval",
                        help="Output directory to save the graph data")
    
    # 模型参数
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-base-en-v1.5",
                        help="Embedding model to use for computing item embeddings")
    
    # 嵌入维度参数
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Target embedding dimension (will use PCA if smaller than model output)")
    
    # 图构建参数
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                        help="Similarity threshold for connecting nodes (0.0-1.0)")
    
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for embedding computation")
    
    return parser

if __name__ == "__main__":
    # 解析命令行参数
    parser = create_parser()
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(2025)
    np.random.seed(2025)
    
    # 打印配置信息
    print("=== Graph Construction Configuration ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 40)
    
    # 验证输入文件存在
    if not os.path.exists(args.json_file_path):
        raise FileNotFoundError(f"Input file not found: {args.json_file_path}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建图
    builder = GraphBuilder(
        json_file_path=args.json_file_path,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold
    )
    
    # 1. 加载数据
    builder.load_data()
    
    # 2. 计算嵌入
    builder.compute_embeddings(batch_size=args.batch_size,target_dim=args.embedding_dim)
    
    # 3. 构建图
    graph = builder.build_graph()
    
    # 4. 保存图数据
    graph_save_path = os.path.join(args.output_dir, "graph.pt")
    builder.save_graph(graph, graph_save_path)
    
    # 5. 保存统计信息
    stats = {
        'num_profiles': len(builder.profiles_data),
        'num_items': len(builder.all_items),
        'num_edges': graph.edge_index.shape[1],
        'num_same_profile_edges': (graph.edge_attr == 0).sum().item(),
        'num_similarity_edges': (graph.edge_attr == 1).sum().item(),
        'embedding_dim': builder.item_embeddings.shape[1],
        'similarity_threshold': builder.similarity_threshold,
        'embedding_model': builder.embedding_model_name,
        'num_original_users': len(builder.original_user_id_to_profile_id)  # 新增
    }
    
    with open(os.path.join(args.output_dir, "graph_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n=== Graph Construction Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\nGraph construction completed!")
    print(f"Results saved to: {args.output_dir}")