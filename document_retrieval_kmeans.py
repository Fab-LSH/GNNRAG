import torch
import argparse
from tqdm import tqdm
import os
import json
import numpy as np
import faiss
from sklearn.cluster import KMeans
from collections import defaultdict

def load_graph_data(graph_path):
    """加载图数据并返回关键组件"""
    print(f"加载图数据: {graph_path}")
    graph_data = torch.load(graph_path, map_location='cpu')
    
    # 提取关键数据
    item_embeddings = graph_data['item_embeddings'].cpu().numpy()
    all_items = graph_data['all_items']
    item_id_to_idx = graph_data['item_id_to_idx']
    profile_id_to_items = graph_data['profile_id_to_items']
    item_to_profile = graph_data['item_to_profile']
    
    # 创建节点索引到文档ID的反向映射
    idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
    
    # 获取用户ID映射
    profile_to_user_id = {}
    user_id_to_profile = {}
    
    # 尝试获取用户ID映射
    if 'profile_id_to_original_user_id' in graph_data and 'original_user_id_to_profile_id' in graph_data:
        profile_to_user_id = graph_data['profile_id_to_original_user_id']
        user_id_to_profile = graph_data['original_user_id_to_profile_id']
        print(f"使用映射: profile_id_to_original_user_id 和 original_user_id_to_profile_id")
    
    # 获取有效的节点索引范围
    max_valid_idx = max(idx_to_item_id.keys())
    print(f"有效节点索引范围: 0-{max_valid_idx}，共{len(idx_to_item_id)}个节点")
    
    return {
        'graph_data': graph_data,
        'item_embeddings': item_embeddings,
        'all_items': all_items,
        'item_id_to_idx': item_id_to_idx,
        'idx_to_item_id': idx_to_item_id,
        'profile_id_to_items': profile_id_to_items,
        'item_to_profile': item_to_profile,
        'profile_to_user_id': profile_to_user_id,
        'user_id_to_profile': user_id_to_profile,
        'max_valid_idx': max_valid_idx
    }

def get_user_clusters(user_embeddings, min_docs=2, max_clusters=3):
    """
    对用户文档embedding进行KMeans聚类
    
    Args:
        user_embeddings: 用户文档的embedding数组
        min_docs: 最少需要的文档数量才执行聚类
        max_clusters: 最大聚类数量
        
    Returns:
        clusters: 聚类结果字典，包含中心点和文档索引
    """
    n_docs = len(user_embeddings)
    
    # 文档太少，不进行聚类
    if n_docs < min_docs:
        return [{
            'center': np.mean(user_embeddings, axis=0),
            'doc_indices': list(range(n_docs)),
            'weight': 1.0
        }]
    
    # 确定聚类数量，最多max_clusters个，但不超过文档数量的一半
    n_clusters = min(max_clusters, max(2, n_docs // 2))
    
    # 执行KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(user_embeddings)
    
    # 统计每个聚类的大小和文档索引
    clusters = []
    for i in range(n_clusters):
        doc_indices = np.where(labels == i)[0].tolist()
        if not doc_indices:  # 空聚类跳过
            continue
            
        # 计算聚类权重 (基于聚类大小)
        weight = len(doc_indices) / n_docs
        
        clusters.append({
            'center': kmeans.cluster_centers_[i],
            'doc_indices': doc_indices,
            'weight': weight
        })
    
    # 按权重排序
    clusters = sorted(clusters, key=lambda x: x['weight'], reverse=True)
    
    return clusters

def search_docs_with_clusters(profile_id, item_ids, item_id_to_idx, item_embeddings, 
                             index, faiss_idx_to_node_idx, top_k=10, filter_own_docs=True, max_clusters=3):
    """
    使用聚类方法搜索文档
    """
    # 获取用户文档索引
    user_item_indices = []
    for item_id in item_ids:
        if item_id in item_id_to_idx:
            user_item_indices.append(item_id_to_idx[item_id])
    
    if not user_item_indices:
        return []
    
    # 获取用户文档embedding
    user_doc_embeddings = item_embeddings[user_item_indices]
    
    # 对用户文档进行聚类
    clusters = get_user_clusters(user_doc_embeddings, min_docs=2, max_clusters=max_clusters)
    
    # 初始搜索数量（考虑可能的重复）
    cluster_top_k = max(top_k, top_k + len(user_item_indices))
    
    # 存储所有检索结果
    all_results = []
    
    # 对每个聚类执行检索
    for cluster in clusters:
        # 准备查询embedding
        query_embedding = cluster['center'].reshape(1, -1).astype('float32')
        weight = cluster['weight']
        
        # 根据聚类权重确定检索数量
        search_k = max(1, int(cluster_top_k * weight * 2))  # 适当放大以确保有足够结果
        
        # 执行检索
        distances, indices = index.search(query_embedding, search_k)
        
        # 处理结果
        for j, faiss_idx in enumerate(indices[0]):
            if faiss_idx >= len(faiss_idx_to_node_idx):
                continue
            
            # 获取原始节点索引
            node_idx = faiss_idx_to_node_idx[faiss_idx]
            
            # 计算加权相似度
            similarity = float(distances[0][j]) * weight
            
            # 存储结果
            all_results.append({
                'node_idx': node_idx,
                'similarity': similarity,
                'cluster_weight': weight
            })
    
    # 合并重复结果，保留最高相似度
    node_to_best_result = {}
    for result in all_results:
        node_idx = result['node_idx']
        if node_idx not in node_to_best_result or result['similarity'] > node_to_best_result[node_idx]['similarity']:
            node_to_best_result[node_idx] = result
    
    # 按相似度排序
    sorted_results = sorted(node_to_best_result.values(), key=lambda x: x['similarity'], reverse=True)
    
    return sorted_results

def save_user_similar_items(graph_path, output_dir, top_k=10, batch_size=100, filter_own_docs=False, max_clusters=3):
    """
    为每个用户计算最相似的文档并保存
    
    Args:
        graph_path: 图数据路径
        output_dir: 输出目录
        top_k: 每个用户保存的最相似文档数量
        batch_size: 批处理大小，避免内存溢出
        filter_own_docs: 是否过滤掉用户自己的文档，默认为True
        max_clusters: 每个用户的最大聚类数量
    """
    # 加载图数据
    data = load_graph_data(graph_path)
    
    # 提取所需数据
    item_embeddings = data['item_embeddings']
    all_items = data['all_items']
    item_id_to_idx = data['item_id_to_idx']
    profile_id_to_items = data['profile_id_to_items']
    item_to_profile = data['item_to_profile']
    profile_to_user_id = data['profile_to_user_id']
    idx_to_item_id = data['idx_to_item_id']
    max_valid_idx = data['max_valid_idx']
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果文件
    final_output_file = os.path.join(output_dir, "user_similar_items_kmeans.json")
    
    # 创建结果字典
    total_users = len(profile_id_to_items)
    print(f"开始为 {total_users} 个用户计算最相似的文档...")
    
    # 创建FAISS索引用于高效相似度搜索
    print("创建FAISS索引...")
    dimension = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 内积索引（余弦相似度）
    
    # 只添加有效的节点嵌入到索引中
    valid_indices = np.array([i for i in range(len(item_embeddings)) if i in idx_to_item_id])
    print(f"添加 {len(valid_indices)} 个有效节点到FAISS索引")
    valid_embeddings = item_embeddings[valid_indices].astype('float32')
    index.add(valid_embeddings)
    
    # 创建索引映射: FAISS索引位置 -> 原始节点索引
    faiss_idx_to_node_idx = {i: idx for i, idx in enumerate(valid_indices)}
    
    # 分批处理用户
    profile_ids = list(profile_id_to_items.keys())
    processed_count = 0
    
    user_results = []

    for i in tqdm(range(0, len(profile_ids), batch_size), desc="处理用户批次"):
        batch_profile_ids = profile_ids[i:i+batch_size]
        
        for profile_id in batch_profile_ids:
            # 获取该用户的所有文档
            item_ids = profile_id_to_items[profile_id]
            
            # 使用聚类方法搜索文档
            search_results = search_docs_with_clusters(
                profile_id, 
                item_ids, 
                item_id_to_idx, 
                item_embeddings, 
                index, 
                faiss_idx_to_node_idx, 
                top_k, 
                filter_own_docs,
                max_clusters  # 添加这个参数
            )
            
            # 构建完整结果
            similar_docs = []
            for result in search_results[:top_k]:
                node_idx = result['node_idx']
                
                # 安全地获取item_id
                if node_idx not in idx_to_item_id:
                    continue
                
                item_id = idx_to_item_id[node_idx]
                
                # 安全地获取profile
                if item_id not in item_to_profile:
                    continue
                    
                similar_profile = item_to_profile[item_id]
                
                # 根据filter_own_docs参数决定是否过滤用户自己的文档
                if filter_own_docs and similar_profile == profile_id:
                    continue
                
                # 确保索引有效
                if node_idx >= len(all_items):
                    continue
                    
                similar_item = all_items[node_idx]
                
                # 获取文档所属用户ID
                if similar_profile in profile_to_user_id:
                    similar_user_id = profile_to_user_id[similar_profile]
                else:
                    similar_user_id = similar_profile
                
                similar_docs.append({
                    "node_idx": int(node_idx),
                    "item_id": item_id,
                    "title": similar_item['title'],
                    "abstract": similar_item['abstract'],
                    "similarity": float(result['similarity']),
                    "user_id": similar_user_id,
                    "cluster_weight": float(result.get('cluster_weight', 1.0))
                })
            
            # 获取真实用户ID
            if profile_id in profile_to_user_id:
                user_id = profile_to_user_id[profile_id]
            else:
                user_id = profile_id
            
            # 使用列表存储结果
            if similar_docs:
                user_results.append({
                    "user_id": int(user_id),
                    "retrieval": similar_docs,
                    "method": "kmeans_clustering"
                })
                processed_count += 1

        print(f"已处理 {min(i+batch_size, len(profile_ids))}/{len(profile_ids)} 个用户，找到结果的用户: {processed_count}")

    # 保存完整结果
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(user_results, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats = {
        'total_users': total_users,
        'users_with_similar_docs': processed_count,
        'top_k': top_k,
        'embedding_dim': dimension,
        'filter_own_docs': filter_own_docs,
        'method': 'kmeans_clustering'
    }
    
    with open(os.path.join(output_dir, "similarity_status_kmeans.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"所有用户的相似文档计算完成！结果保存在: {final_output_file}")
    print(f"共找到 {processed_count}/{total_users} 个用户的相似文档")
    return final_output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用KMeans聚类为每个用户计算相似文档并保存")
    parser.add_argument("--graph_path", type=str, required=True, help="图数据路径")
    parser.add_argument("--output_dir", type=str, default="./user_similar_items_kmeans", help="输出目录")
    parser.add_argument("--top_k", type=int, default=10, help="每个用户保存的相似文档数量")
    parser.add_argument("--batch_size", type=int, default=100, help="批处理大小")
    parser.add_argument("--filter_own_docs", type=bool, default=False, help="是否过滤用户自己的文档，默认为False")
    parser.add_argument("--max_clusters", type=int, default=3, help="每个用户的最大聚类数量")
    
    args = parser.parse_args()
    
    save_user_similar_items(
        args.graph_path, 
        args.output_dir, 
        args.top_k, 
        args.batch_size,
        args.filter_own_docs,
        args.max_clusters  # 添加这个参数
    )