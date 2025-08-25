import torch
import argparse
from tqdm import tqdm
import os
import json
import numpy as np
import faiss

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

def save_user_similar_items(graph_path, output_dir, top_k=10, batch_size=100):
    """
    为每个用户计算最相似的文档并保存
    
    Args:
        graph_path: 图数据路径
        output_dir: 输出目录
        top_k: 每个用户保存的最相似文档数量
        batch_size: 批处理大小，避免内存溢出
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
    final_output_file = os.path.join(output_dir, "user_similar_items.json")
    
    # 创建结果字典
    result_dict = {}
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
            # 获取该用户的所有文档索引
            item_ids = profile_id_to_items[profile_id]
            user_item_indices = []
            for item_id in item_ids:
                if item_id in item_id_to_idx:
                    user_item_indices.append(item_id_to_idx[item_id])
            
            if not user_item_indices:
                continue
            
            # 计算用户embedding（所有节点embedding的平均值）
            user_embeddings = item_embeddings[user_item_indices]
            user_embedding = np.mean(user_embeddings, axis=0).reshape(1, -1).astype('float32')
            
            # 查找top_k个最相似的节点
            distances, indices = index.search(user_embedding, top_k + len(user_item_indices))
            
            # 过滤掉属于同一用户的节点
            similar_docs = []
            
            for j, faiss_idx in enumerate(indices[0]):
                if faiss_idx >= len(faiss_idx_to_node_idx):
                    continue
                
                # 转换FAISS索引到原始节点索引
                idx = faiss_idx_to_node_idx[faiss_idx]
                
                # 安全地获取item_id
                if idx not in idx_to_item_id:
                    continue
                
                similar_item_id = idx_to_item_id[idx]
                
                # 安全地获取profile
                if similar_item_id not in item_to_profile:
                    continue
                    
                similar_profile = item_to_profile[similar_item_id]
                
                # 只保留不同用户的文档
                if similar_profile != profile_id:
                    # 确保索引有效
                    if idx >= len(all_items):
                        continue
                        
                    similar_item = all_items[idx]
                    
                    # 获取文档所属用户ID
                    if similar_profile in profile_to_user_id:
                        similar_user_id = profile_to_user_id[similar_profile]
                    else:
                        similar_user_id = similar_profile
                    
                    similar_docs.append({
                        "node_idx": int(idx),
                        "item_id": similar_item_id,
                        "title": similar_item['title'],
                        "abstract": similar_item['abstract'],
                        "similarity": float(distances[0][j]),
                        "user_id": similar_user_id
                    })
                
                if len(similar_docs) >= top_k:
                    break
            
            # 获取真实用户ID
            if profile_id in profile_to_user_id:
                user_id = profile_to_user_id[profile_id]
            else:
                user_id = profile_id
            
            # 使用列表存储结果
            if similar_docs:
                user_results.append({
                    "user_id": int(user_id),
                    "retrieval": similar_docs
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
        'embedding_dim': dimension
    }
    
    with open(os.path.join(output_dir, "similarity_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"所有用户的相似文档计算完成！结果保存在: {final_output_file}")
    print(f"共找到 {processed_count}/{total_users} 个用户的相似文档")
    return final_output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为每个用户计算相似文档并保存")
    parser.add_argument("--graph_path", type=str, required=True, help="图数据路径")
    parser.add_argument("--output_dir", type=str, default="./user_similar_items", help="输出目录")
    parser.add_argument("--top_k", type=int, default=10, help="每个用户保存的相似文档数量")
    parser.add_argument("--batch_size", type=int, default=100, help="批处理大小")
    
    args = parser.parse_args()
    
    save_user_similar_items(
        args.graph_path, 
        args.output_dir, 
        args.top_k, 
        args.batch_size
    )
