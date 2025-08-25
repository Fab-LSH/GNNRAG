import torch

def load_graph_data(graph_path):
    print(f"加载图数据: {graph_path}")
    return torch.load(graph_path, map_location='cpu')

graph_data = load_graph_data("./data/LaMP_1_time/graph_output/graph.pt")

# 从节点索引到文档ID的反向映射
item_id_to_idx = graph_data['item_id_to_idx']
idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}

# 文档ID到用户ID的映射
item_to_profile = graph_data['item_to_profile'] 

# 用户ID到文档ID列表的映射
profile_id_to_items = graph_data['profile_id_to_items']

# 所有文档信息
all_items = graph_data['all_items']

def get_user_id_from_node_idx(node_idx, idx_to_item_id, item_to_profile):
    """根据节点索引获取user_id"""
    # 从节点索引获取文档ID
    item_id = idx_to_item_id[node_idx]
    
    # 从文档ID获取profile_id
    profile_id = item_to_profile[item_id]
    
    # 从profile_id提取user_id (通常格式为'profile_X')
    if profile_id.startswith("profile_"):
        user_id = profile_id.replace("profile_", "")
    else:
        user_id = profile_id
        
    return user_id, profile_id

def get_all_user_items(user_id_or_profile_id, profile_id_to_items, item_id_to_idx, all_items):
    """获取指定用户的所有文档"""
    # 确保我们有profile_id格式
    if not user_id_or_profile_id.startswith("profile_"):
        profile_id = f"profile_{user_id_or_profile_id}"
    else:
        profile_id = user_id_or_profile_id
    
    # 获取该用户的所有文档ID
    if profile_id in profile_id_to_items:
        item_ids = profile_id_to_items[profile_id]
        
        # 获取完整的文档信息
        user_items = []
        for item_id in item_ids:
            if item_id in item_id_to_idx:
                node_idx = item_id_to_idx[item_id]
                item_info = all_items[node_idx].copy()  # 复制一份避免修改原始数据
                item_info['node_idx'] = node_idx  # 添加节点索引信息
                user_items.append(item_info)
        
        return user_items
    else:
        return []
    
# 示例：从节点索引获取用户信息和文档
def process_node(node_idx, graph_data):
    # 创建必要的映射
    item_id_to_idx = graph_data['item_id_to_idx']
    idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
    item_to_profile = graph_data['item_to_profile']
    profile_id_to_items = graph_data['profile_id_to_items']
    all_items = graph_data['all_items']
    
    # 1. 获取该节点对应的user_id
    user_id, profile_id = get_user_id_from_node_idx(node_idx, idx_to_item_id, item_to_profile)
    print(f"节点 {node_idx} 对应的用户ID: {user_id} (profile_id: {profile_id})")
    
    # 2. 获取该用户的所有文档
    user_items = get_all_user_items(profile_id, profile_id_to_items, item_id_to_idx, all_items)
    print(f"用户 {user_id} 共有 {len(user_items)} 篇文档")
    
    # 3. 显示部分文档信息
    for i, item in enumerate(user_items[:3]):  # 只显示前3个
        print(f"  文档 {i+1}: ID={item['id']}, 标题={item['title'][:50]}...")
    
    return user_id, user_items

user_ids = ["700", "701", "702", "703", "704", "70119", "70502"]
for user_id in user_ids:

    profile_id = f"profile_{user_id}"  # 转换为profile_id格式

    # 获取该用户的所有文档
    user_items = get_all_user_items(profile_id, profile_id_to_items, item_id_to_idx, all_items)
    print(f"用户 {user_id} 共有 {len(user_items)} 篇文档")