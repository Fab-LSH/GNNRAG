import json
import os

def count_missing_users(test_question_path, user_similar_items_path):
    """
    统计在test_question.json中有多少user_id没有出现在user_similiar_items中
    
    参数:
    test_question_path (str): test_question.json文件的路径
    user_similar_items_path (str): user_similiar_items文件的路径或格式类似的列表文件
    
    返回:
    tuple: (缺失用户数量, 缺失用户ID列表, 统计信息字典)
    """
    # 读取test_question.json文件
    with open(test_question_path, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)
    
    # 读取user_similar_items文件
    with open(user_similar_items_path, 'r', encoding='utf-8') as f:
        user_similar_items = json.load(f)
    
    # 提取所有test_question中的user_id
    test_user_ids = set()
    for question in test_questions:
        if 'user_id' in question:
            test_user_ids.add(question['user_id'])
    
    # 获取user_similar_items中的user_id
    retrieved_user_ids = set()
    
    # 判断文件格式并相应提取user_id
    if isinstance(user_similar_items, dict):
        # 如果是字典格式，直接获取键
        retrieved_user_ids = set(user_similar_items.keys())
    elif isinstance(user_similar_items, list):
        # 如果是列表格式，从每个项目中提取user_id
        for item in user_similar_items:
            if 'user_id' in item:
                retrieved_user_ids.add(int(item['user_id']))
    print(len(retrieved_user_ids))
    print(list(retrieved_user_ids)[:5])
    print(len(test_user_ids))
    print(list(test_user_ids)[:5])
    
    # 找出未出现的user_id
    missing_user_ids = test_user_ids - retrieved_user_ids
    
    # 统计信息
    stats = {
        'total_test_users': len(test_user_ids),
        'retrieved_users': len(retrieved_user_ids & test_user_ids),
        'missing_users': len(missing_user_ids),
        'coverage_percentage': round((len(test_user_ids) - len(missing_user_ids)) / len(test_user_ids) * 100, 2) if test_user_ids else 0
    }
    
    return len(missing_user_ids), list(missing_user_ids), stats

# 使用示例
if __name__ == "__main__":
    test_path = "./data/LaMP_1_time/test/test_questions.json"
    # similar_items_path = "../CFRAG-raw/Meta-Llama-3-8B-Instruct_outputs/LaMP_1_time/dev/recency/bge-base-en-v1.5_5/bge-reranker-base/20250709-171400_rerank_5/20250709-163653_user-6_20250709-150551.json"
    similar_items_path = "./data/LaMP_1_time/dev/dev_questions.json"
    print(os.getcwd())
    
    count, missing_ids, statistics = count_missing_users(test_path, similar_items_path)
    
    print(f"总共有 {count} 个用户未被检索到")
    print(f"统计信息: {statistics}")
    if len(missing_ids) > 0 and len(missing_ids) <= 5:
        print(f"未检索到的用户ID: {missing_ids}")
    elif len(missing_ids) > 5:
        print(f"未检索到的用户ID示例: {missing_ids[:5]}...")