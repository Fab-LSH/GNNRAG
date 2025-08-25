#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import argparse

def add_train_outputs_to_test_questions(test_questions_path, train_questions_path, train_outputs_path, output_path=None):
    """
    将train_outputs.json中的output字段添加到test_questions.json中对应user_id的条目
    
    参数:
    test_questions_path (str): test_questions.json文件路径
    train_questions_path (str): train_questions.json文件路径（用于建立user_id映射）
    train_outputs_path (str): train_outputs.json文件路径
    output_path (str, optional): 输出文件路径，默认为None(覆盖原文件)
    
    返回:
    bool: 操作是否成功
    """
    # 读取测试问题数据
    with open(test_questions_path, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)
    
    # 读取训练问题数据
    with open(train_questions_path, 'r', encoding='utf-8') as f:
        train_questions = json.load(f)
    
    # 读取输出数据
    with open(train_outputs_path, 'r', encoding='utf-8') as f:
        train_outputs_data = json.load(f)
    
    # 确保golds列表存在
    if 'golds' not in train_outputs_data:
        print("错误: train_outputs.json 中没有找到 'golds' 字段")
        return False
    
    train_outputs = train_outputs_data['golds']
    
    # 检查训练数据和输出长度是否匹配
    if len(train_questions) != len(train_outputs):
        print(f"警告: train_questions 和 train_outputs 长度不匹配: {len(train_questions)} vs {len(train_outputs)}")
        print("继续处理，但可能会导致数据不一致")
    
    # 创建user_id到output的映射
    user_id_to_output = {}
    for i, question in enumerate(train_questions):
        if i < len(train_outputs) and 'user_id' in question:
            user_id_to_output[question['user_id']] = train_outputs[i]['output']
    
    # 为test_questions添加output字段
    added_count = 0
    missing_count = 0
    for question in test_questions:
        if 'user_id' in question and question['user_id'] in user_id_to_output:
            question['output'] = user_id_to_output[question['user_id']]
            added_count += 1
        else:
            missing_count += 1
    
    # 保存更新后的数据
    if output_path is None:
        output_path = test_questions_path
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_questions, f, ensure_ascii=False, indent=2)
    
    print(f"已成功为 {added_count} 条测试问题添加output字段")
    if missing_count > 0:
        print(f"警告: {missing_count} 条测试问题未找到对应的output")
    print(f"更新后的数据已保存至: {output_path}")
    
    return True

# 合并profile和retrival
def merge(test_questions_path: str, 
          similar_items_path: str, 
          output_path: str) -> None:
    """
    将similar_items中的retrieval直接作为一个新键插入到test_questions的每个样本（user_id对应），不拼接到input。
    """
    # 读取文件
    with open(test_questions_path, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)
    
    with open(similar_items_path, 'r', encoding='utf-8') as f:
        similar_items = json.load(f)
    
    # 创建user_id到相似文档的映射
    user_to_similar = {}
    if isinstance(similar_items, dict):
        user_to_similar = similar_items
    elif isinstance(similar_items, list):
        for item in similar_items:
            user_id = str(item.get('user_id', ''))
            retrieval_data = item.get('retrieval', item.get('retrival', []))
            if user_id:
                user_to_similar[user_id] = retrieval_data

    # 处理每个测试问题
    for item in test_questions:
        user_id = str(item.get('user_id', ''))
        # 只保留top_k个检索结果
        similar_docs = user_to_similar.get(user_id, [])
        if similar_docs:
            item['retrieval'] = similar_docs
        else:
            item['retrieval'] = []

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_questions, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成，结果保存到: {output_path}")

# 只有retrieval
def merge2(retrieval_path: str, input_output_path: str, output_path: str = None) -> None:
    """
    合并检索结果和输入输出数据
    
    Args:
        retrival_path (str): 包含检索结果的JSON文件路径
        input_output_path (str): 包含input和output数据的JSON文件路径
        output_path (str, optional): 输出文件路径，默认为None（覆盖retrival_path）
    """
    try:
        # 加载检索结果数据
        with open(retrieval_path, 'r', encoding='utf-8') as f:
            retrivals = json.load(f)
        
        print(f"加载了 {len(retrivals)} 条检索结果数据")
        
        # 加载 input 和 output 数据
        with open(input_output_path, 'r', encoding='utf-8') as f:
            input_output_data = json.load(f)
        
        print(f"加载了 {len(input_output_data)} 条输入输出数据")
        
        # 创建 user_id 到 input 和 output 的映射
        user_id_to_input_output = {}
        for item in input_output_data:
            user_id = item.get('user_id')
                
            input_text = item.get('input')
            output_text = item.get('output')
            if user_id is not None and input_text is not None and output_text is not None:
                user_id_to_input_output[user_id] = {
                    'input': input_text,
                    'output': output_text
                }
        
        print(f"创建了 {len(user_id_to_input_output)} 个用户ID映射")
        
        # 准备结果数据
        result_data = []
        
        # 更新检索数据并添加 input、output 和 query 键
        updated_count = 0
        missing_count = 0

        for retrival in retrivals:
            user_id = retrival.get('user_id')
                
            new_entry = {}
            
            # 复制检索数据的所有键值，将'retrival'键名改为'retrieval'
            for key, value in retrival.items():
                if key == 'retrival':
                    new_entry['retrieval'] = value
                else:
                    new_entry[key] = value
                
            if user_id in user_id_to_input_output:
                # 添加input和output
                new_entry['input'] = user_id_to_input_output[user_id]['input']
                new_entry['output'] = user_id_to_input_output[user_id]['output']
                
                # 从input中提取[1]和[2]的内容作为query
                input_text = user_id_to_input_output[user_id]['input']
                
                try:
                    # 使用正则表达式提取引号中的内容
                    pattern = r'"(.*?)"'
                    titles = re.findall(pattern, input_text)
                    
                    # 打印调试信息
                    if len(titles) < 3:
                        print(f"警告: 用户ID {user_id} 的titles长度不足: {len(titles)}")
                        print(f"Input文本前100个字符: {input_text[:100]}...")
                        print(f"提取到的titles: {titles}")
                    
                    # 确保titles有足够的元素
                    if len(titles) >= 3:
                        new_entry['query'] = f"'title': '{titles[1]}'  'title': '{titles[2]}'"
                    elif len(titles) == 2:
                        new_entry['query'] = f"'title': '{titles[0]}'  'title': '{titles[1]}'"
                    else:
                        new_entry['query'] = ""
                        
                except Exception as e:
                    print(f"处理user_id={user_id}的query时出错: {e}")
                    new_entry['query'] = ""
                
                updated_count += 1
            else:
                missing_count += 1
                continue  # 跳过没有找到input/output的条目
            
            result_data.append(new_entry)
        
        # 如果未指定输出路径，使用检索路径
        if output_path is None:
            output_path = retrival_path
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"已成功更新 {updated_count} 条记录，结果保存至: {output_path}")
        print(f"最终输出文件包含 {len(result_data)} 条记录")
        
        if missing_count > 0:
            print(f"警告: {missing_count} 条记录未找到对应的 input 和 output")
    
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except json.JSONDecodeError as e:
        print(f"无法解析JSON文件: {e}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 把profile转变为retrieval
def convert(input: str, output: str):
    with open(input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def replace_key(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_key = 'retrieval' if k == 'profile' else k
                new_obj[new_key] = replace_key(v)
                pattern = r'"(.*?)"'
                if k == 'input':
                    titles = re.findall(pattern, v)
                    new_obj['query'] = f"'title': '{titles[1]}'  'title': '{titles[2]}'"

            return new_obj
        elif isinstance(obj, list):
            return [replace_key(item) for item in obj]
        else:
            return obj

    new_data = replace_key(data)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"结果保存至{output}")

# 生成zeroshot文件
def zeroshot(input: str, output: str):
    with open(input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if 'retrieval' in item:
            item['retrieval'] = []

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存至发{output}")

if __name__ == "__main__":
    # 设置文件路径
    test_questions_path = "./data/LaMP_1_time/test/test_questions.json"
    train_questions_path = "./data/LaMP_1_time/train/train_questions.json"
    train_outputs_path = "./data/LaMP_1_time/train/train_outputs.json"
    
    # 如果希望生成新文件而不是覆盖原文件，可以设置output_path
    output_path = "./data/LaMP_1_time/test/test_questions_with_output.json"
    
    # 执行添加操作
    # add_train_outputs_to_test_questions(test_questions_path, train_questions_path, train_outputs_path, output_path)
    
    test_questions = "./data/LaMP_1_time/dev/recency/rank_merge.json"
    similar_items = "./user_similar_items_top1/user_similar_items.json"
    output = "./data/LaMP_1_time/dev/llm_input_retrieval_top1.json"
    # merge(test_questions, similar_items, output)

    merge2(similar_items, test_questions, output)
    # convert("./data/LaMP_1_time/dev/recency/rank_merge.json", "./data/LaMP_1_time/dev/progileToretrieval.json")

    # zeroshot("./data/LaMP_1_time/dev/profileToretrieval.json", "./data/LaMP_1_time/dev/zeroshot.json")