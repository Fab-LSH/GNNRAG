#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import logging

from metrics.eval_metrics import LaMPEvaluation
from data.datasets import Seq2SeqDataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_text_template(tokenizer, prompt):
    """构建模型输入模板"""
    message = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(message,
                                         tokenize=False,
                                         add_generation_prompt=True)

def create_prompts_with_retrieval(test_data, user_similar_items, top_k=5):
    """根据检索结果构建带有相似文档的提示词"""
    prompts_data = []  # 只包含输入提示词
    evaluation_data = []  # 包含用于评估的标签
    
    for user_data in test_data:
        user_id = str(user_data.get('user_id', ''))
        
        # 从test_data中获取query和task信息
        query = user_data.get('query', user_data.get('input', ''))
        task = user_data.get('task', '')
        
        # 获取真实输出标签 - 但不传入模型
        ground_truth = user_data.get('output', user_data.get('response', ''))
        
        # 查找用户的相似文档 - 支持多种格式
        similar_docs = []
        
        # 尝试不同的用户ID格式
        possible_keys = [
            user_id,
            f"user_id: {user_id}",
            f"user_id:{user_id}",
            str(user_id)
        ]
        
        found_key = None
        for key in possible_keys:
            if key in user_similar_items:
                similar_docs = user_similar_items[key]
                found_key = key
                break
        
        # 如果user_similar_items是列表格式，查找匹配的用户
        if not similar_docs and isinstance(user_similar_items, list):
            for item in user_similar_items:
                if item.get('user_id') == user_id:
                    similar_docs = item.get('retrival', item.get('retrieval', []))
                    break
        
        # 限制文档数量
        if isinstance(similar_docs, list):
            similar_docs = similar_docs[:top_k]
        else:
            similar_docs = []
        
        # 构建上下文
        context_parts = []
        for doc in similar_docs:
            if isinstance(doc, dict):
                title = doc.get('title', '')
                abstract = doc.get('abstract', '')
                context_parts.append(f"Title: {title}\nAbstract: {abstract}")
        
        context = "\n\n".join(context_parts)
        
        # 根据任务类型构建不同的提示词
        if task.startswith('LaMP_1'):
            prompt = f"Based on the following user query and retrieved documents, classify the query as either [1] or [2]. Please just answer with '[1]' or '[2]' without explanation.\n\nQuery: {query}\n\nRetrieved documents:\n{context}\n\nClassification:"
        elif task.startswith('LaMP_2'):
            prompt = f"Based on the following user query and retrieved documents, predict the most appropriate tag.\n\nQuery: {query}\n\nRetrieved documents:\n{context}\n\nTag:"
        elif task.startswith('LaMP_3'):
            prompt = f"Based on the following user query and retrieved documents, predict a rating from 1 to 5.\n\nQuery: {query}\n\nRetrieved documents:\n{context}\n\nRating:"
        elif task.startswith('LaMP_4') or task.startswith('LaMP_5'):
            prompt = f"Based on the following user query and retrieved documents, generate an appropriate title.\n\nQuery: {query}\n\nRetrieved documents:\n{context}\n\nTitle:"
        elif task.startswith('LaMP_7'):
            prompt = f"Based on the following user query and retrieved documents, generate an appropriate tweet.\n\nQuery: {query}\n\nRetrieved documents:\n{context}\n\nTweet:"
        else:
            prompt = f"Based on the following user query and retrieved documents, generate a response.\n\nQuery: {query}\n\nRetrieved documents:\n{context}\n\nResponse:"
        
        # 输入模型的数据（不包含答案）
        prompts_data.append({
            "user_id": user_id,
            "query": query,
            "prompt": prompt,
            "found_similar_docs": len(similar_docs)
        })
        
        # 用于评估的数据（包含答案）
        evaluation_data.append({
            "user_id": user_id,
            "task": task,
            "query": query,
            "ground_truth": ground_truth
        })
    
    return prompts_data, evaluation_data

def load_post_process_function(task):
    """根据任务加载相应的后处理函数"""
    
    def post_process(predictions):
        if task.startswith("LaMP_1"):
            # 二分类任务
            processed = []
            for pred in predictions:
                pred_str = str(pred).strip()
                if "[1]" in pred_str or (pred_str.startswith("1") and len(pred_str) <= 3):
                    processed.append("1")
                elif "[2]" in pred_str or (pred_str.startswith("2") and len(pred_str) <= 3):
                    processed.append("2")
                else:
                    # 默认分类
                    processed.append("1")
            return processed
        
        elif task.startswith("LaMP_2"):
            # 标签预测任务
            import re
            labels = [
                'sci-fi', 'based on a book', 'comedy', 'action',
                'twist ending', 'dystopia', 'dark comedy', 'classic',
                'psychology', 'fantasy', 'romance', 'thought-provoking',
                'social commentary', 'violence', 'true story'
            ]
            
            processed = []
            for pred in predictions:
                pred_lower = str(pred).lower()
                found = False
                for label in labels:
                    if re.search(r'\b' + re.escape(label) + r'\b', pred_lower):
                        processed.append(label)
                        found = True
                        break
                if not found:
                    # 如果没找到匹配的标签，返回第一个标签作为默认
                    processed.append(labels[0])
            return processed
        
        elif task.startswith("LaMP_3"):
            # 评分预测任务
            processed = []
            for pred in predictions:
                pred_str = str(pred).strip()
                # 查找1-5之间的数字
                for rating in ["5", "4", "3", "2", "1"]:
                    if rating in pred_str:
                        processed.append(rating)
                        break
                else:
                    # 默认评分
                    processed.append("3")
            return processed
        
        elif task.startswith("LaMP_4") or task.startswith("LaMP_5"):
            # 标题生成任务
            import re
            processed = []
            for pred in predictions:
                pred = str(pred).strip()
                # 尝试提取 {'title': 'xxx'} 格式
                title_match = re.search(r"['\"]{1}title['\"]{1}\s*:\s*['\"]{1}(.*?)['\"]{1}", pred)
                if title_match:
                    processed.append(title_match.group(1))
                else:
                    # 直接使用预测文本
                    processed.append(pred)
            return processed
        
        elif task.startswith("LaMP_7"):
            # 推文生成任务
            processed = []
            for pred in predictions:
                pred = str(pred).strip()
                # 移除可能的格式标记，只保留推文内容
                pred = pred.replace('{"tweet": "', '').replace('"}', '').strip()
                processed.append(pred)
            return processed

        else:
            # 默认不处理
            return [str(pred).strip() for pred in predictions]
    
    return post_process

def main():
    parser = argparse.ArgumentParser()
    
    # 设备配置
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default='0,1')
    parser.add_argument("--random_seed", type=int, default=2025)
    
    # 数据路径
    parser.add_argument("--test_path", type=str, required=True, 
                        help="测试数据路径")
    parser.add_argument("--similar_items_path", type=str, required=True, 
                        help="检索到的相似文档路径")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", 
                        help="输出目录")
    
    # 模型配置
    parser.add_argument("--model_name", default="Meta-Llama-3-8B-Instruct",
                        choices=['Meta-Llama-3-8B-Instruct', 'Qwen2-7B-Instruct',
                                'Qwen2-7B', 'Meta-Llama-3-8B'])
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径，如果不指定则根据model_name自动设置")
    
    # 任务配置
    parser.add_argument("--task", default=None, 
                        help="任务类型，如果不指定则从测试数据中获取")
    
    # 生成配置
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--cutoff_len", type=int, default=20000)
    parser.add_argument("--top_k", type=int, default=3,
                        help="每个用户使用的相似文档数量")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000000)
    
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES
    
    # 自动设置模型路径
    if not opts.model_path:
        opts.model_path = f'../models/{opts.model_name}'
    
    # 输出配置信息
    for flag, value in opts.__dict__.items():
        print('{}: {}'.format(flag, value))
    
    # 创建输出目录
    os.makedirs(opts.output_dir, exist_ok=True)
    
    # 加载测试数据
    logger.info(f"加载测试数据: {opts.test_path}")
    with open(opts.test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 加载相似文档
    logger.info(f"加载相似文档: {opts.similar_items_path}")
    with open(opts.similar_items_path, 'r', encoding='utf-8') as f:
        user_similar_items = json.load(f)
    
    logger.info(f"加载了 {len(test_data)} 条测试数据")
    logger.info(f"加载了 {len(user_similar_items)} 个用户的相似文档")
    
    # 确定任务类型
    if not opts.task:
        # 尝试从测试数据的文件名或第一条数据中获取任务类型
        if 'task' in test_data[0]:
            opts.task = test_data[0]['task']
        else:
            # 从文件路径中推断任务类型
            if 'LaMP_1' in opts.test_path:
                opts.task = 'LaMP_1_time'
            elif 'LaMP_2' in opts.test_path:
                opts.task = 'LaMP_2'
            elif 'LaMP_3' in opts.test_path:
                opts.task = 'LaMP_3'
            elif 'LaMP_4' in opts.test_path:
                opts.task = 'LaMP_4'
            elif 'LaMP_5' in opts.test_path:
                opts.task = 'LaMP_5'
            elif 'LaMP_7' in opts.test_path:
                opts.task = 'LaMP_7'
            else:
                opts.task = 'LaMP_1_time'  # 默认任务类型
    
    logger.info(f"任务类型: {opts.task}")
    
    # 应用数据范围限制
    test_data_slice = test_data[opts.begin_idx:min(opts.end_idx, len(test_data))]
    
    # 准备带有检索文档的提示词 - 分离输入和评估数据
    prompts_data, evaluation_data = create_prompts_with_retrieval(
        test_data_slice, user_similar_items, opts.top_k)
    
    # 统计检索成功率
    found_docs = sum(1 for item in prompts_data if item['found_similar_docs'] > 0)
    logger.info(f"成功检索到相似文档的用户数: {found_docs}/{len(prompts_data)} ({found_docs/len(prompts_data)*100:.2f}%)")
    
    # 调试信息：检查ground_truth
    ground_truth_list = [item['ground_truth'] for item in evaluation_data]
    logger.info(f"真实标签示例: {ground_truth_list[:5]}")
    logger.info(f"标签唯一值: {set(ground_truth_list)}")
    empty_labels = sum(1 for label in ground_truth_list if not label)
    logger.info(f"空标签数量: {empty_labels}/{len(ground_truth_list)}")
    
    # 加载分词器
    logger.info(f"加载分词器: {opts.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(opts.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 准备提示词 - 只包含问题和检索文档，不包含答案
    prompts = [
        get_text_template(tokenizer, item['prompt'])
        for item in prompts_data
    ]
    
    # 加载LLM模型
    logger.info(f"加载模型: {opts.model_path}")
    llm = LLM(model=opts.model_path,
              gpu_memory_utilization=0.8,
              max_seq_len_to_capture=opts.cutoff_len,
              max_model_len=opts.cutoff_len)
    
    # 设置生成参数
    sampling_params = SamplingParams(
        seed=opts.random_seed,
        temperature=0,
        best_of=1,
        max_tokens=opts.max_new_tokens
    )
    
    # 批量生成
    logger.info("开始生成输出...")
    model_outputs = []
    for idx in tqdm(range(0, len(prompts), opts.batch_size), desc="Generating"):
        batch_prompts = prompts[idx:idx + opts.batch_size]
        batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        model_outputs.extend(batch_outputs)
    
    model_preds = [x.outputs[0].text for x in model_outputs]
    
    # 后处理生成结果
    logger.info("后处理生成结果...")
    post_process_fun = load_post_process_function(opts.task)
    processed_preds = post_process_fun(model_preds)
    
    # 调试信息：检查预测结果
    logger.info(f"原始预测示例: {model_preds[:5]}")
    logger.info(f"处理后预测示例: {processed_preds[:5]}")
    logger.info(f"预测唯一值: {set(processed_preds)}")
    
    # ===== 参考generate.py的评估方式 =====
    logger.info("评估生成结果...")
    try:
        eval_method = LaMPEvaluation(opts.task)
        pred_scores = eval_method.compute_metrics(processed_preds, ground_truth_list, avg=False)
        logger.info("使用LaMPEvaluation计算成功")
    except Exception as e:
        logger.warning(f"LaMPEvaluation计算失败: {e}")
        # 如果LaMPEvaluation失败，创建空的评分字典
        pred_scores = {
            "accuracy": [0.0] * len(processed_preds),
            "f1": [0.0] * len(processed_preds)
        }
    
    # 准备保存结果
    generate_results = []
    all_scores = None
    
    # ===== 完全按照generate.py的方式处理结果 =====
    for idx in tqdm(range(len(prompts)), desc="Processing results"):
        prompt_data = prompts_data[idx]
        eval_data = evaluation_data[idx]
        
        save_dict = {
            "user_id": prompt_data['user_id'],
            "task": eval_data['task'],
            "query": prompt_data['query'],
            "input": prompt_data['prompt'],
            "output": model_preds[idx],
            "predict": processed_preds[idx],
            "label": eval_data['ground_truth'],
            "found_similar_docs": prompt_data['found_similar_docs']
        }
        
        # 添加评分 - 按照generate.py的方式
        scores = {k: v[idx] for k, v in pred_scores.items()}
        if all_scores is None:
            all_scores = {k: [v] for k, v in scores.items()}
        else:
            for k in all_scores.keys():
                all_scores[k].append(scores[k])
        
        save_dict.update(scores)
        generate_results.append(save_dict)
    
    # 更新end_idx以匹配实际处理的数据量
    opts.end_idx = opts.begin_idx + len(prompts_data)
    
    # 保存详细结果 - 文件命名方式参考generate.py
    predictions_file = os.path.join(opts.output_dir, f"predictions_{opts.begin_idx}-{opts.end_idx}.json")
    with open(predictions_file, "w", encoding="utf-8") as file:
        json.dump(generate_results, file, indent=4, ensure_ascii=False)
    
    # 计算并保存平均评分 - 完全按照generate.py的方式
    mean_scores = {k: float(np.mean(v)) for k, v in all_scores.items()}
    logger.info("=== 评估结果 ===")
    for k, v in mean_scores.items():
        logger.info(f"{k}: {v:.4f}")
    
    # 保存平均分数 - 文件命名方式参考generate.py
    scores_file = os.path.join(opts.output_dir, f"scores_{opts.begin_idx}-{opts.end_idx}.json")
    with open(scores_file, "w", encoding="utf-8") as file:
        json.dump(mean_scores, file, indent=4, ensure_ascii=False)
    
    logger.info(f"评估完成！详细结果已保存到: {predictions_file}")
    logger.info(f"评分已保存到: {scores_file}")
    
    return opts

if __name__ == "__main__":
    main()