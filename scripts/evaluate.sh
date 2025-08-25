export CUDA_VISIBLE_DEVICES=1
nohup python ./generation/generate.py \
    --task LaMP_1_time \
    --input_path './data/LaMP_1_time' \
    --file_name 'llm_input_retrival' \
    --model_name 'Qwen2-7B-Instruct' \
    > evaluate.log 2>&1 &