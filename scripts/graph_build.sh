nohup python ./GNNRAG/graph_builder.py \
    --json_file_path "./data/LaMP_1_time/train/train_questions.json" \
    --output_dir "./data/LaMP_1_time/graph_output" \
    --embedding_model "../models/bge-base-en-v1.5" \
    --embedding_dim "128" \
    --similarity_threshold 0.8 \
    --batch_size 512 > graph_build.log 2>&1 &