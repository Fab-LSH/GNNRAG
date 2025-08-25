nohup python ./GNNRAG/document_retrival.py \
    --graph_path ./data/LaMP_1_time/graph_output/graph.pt \
    --output_dir ./user_similar_items_top1 \
    --batch_size 128 \
    --top_k 1 > retrive_similar_item_top1.log 2>&1 &