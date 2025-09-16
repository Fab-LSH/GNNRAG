nohup python ./GNNRAG/document_retrival_raw.py \
    --graph_path ./data/LaMP_1_time/graph_output/graph.pt \
    --output_dir ./filter_own_False/user_similar_items_top3 \
    --batch_size 128 \
    --filter_own_docs False \
    --top_k 3 > ./filter_own_False/retrive_similar_item_top3.log 2>&1 &