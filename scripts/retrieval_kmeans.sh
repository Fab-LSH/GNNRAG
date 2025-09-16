nohup python ./GNNRAG/document_retrieval_kmeans.py \
    --graph_path ./data/LaMP_1_time/graph_output/graph.pt \
    --output_dir ./similiar_items_kmeans/top_5_cluster_5 \
    --top_k 5 \
    --batch_size 128 \
    --filter_own_docs False \
    --max_clusters 5 > ./retrieval_kmeans/top_5_cluster_5.log 2>&1 &