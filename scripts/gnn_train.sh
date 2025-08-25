nohup python ./GNNRAG/gnn_trainer.py \
    --graph_path ./data/LaMP_1_time/graph_output/graph.pt \
    --output_dir ./gnn_model_output \
    --model_type sage \
    --batch_size 2048 \
    --num_neighbors 20,15 \
    --epochs 50 \
    --use_amp > gnn_train.log 2>&1 &