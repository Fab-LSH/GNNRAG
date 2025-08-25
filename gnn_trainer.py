import os
import json
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score, precision_recall_fscore_support
import logging
import wandb
from typing import Dict, List, Tuple, Optional

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class GNNRetriever(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_type: str = 'sage',
        use_edge_attr: bool = True
    ):
        """
        GNN检索模型
        
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度
            num_layers: GNN层数
            dropout: Dropout比例
            model_type: GNN类型 ('sage', 'gat')
            use_edge_attr: 是否使用边属性
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        self.use_edge_attr = use_edge_attr
        
        # 查询编码器
        self.query_encoder = torch.nn.Sequential(
            torch.nn.Linear(768, hidden_channels),  # 假设原始查询嵌入是768维
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
        # GNN层
        self.convs = torch.nn.ModuleList()
        
        # 第一层
        if model_type == 'sage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        elif model_type == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels // 4, heads=4))
        else:
            raise ValueError(f"不支持的GNN类型: {model_type}")
        
        # 中间层
        for _ in range(num_layers - 2):
            if model_type == 'sage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif model_type == 'gat':
                self.convs.append(GATConv(hidden_channels, hidden_channels // 4, heads=4))
        
        # 最后一层
        if num_layers > 1:
            if model_type == 'sage':
                self.convs.append(SAGEConv(hidden_channels, out_channels))
            elif model_type == 'gat':
                self.convs.append(GATConv(hidden_channels, out_channels // 4, heads=4))
        
        # 如果使用边属性，添加边嵌入层
        if use_edge_attr:
            self.edge_embedding = torch.nn.Embedding(2, hidden_channels)  # 2种边类型
    
    def encode_query(self, query_embedding):
        """编码查询"""
        return self.query_encoder(query_embedding)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """前向传播"""
        # 使用边属性来增强消息传递
        edge_weight = None
        if self.use_edge_attr and edge_attr is not None:
            edge_weight = self.edge_embedding(edge_attr)
            edge_weight = edge_weight.sum(dim=1)
            edge_weight = torch.sigmoid(edge_weight)
        
        # 应用GNN层
        for i, conv in enumerate(self.convs):
            if self.model_type == 'sage':
                x = conv(x, edge_index)
            else:  # gat
                x = conv(x, edge_index)
            
            if i != len(self.convs) - 1:  # 非最后一层
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 如果指定了batch，则进行节点聚合
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class GNNTrainer:
    def __init__(
        self,
        graph_path: str,
        model_config: Dict,
        train_config: Dict,
        output_dir: str,
        device: torch.device = None
    ):
        """
        GNN训练器
        
        Args:
            graph_path: 图数据路径
            model_config: 模型配置
            train_config: 训练配置
            output_dir: 输出目录
            device: 训练设备
        """
        self.graph_path = graph_path
        self.model_config = model_config
        self.train_config = train_config
        self.output_dir = output_dir
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载图数据
        self.load_graph_data()

        # 添加：检查并转换图属性
        self.convert_graph_attributes()
        
        # 加载查询编码器
        self.query_encoder = SentenceTransformer(self.graph_data['embedding_model_name'])
        
        # 创建模型
        self.create_model()
        
        # 初始化优化器和损失函数
        self.init_optimizer()
        
        # 混合精度训练
        self.scaler = GradScaler() if train_config.get('use_amp', False) else None
        
        # 创建数据加载器
        self.create_dataloaders()
        
    def load_graph_data(self):
        """加载图数据"""
        logger.info(f"从 {self.graph_path} 加载图数据")
        self.graph_data = torch.load(self.graph_path, map_location='cpu')
        self.graph = self.graph_data['graph_data']
        
        # 打印图统计信息
        logger.info(f"图节点数: {self.graph.num_nodes}")
        logger.info(f"图边数: {self.graph.edge_index.shape[1]}")
        logger.info(f"同一profile边数: {(self.graph.edge_attr == 0).sum().item()}")
        logger.info(f"相似度边数: {(self.graph.edge_attr == 1).sum().item()}")
        
    def create_model(self):
        """创建模型"""
        in_channels = self.graph_data['embedding_dim']
        hidden_channels = self.model_config.get('hidden_channels', 256)
        out_channels = self.model_config.get('out_channels', 128)
        num_layers = self.model_config.get('num_layers', 2)
        dropout = self.model_config.get('dropout', 0.2)
        model_type = self.model_config.get('model_type', 'sage')
        use_edge_attr = self.model_config.get('use_edge_attr', True)
        
        self.model = GNNRetriever(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type,
            use_edge_attr=use_edge_attr
        ).to(self.device)
        
        logger.info(f"创建 {model_type} 模型，层数 {num_layers}")
        
    def init_optimizer(self):
        """初始化优化器和学习率调度器"""
        lr = self.train_config.get('lr', 1e-3)
        weight_decay = self.train_config.get('weight_decay', 1e-5)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        epochs = self.train_config.get('epochs', 50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs
        )


        
    def create_dataloaders(self):
        """创建数据加载器"""
        # 邻居采样参数
        num_neighbors = self.train_config.get('num_neighbors', [20, 15])
        batch_size = self.train_config.get('batch_size', 2048)
        num_workers = self.train_config.get('num_workers', 4)
        
        logger.info(f"创建数据加载器，批次大小={batch_size}，邻居采样=[{num_neighbors}]")
        
        self.graph.edge_index = self.graph.edge_index.contiguous()
        if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
            self.graph.edge_attr = self.graph.edge_attr.contiguous()
        self.graph.x = self.graph.x.contiguous()

        # 创建训练加载器
        self.train_loader = NeighborLoader(
            self.graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        # 创建验证加载器
        val_size = int(0.1 * len(self.graph.x))
        val_indices = torch.randperm(len(self.graph.x))[:val_size]
        
        self.val_loader = NeighborLoader(
            self.graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=val_indices,
            num_workers=num_workers
        )

    def convert_graph_attributes(self):
        """
        检查并转换图的属性，确保所有特征都是张量而不是列表
        """
        logger.info("检查并转换图属性...")

        # 获取图中所有属性
        for key, value in self.graph:
            # 跳过已经是张量的属性
            if isinstance(value, torch.Tensor):
                continue

            # 跳过None值
            if value is None:
                continue

            # 检查是否为列表
            if isinstance(value, list):
                logger.info(f"将属性 '{key}' 从列表转换为张量")

                try:
                    # 尝试转换为张量
                    if all(isinstance(item, (int, float)) for item in value):
                        # 数值列表
                        self.graph[key] = torch.tensor(value)
                    elif all(isinstance(item, list) for item in value):
                        # 嵌套列表，可能是二维数组
                        if all(isinstance(subitem, (int, float)) for item in value for subitem in item):
                            self.graph[key] = torch.tensor(value)
                        else:
                            # 复杂列表，跳过或执行特殊处理
                            logger.warning(f"属性 '{key}' 是复杂列表，无法直接转换为张量，将其从图中移除")
                            delattr(self.graph, key)
                    else:
                        # 其他类型的列表，跳过
                        logger.warning(f"属性 '{key}' 包含混合类型，无法转换为张量，将其从图中移除")
                        delattr(self.graph, key)
                except Exception as e:
                    logger.error(f"转换属性 '{key}' 时出错: {str(e)}")
                    logger.warning(f"从图中移除属性 '{key}'")
                    delattr(self.graph, key)

        # 确保核心属性是连续的张量
        self.graph.x = self.graph.x.contiguous()
        self.graph.edge_index = self.graph.edge_index.contiguous()

        if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
            self.graph.edge_attr = self.graph.edge_attr.contiguous()

        logger.info("图属性转换完成")
        
    def info_nce_loss(self, query_embs, doc_embs, temperature=0.1):
        """
        InfoNCE对比损失
        
        Args:
            query_embs: 查询嵌入 [batch_size, dim]
            doc_embs: 文档嵌入 [batch_size, dim]
            temperature: 温度参数
            
        Returns:
            loss: 损失值
        """
        # 归一化嵌入
        query_embs = F.normalize(query_embs, dim=1)
        doc_embs = F.normalize(doc_embs, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(query_embs, doc_embs.T) / temperature
        
        # 对角线是正例
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def generate_batch_queries(self, batch):
        """生成批次查询"""
        # 防御性编程 - 检查batch属性
        if not hasattr(batch, 'batch') or batch.batch is None:
            # 备用方案：使用n_id属性
            nodes = batch.n_id.tolist()[:32]  # 限制处理的节点数
            batch_size = len(nodes)
        else:
            # 原始方案
            batch_size = batch.batch[-1].item() + 1
            nodes = []
            for i in range(batch_size):
                mask = batch.batch == i
                if torch.any(mask):
                    nodes.append(batch.n_id[torch.where(mask)[0][0]].item())
    
        # 生成查询
        queries = []
        for node_idx in nodes:
            # 使用self.graph_data['all_items']而不是可能已被移除的item_data
            item = self.graph_data['all_items'][node_idx]
            text = f"{item['title']} {item['abstract']}"
            queries.append(text)
        
        # 编码查询
        query_embeddings = self.query_encoder.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        )
        
        return query_embeddings
        
    def train_epoch(self, epoch):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        batch_bar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch}")
        for batch in batch_bar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # 获取批次查询
            query_embeddings = self.generate_batch_queries(batch).to(self.device)
            
            # 获取批次大小 - 不再使用batch.num_graphs
            batch_size = query_embeddings.size(0)
            
            # 使用混合精度训练
            if self.scaler is not None:
                with autocast():
                    # 文档编码
                    doc_embeddings = self.model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    
                    # 查询编码
                    query_embeddings = self.model.encode_query(query_embeddings)
                    
                    # 计算损失
                    loss = self.info_nce_loss(query_embeddings, doc_embeddings)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 文档编码
                doc_embeddings = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # 查询编码
                query_embeddings = self.model.encode_query(query_embeddings)
                
                # 计算损失
                loss = self.info_nce_loss(query_embeddings, doc_embeddings)
                
                loss.backward()
                self.optimizer.step()
            
            # 使用batch_size而不是batch.num_graphs
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            batch_bar.set_postfix({"loss": loss.item()})
        
        # 修正损失计算
        epoch_loss = total_loss / total_samples
        self.scheduler.step()
        
        return epoch_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                batch = batch.to(self.device)
                
                # 获取批次查询
                query_embeddings = self.generate_batch_queries(batch).to(self.device)
                
                # 获取批次大小
                batch_size = query_embeddings.size(0)
                
                # 文档编码
                doc_embeddings = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # 查询编码
                query_embeddings = self.model.encode_query(query_embeddings)
                
                # 计算损失
                loss = self.info_nce_loss(query_embeddings, doc_embeddings)
                
                # 使用batch_size
                total_loss += loss.item() * batch_size
                total_samples += batch_size
    
        val_loss = total_loss / total_samples
        return val_loss
    
    def train(self):
        """训练模型"""
        epochs = self.train_config.get('epochs', 50)
        patience = self.train_config.get('patience', 5)
        use_wandb = self.train_config.get('use_wandb', False)
        
        if use_wandb:
            wandb.init(
                project="gnn-retrieval",
                config={
                    "model": self.model_config,
                    "train": self.train_config,
                    "graph_nodes": self.graph.num_nodes,
                    "graph_edges": self.graph.edge_index.shape[1]
                }
            )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = None
        
        start_time = time.time()
        logger.info("开始训练...")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Time: {epoch_time:.1f}s")
            
            if use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch_time": epoch_time
                })
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存模型
                if best_model_path:
                    os.remove(best_model_path)
                
                best_model_path = os.path.join(self.output_dir, f"best_model_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'model_config': self.model_config
                }, best_model_path)
                
                logger.info(f"保存最佳模型到 {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停: {patience} 个epoch内没有改善")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"训练完成，总时间: {total_time:.1f}s")
        
        if use_wandb:
            wandb.finish()
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, "final_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'model_config': self.model_config
        }, final_model_path)
        
        logger.info(f"保存最终模型到 {final_model_path}")
        
        return best_model_path

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="训练GNN检索模型")
    
    # 输入/输出参数
    parser.add_argument("--graph_path", type=str, required=True,
                        help="图数据路径 (.pt文件)")
    parser.add_argument("--output_dir", type=str, default="gnn_retrieval_model",
                        help="输出目录，用于保存模型和结果")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="sage",
                        choices=["sage", "gat"], help="GNN模型类型")
    parser.add_argument("--hidden_channels", type=int, default=256,
                        help="隐藏层维度")
    parser.add_argument("--out_channels", type=int, default=128,
                        help="输出特征维度")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="GNN层数")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout比例")
    parser.add_argument("--use_edge_attr", type=bool, default=True,
                        help="是否使用边属性")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮次")
    parser.add_argument("--patience", type=int, default=5,
                        help="早停耐心值")
    parser.add_argument("--num_neighbors", type=str, default="20,15",
                        help="每层采样的邻居数量，用逗号分隔")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载器的工作进程数")
    parser.add_argument("--use_amp", action="store_true",
                        help="是否使用混合精度训练")
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用Weights & Biases记录训练过程")
    
    # 设备参数
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备 (cpu, cuda, cuda:0, etc.)")
    
    return parser

if __name__ == "__main__":
    # 解析命令行参数
    parser = create_parser()
    args = parser.parse_args()
    # 设置随机种子
    torch.manual_seed(2025)
    np.random.seed(2025)

    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析邻居数量
    num_neighbors = [int(x) for x in args.num_neighbors.split(",")]
    
    # 创建模型配置
    model_config = {
        'model_type': args.model_type,
        'hidden_channels': args.hidden_channels,
        'out_channels': args.out_channels,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'use_edge_attr': args.use_edge_attr
    }
    
    # 创建训练配置
    train_config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'num_neighbors': num_neighbors,
        'num_workers': args.num_workers,
        'use_amp': args.use_amp,
        'use_wandb': args.use_wandb
    }
    
    # 打印配置信息
    logger.info("=== 训练配置 ===")
    logger.info(f"图数据: {args.graph_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"设备: {device}")
    logger.info(f"模型配置: {model_config}")
    logger.info(f"训练配置: {train_config}")
    logger.info("=" * 40)
    
    # 创建训练器
    trainer = GNNTrainer(
        graph_path=args.graph_path,
        model_config=model_config,
        train_config=train_config,
        output_dir=args.output_dir,
        device=device
    )
    
    # 训练模型
    best_model_path = trainer.train()
    
    logger.info("训练与文档编码完成!")