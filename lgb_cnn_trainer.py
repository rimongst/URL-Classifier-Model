# -*- coding: utf-8 -*-
"""
LightGBM + CNN 
"""
import json
import pickle
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse, parse_qs, unquote
import os, gc, re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """小数据集优化配置"""
    # TF-IDF - 减少特征数
    max_tfidf_features: int = 8000  # 15000 -> 8000
    tfidf_ngram_range: Tuple[int, int] = (1, 4)  # (1,5) -> (1,4)
    
    # CNN - 大幅简化
    cnn_embedding_dim: int = 32  # 64 -> 32
    cnn_num_filters: int = 64  # 128 -> 64
    cnn_kernel_sizes: List[int] = None  # [3, 4]
    cnn_dropout: float = 0.6  # 0.5 -> 0.6 (更强正则化)
    cnn_max_length: int = 128
    cnn_epochs: int = 30  # 增加epoch
    cnn_batch_size: int = 16  # 64 -> 16 (小batch更稳定)
    cnn_learning_rate: float = 0.0005  # 0.001 -> 0.0005 (更小学习率)
    cnn_weight_decay: float = 0.001  # L2正则化
    
    # LightGBM - 防止过拟合
    lgb_n_estimators: int = 200  # 500 -> 200
    lgb_learning_rate: float = 0.03  # 0.05 -> 0.03
    lgb_max_depth: int = 5  # 8 -> 5
    lgb_num_leaves: int = 20  # 80 -> 20
    lgb_subsample: float = 0.7  # 0.85 -> 0.7
    lgb_colsample_bytree: float = 0.7  # 0.85 -> 0.7
    lgb_reg_alpha: float = 5.0  # 2.0 -> 5.0
    lgb_reg_lambda: float = 5.0  # 3.0 -> 5.0
    lgb_min_child_samples: int = 10  # 更严格
    
    random_state: int = 42
    validation_split: float = 0.25  # 0.2 -> 0.25 (更多验证数据)
    device: Optional[str] = None
    
    remove_duplicates: bool = True
    url_normalization: bool = True
    
    def __post_init__(self):
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 4]  # 减少到2个尺度
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CATEGORIES: Dict[str, str] = {
    'article_page': '文章页',
    'article_list': '文章列表',
    'product_detail': '产品详情',
    'product_list': '产品列表',
    'brand_info': '品牌',
    'irrelevant': '无关页面'
}

_ALLOWED_LABELS = frozenset(CATEGORIES.keys())
MERGE_TO_IRRELEVANT = {"irrelevant", "legal", "account", "commerce"}
OTHER_RE = re.compile(
    r"/(privacy|terms|legal|mentions[-_]?legales|rgpd|gdpr)\b|"
    r"/(login|signin|signup|register|account|my[-_]?account)\b|"
    r"/(cart|checkout|panier|paiement|basket)\b",
    re.I
)


def map_label(label: str, url: str) -> str:
    if label in MERGE_TO_IRRELEVANT:
        return "irrelevant"
    if OTHER_RE.search(url):
        return "irrelevant"
    return label


# ==================== 简化的CNN ====================

class SimplifiedTextCNN(nn.Module):
    """简化的TextCNN（防止过拟合）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int,
                 kernel_sizes: List[int], num_classes: int, dropout: float = 0.6):
        super(SimplifiedTextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 减少卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes
        ])
        
        # 批归一化（稳定训练）
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 简化的输出层
        feature_dim = num_filters * len(kernel_sizes)
        self.fc_features = nn.Linear(feature_dim, 128)  # 256 -> 128
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x, return_features=False):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(embedded)
            conv_out = bn(conv_out)  # 批归一化
            conv_out = F.relu(conv_out)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        concat = torch.cat(conv_outputs, dim=1)
        concat = self.dropout(concat)
        
        features = F.relu(self.fc_features(concat))
        features = self.dropout(features)
        
        if return_features:
            return features
        
        out = self.fc_out(features)
        return out


class URLDataset(Dataset):
    def __init__(self, encoded_urls, labels=None):
        self.encoded_urls = encoded_urls
        self.labels = labels
    
    def __len__(self):
        return len(self.encoded_urls)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.encoded_urls[idx], self.labels[idx]
        return self.encoded_urls[idx]


class URLEncoder:
    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.char2idx = {}
        self.vocab_size = 0
        
    def build_vocab(self, urls: List[str]):
        chars = set()
        for url in urls:
            chars.update(url.lower())
        
        self.char2idx = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(sorted(chars), start=2):
            self.char2idx[char] = i
        
        self.vocab_size = len(self.char2idx)
        logger.info(f"词汇表大小: {self.vocab_size}")
    
    def encode(self, urls: List[str]) -> np.ndarray:
        encoded = []
        for url in urls:
            url = url.lower()[:self.max_length]
            indices = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in url]
            if len(indices) < self.max_length:
                indices += [self.char2idx['<PAD>']] * (self.max_length - len(indices))
            encoded.append(indices)
        return np.array(encoded, dtype=np.int64)


class URLNormalizer:
    @staticmethod
    def normalize(url: str) -> str:
        try:
            url = unquote(url).lower()
            tracking_params = {'utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid'}
            parsed = urlparse(url)
            if parsed.query:
                params = parse_qs(parsed.query)
                filtered = {k: v for k, v in params.items() if k not in tracking_params}
                if filtered:
                    query_str = '&'.join(f"{k}={v[0]}" for k, v in filtered.items())
                    url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query_str}"
                else:
                    url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if url.endswith('/') and url.count('/') > 3:
                url = url.rstrip('/')
            return url
        except:
            return url.lower()


# ==================== 特征提取 ====================

class EnhancedFeatureExtractor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.normalizer = URLNormalizer()
        self.tfidf_url = None
        self.tfidf_path = None
        self.ohe_location = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        self.scaler = StandardScaler(with_mean=False)
        self.url_encoder = URLEncoder(max_length=config.cnn_max_length)
        self.cnn_model = None
        self.device = torch.device(config.device)
        logger.info(f"设备: {self.device}")
    
    def extract_structural_features(self, url: str) -> np.ndarray:
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            url_len = len(url)
            path_len = len(path)
            
            path_segments = [s for s in path.split('/') if s]
            num_path_segments = len(path_segments)
            
            num_digits = sum(c.isdigit() for c in url)
            num_hyphens = url.count('-')
            num_dots = url.count('.')
            
            digit_ratio = num_digits / max(url_len, 1)
            
            features = [
                url_len, path_len, num_path_segments,
                num_digits, num_hyphens, num_dots, digit_ratio
            ]
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(7, dtype=np.float32)
    
    def extract_text_features(self, urls: List[str], is_training: bool = True):
        if self.config.url_normalization:
            urls = [self.normalizer.normalize(u) for u in urls]
        
        url_texts = urls
        path_texts = [urlparse(u).path for u in urls]
        
        if is_training:
            # 只保留URL和Path的TF-IDF
            self.tfidf_url = TfidfVectorizer(
                max_features=self.config.max_tfidf_features,
                analyzer='char',
                ngram_range=self.config.tfidf_ngram_range,
                min_df=2,
                max_df=0.85,
                sublinear_tf=True
            )
            
            self.tfidf_path = TfidfVectorizer(
                max_features=min(5000, self.config.max_tfidf_features // 2),
                analyzer='char',
                ngram_range=(2, 3),
                min_df=2,
                max_df=0.85,
                sublinear_tf=True
            )
            
            Xurl = self.tfidf_url.fit_transform(url_texts)
            Xpath = self.tfidf_path.fit_transform(path_texts)
        else:
            Xurl = self.tfidf_url.transform(url_texts)
            Xpath = self.tfidf_path.transform(path_texts)
        
        return hstack([Xurl, Xpath], format='csr')
    
    def train_cnn(self, urls: List[str], labels: List[int], 
                  urls_val: Optional[List[str]] = None, labels_val: Optional[List[int]] = None):
        logger.info("训练简化CNN模型...")
        
        self.url_encoder.build_vocab(urls)
        encoded_urls = self.url_encoder.encode(urls)
        
        # 计算类别权重（处理不平衡）
        class_counts = Counter(labels)
        total = len(labels)
        class_weights = torch.tensor([
            total / (len(class_counts) * class_counts[i]) 
            for i in range(len(class_counts))
        ], dtype=torch.float32).to(self.device)
        
        self.cnn_model = SimplifiedTextCNN(
            vocab_size=self.url_encoder.vocab_size,
            embedding_dim=self.config.cnn_embedding_dim,
            num_filters=self.config.cnn_num_filters,
            kernel_sizes=self.config.cnn_kernel_sizes,
            num_classes=len(set(labels)),
            dropout=self.config.cnn_dropout
        ).to(self.device)
        
        train_dataset = URLDataset(encoded_urls, labels)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.cnn_batch_size,
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = None
        if urls_val is not None and labels_val is not None:
            encoded_urls_val = self.url_encoder.encode(urls_val)
            val_dataset = URLDataset(encoded_urls_val, labels_val)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.cnn_batch_size,
                shuffle=False, 
                num_workers=0
            )
        
        # 加权交叉熵
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            self.cnn_model.parameters(),
            lr=self.config.cnn_learning_rate,
            weight_decay=self.config.cnn_weight_decay  # L2正则化
        )
        
        # 学习率衰减
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )
        
        best_val_acc = 0.0
        patience = 8  # 增加耐心
        patience_counter = 0
        
        for epoch in range(self.config.cnn_epochs):
            # 训练
            self.cnn_model.train()
            train_loss = 0.0
            for batch_urls, batch_labels in train_loader:
                batch_urls = batch_urls.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.cnn_model(batch_urls)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证
            if val_loader is not None:
                self.cnn_model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_urls, batch_labels in val_loader:
                        batch_urls = batch_urls.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        outputs = self.cnn_model(batch_urls)
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_labels.size(0)
                        correct += (predicted == batch_labels).sum().item()
                
                val_acc = correct / total
                
                if epoch % 3 == 0 or epoch == self.config.cnn_epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{self.config.cnn_epochs} - "
                              f"Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"早停于 epoch {epoch+1}")
                        break
        
        logger.info(f"CNN训练完成，最佳验证准确率: {best_val_acc:.4f}")
    
    def extract_cnn_features(self, urls: List[str]) -> np.ndarray:
        if self.cnn_model is None:
            raise ValueError("CNN模型未训练")
        
        encoded_urls = self.url_encoder.encode(urls)
        dataset = URLDataset(encoded_urls)
        loader = DataLoader(
            dataset, 
            batch_size=self.config.cnn_batch_size,
            shuffle=False, 
            num_workers=0
        )
        
        features = []
        self.cnn_model.eval()
        with torch.no_grad():
            for batch_urls in loader:
                batch_urls = batch_urls.to(self.device)
                batch_features = self.cnn_model(batch_urls, return_features=True)
                features.append(batch_features.cpu().numpy())
        
        return np.vstack(features)
    
    def extract_all_features(self, urls: List[str], locations: Optional[List[str]] = None,
                            is_training: bool = True):
        # 1. 简化的结构化特征
        structural = np.vstack([self.extract_structural_features(u) for u in urls])
        
        # 2. TF-IDF特征（移除anchor）
        X_text = self.extract_text_features(urls, is_training=is_training)
        
        # 3. 位置特征
        if locations is None:
            locations = ['body'] * len(urls)
        if is_training:
            self.ohe_location.fit(np.array(locations).reshape(-1, 1))
        X_loc = self.ohe_location.transform(np.array(locations).reshape(-1, 1))
        
        # 4. CNN特征
        X_cnn = self.extract_cnn_features(urls)
        
        # 5. 合并
        dense_features = np.hstack([structural, X_cnn]).astype(np.float32)
        
        if is_training:
            self.scaler.fit(dense_features)
        dense_features = self.scaler.transform(dense_features)
        
        X_dense = csr_matrix(dense_features)
        X = hstack([X_text, X_loc, X_dense], format='csr')
        
        logger.info(f"最终特征维度: {X.shape}")
        return X


class DataProcessor:
    @staticmethod
    def remove_duplicates(urls, labels, locations):
        seen = {}
        unique_urls, unique_labels, unique_locations = [], [], []
        
        for u, l, loc in zip(urls, labels, locations):
            if u not in seen:
                seen[u] = True
                unique_urls.append(u)
                unique_labels.append(l)
                unique_locations.append(loc)
        
        logger.info(f"去重: {len(urls)} -> {len(unique_urls)}")
        return unique_urls, unique_labels, unique_locations


# ==================== 训练器 ====================

class OptimizedLightGBMCNNTrainer:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.fe = EnhancedFeatureExtractor(self.config)
        self.le = LabelEncoder()
        self.lgb_model = None
        self.processor = DataProcessor()
        logger.info("优化版训练器初始化完成")
    
    def load_data(self, json_file: str):
        filepath = Path(json_file)
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        logger.info(f"加载数据: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"原始数据: {len(data)}条")
        
        rows = []
        for it in data:
            if "url" not in it:
                continue
            url = it["url"]
            lab_raw = it.get("label") or it.get("weak_label")
            if not lab_raw or lab_raw == "skip":
                continue
            lab = map_label(lab_raw, url)
            if lab not in _ALLOWED_LABELS:
                continue
            rows.append({**it, "label": lab})
        
        logger.info(f"有效数据: {len(rows)}条")
        
        urls = [r["url"] for r in rows]
        labels = [r.get("label") for r in rows]
        locations = [r.get("location") or "body" for r in rows]
        
        label_counts = Counter(labels)
        logger.info("标签分布:")
        for lab, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {lab}: {cnt} ({cnt/len(labels)*100:.1f}%)")
        
        return urls, labels, locations
    
    def train(self, json_file: str, save_path: Optional[str] = None):
        logger.info("="*60)
        logger.info("开始训练 优化版 LightGBM + CNN")
        logger.info("="*60)
        
        # 加载数据
        urls, labels, locations = self.load_data(json_file)
        
        # 去重
        if self.config.remove_duplicates:
            urls, labels, locations = self.processor.remove_duplicates(urls, labels, locations)
        
        # 划分数据
        domains = [urlparse(u).netloc for u in urls]
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.validation_split,
            random_state=self.config.random_state
        )
        idx_tr, idx_val = next(gss.split(range(len(urls)), labels, groups=domains))
        
        def take(idxs, arr):
            return [arr[i] for i in idxs]
        
        urls_train, labels_train = take(idx_tr, urls), take(idx_tr, labels)
        locations_train = take(idx_tr, locations)
        urls_val, labels_val = take(idx_val, urls), take(idx_val, labels)
        locations_val = take(idx_val, locations)
        
        logger.info(f"训练集: {len(urls_train)}条, 验证集: {len(urls_val)}条")
        
        # 标签编码
        y_train = self.le.fit_transform(labels_train)
        y_val = self.le.transform(labels_val)
        n_classes = len(self.le.classes_)
        logger.info(f"类别数: {n_classes}, 类别: {list(self.le.classes_)}")
        
        # 训练CNN
        self.fe.train_cnn(urls_train, y_train, urls_val, y_val)
        
        # 提取特征
        logger.info("提取特征...")
        X_train = self.fe.extract_all_features(urls_train, locations_train, is_training=True)
        X_val = self.fe.extract_all_features(urls_val, locations_val, is_training=False)
        
        # 训练LightGBM（更强正则化）
        logger.info("训练LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=self.config.lgb_n_estimators,
            learning_rate=self.config.lgb_learning_rate,
            max_depth=self.config.lgb_max_depth,
            num_leaves=self.config.lgb_num_leaves,
            subsample=self.config.lgb_subsample,
            colsample_bytree=self.config.lgb_colsample_bytree,
            reg_alpha=self.config.lgb_reg_alpha,
            reg_lambda=self.config.lgb_reg_lambda,
            min_child_samples=self.config.lgb_min_child_samples,
            objective="multiclass",
            class_weight="balanced",
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)]  # 更早停止
        )
        
        # 评估
        y_train_pred = self.lgb_model.predict(X_train)
        y_val_pred = self.lgb_model.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
        
        logger.info(f"\n训练集: 准确率={train_acc:.4f}, F1={train_f1:.4f}")
        logger.info(f"验证集: 准确率={val_acc:.4f}, F1={val_f1:.4f}")
        logger.info(f"过拟合检测: {train_acc - val_acc:.4f} (应<0.15)")
        
        print("\n验证集分类报告:")
        print(classification_report(
            y_val, y_val_pred,
            target_names=list(self.le.classes_),
            zero_division=0,
            digits=4
        ))
        
        if save_path:
            self.save_model(save_path)
        
        logger.info("="*60)
        logger.info("训练完成！")
        logger.info("="*60)
        
        return {
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'overfitting': train_acc - val_acc,
            'n_samples': len(urls_train),
            'n_classes': n_classes,
            'classes': list(self.le.classes_)
        }
    
    def save_model(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        pkg = {
            'lgb_model': self.lgb_model,
            'feature_extractor': self.fe,
            'label_encoder': self.le,
            'config': self.config.to_dict()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"模型已保存: {filepath} ({file_size:.2f} MB)")
    
    @staticmethod
    def load_model(filepath: str):
        with open(filepath, 'rb') as f:
            pkg = pickle.load(f)
        
        trainer = OptimizedLightGBMCNNTrainer(ModelConfig(**pkg['config']))
        trainer.lgb_model = pkg['lgb_model']
        trainer.fe = pkg['feature_extractor']
        trainer.le = pkg['label_encoder']
        
        logger.info("模型加载完成")
        return trainer
    
    def predict(self, urls: List[str], locations: Optional[List[str]] = None):
        X = self.fe.extract_all_features(urls, locations, is_training=False)
        y_pred = self.lgb_model.predict(X)
        return list(self.le.inverse_transform(y_pred))
    
    def predict_proba(self, urls: List[str], locations: Optional[List[str]] = None):
        X = self.fe.extract_all_features(urls, locations, is_training=False)
        return self.lgb_model.predict_proba(X)


def main():
    config = ModelConfig()
    trainer = OptimizedLightGBMCNNTrainer(config)
    
    results = trainer.train(
        json_file="training_data/labeled_urls.json",
        save_path="optimized_lgb_cnn_model.pkl"
    )
    
    print("\n优化效果:")
    print(f"验证准确率: {results['val_accuracy']:.4f}")
    print(f"验证F1: {results['val_f1']:.4f}")
    print(f"过拟合程度: {results['overfitting']:.4f}")


if __name__ == "__main__":
    main()