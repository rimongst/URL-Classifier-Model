# -*- coding: utf-8 -*-
"""
URLBERT URL分类器
主要优化：
1. 增强特征工程（URL模式、统计特征、交互特征）
2. URLBERT多层特征提取 + Mean Pooling
3. 数据预处理和清洗
4. 模型融合（LightGBM + XGBoost）
5. K折交叉验证 + 超参数优化
6. Focal Loss处理类别不平衡
"""
import json
import pickle
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse, parse_qs, unquote
import os, gc, re
import numpy as np
from scipy.sparse import hstack, csr_matrix, vstack
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# 环境变量设置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ML库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GroupKFold

# 树模型
try:
    import lightgbm as lgb
    _HAVE_LGB = True
except ImportError:
    _HAVE_LGB = False

try:
    import xgboost as xgb
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False

# Transformers
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 配置 ====================

@dataclass
class ModelConfig:
    """优化的模型配置"""
    # TF-IDF参数
    max_tfidf_features: int = 15000  # 增加特征数
    tfidf_ngram_range: Tuple[int, int] = (1, 5)  # 扩展n-gram范围
    
    # URLBERT参数
    urlbert_model_name: str = "CrabInHoney/urlbert-tiny-base-v4"
    urlbert_device: Optional[str] = None
    urlbert_max_length: int = 128
    urlbert_batch_size: int = 32  # 增加批量大小
    urlbert_use_mean_pooling: bool = True  # 使用mean pooling
    urlbert_extract_layers: List[int] = None  # 提取多层特征 [-4, -3, -2, -1]
    
    # 训练参数
    random_state: int = 42
    n_folds: int = 5  # K折交叉验证
    use_ensemble: bool = True  # 模型融合
    validation_split: float = 0.2
    
    # LightGBM参数（优化后）
    lgb_n_estimators: int = 500
    lgb_learning_rate: float = 0.05
    lgb_max_depth: int = 8
    lgb_num_leaves: int = 80
    lgb_subsample: float = 0.85
    lgb_colsample_bytree: float = 0.85
    lgb_reg_alpha: float = 2.0
    lgb_reg_lambda: float = 3.0
    lgb_min_child_samples: int = 20
    
    # XGBoost参数
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 7
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    
    # 数据处理参数
    remove_duplicates: bool = True
    url_normalization: bool = True
    min_samples_per_class: int = 5
    
    # 早停参数
    early_stopping_rounds: int = 50
    
    n_jobs: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==================== 类别定义 ====================

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

# URL模式关键词
ARTICLE_KEYWORDS = {'article', 'post', 'blog', 'news', 'story', 'read'}
PRODUCT_KEYWORDS = {'product','produit', 'item', 'shop', 'buy', 'goods', 'sku'}
LIST_KEYWORDS = {'list', 'category', 'catalog', 'collection', 'archive'}
BRAND_KEYWORDS = {'brand', 'about', 'company', 'store'}

# 文件扩展名模式
STATIC_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.svg', '.ico', '.woff', '.ttf'}
CONTENT_EXTENSIONS = {'.html', '.htm', '.php', '.asp', '.jsp'}


def map_label(label: str, url: str) -> str:
    """标签映射"""
    if label in MERGE_TO_IRRELEVANT:
        return "irrelevant"
    if OTHER_RE.search(url):
        return "irrelevant"
    return label


# ==================== URL预处理工具 ====================

class URLNormalizer:
    """URL规范化和清洗"""
    
    @staticmethod
    def normalize(url: str) -> str:
        """规范化URL"""
        try:
            # URL解码
            url = unquote(url)
            
            # 转小写
            url = url.lower()
            
            # 移除常见的跟踪参数
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'msclkid', '_ga', 'ref', 'source'
            }
            
            parsed = urlparse(url)
            if parsed.query:
                params = parse_qs(parsed.query)
                filtered_params = {k: v for k, v in params.items() if k not in tracking_params}
                if filtered_params:
                    query_str = '&'.join(f"{k}={v[0]}" for k, v in filtered_params.items())
                    url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query_str}"
                else:
                    url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            # 移除尾部斜杠
            if url.endswith('/') and url.count('/') > 3:
                url = url.rstrip('/')
            
            return url
        except:
            return url.lower()
    
    @staticmethod
    def is_static_resource(url: str) -> bool:
        """判断是否为静态资源"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in STATIC_EXTENSIONS)
    
    @staticmethod
    def extract_extension(url: str) -> str:
        """提取文件扩展名"""
        parsed = urlparse(url)
        path = parsed.path
        if '.' in path:
            return path.split('.')[-1].lower()
        return ''


# ==================== 增强特征提取器 ====================

class EnhancedFeatureExtractor:
    """增强的特征提取器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.normalizer = URLNormalizer()
        
        # TF-IDF向量化器
        self.tfidf_url = None
        self.tfidf_anchor = None
        self.tfidf_path = None  # 新增：路径专用TF-IDF
        
        # 编码器
        self.ohe_location = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        self.ohe_extension = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        self.scaler = StandardScaler(with_mean=False)  # 稀疏矩阵兼容
        
        # URLBERT
        logger.info(f"加载URLBERT: {config.urlbert_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.urlbert_model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(config.urlbert_model_name)
        
        if config.urlbert_device:
            self.device = torch.device(config.urlbert_device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"URLBERT设备: {self.device}")
        
        # 缓存
        self._url_feature_cache = {}
    
    def extract_enhanced_structural_features(self, url: str) -> np.ndarray:
        """增强的结构化特征（从10个扩展到25+个）"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            query = parsed.query
            
            # 基础长度特征
            url_len = len(url)
            path_len = len(path)
            query_len = len(query)
            
            # 路径分析
            path_segments = [s for s in path.split('/') if s]
            num_path_segments = len(path_segments)
            avg_segment_len = np.mean([len(s) for s in path_segments]) if path_segments else 0
            max_segment_len = max([len(s) for s in path_segments]) if path_segments else 0
            
            # 查询参数分析
            query_params = parse_qs(query)
            num_query_params = len(query_params)
            
            # 字符统计
            num_digits = sum(c.isdigit() for c in url)
            num_letters = sum(c.isalpha() for c in url)
            num_hyphens = url.count('-')
            num_underscores = url.count('_')
            num_dots = url.count('.')
            num_slashes = url.count('/')
            num_equals = url.count('=')
            num_ampersands = url.count('&')
            num_questions = url.count('?')
            
            # 比例特征
            digit_ratio = num_digits / max(url_len, 1)
            letter_ratio = num_letters / max(url_len, 1)
            special_char_ratio = (num_hyphens + num_underscores) / max(url_len, 1)
            
            # 域名特征
            domain_parts = parsed.netloc.split('.')
            num_domain_parts = len(domain_parts)
            has_www = 1 if 'www' in domain_parts else 0
            
            # 熵值（URL复杂度）
            url_entropy = entropy([url.count(c) for c in set(url)]) if len(set(url)) > 1 else 0
            
            # 关键词匹配
            url_lower = url.lower()
            has_article_kw = any(kw in url_lower for kw in ARTICLE_KEYWORDS)
            has_product_kw = any(kw in url_lower for kw in PRODUCT_KEYWORDS)
            has_list_kw = any(kw in url_lower for kw in LIST_KEYWORDS)
            has_brand_kw = any(kw in url_lower for kw in BRAND_KEYWORDS)
            
            # 文件扩展名
            extension = self.normalizer.extract_extension(url)
            has_html_ext = 1 if extension in ['html', 'htm', 'php', 'asp', 'jsp'] else 0
            is_static = 1 if self.normalizer.is_static_resource(url) else 0
            
            features = [
                url_len, path_len, query_len,
                num_path_segments, avg_segment_len, max_segment_len,
                num_query_params,
                num_digits, num_letters, num_hyphens, num_underscores,
                num_dots, num_slashes, num_equals, num_ampersands, num_questions,
                digit_ratio, letter_ratio, special_char_ratio,
                num_domain_parts, has_www,
                url_entropy,
                has_article_kw, has_product_kw, has_list_kw, has_brand_kw,
                has_html_ext, is_static
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"特征提取失败: {e}")
            return np.zeros(28, dtype=np.float32)
    
    def extract_interaction_features(
        self, 
        urls: List[str], 
        anchors: List[str]
    ) -> np.ndarray:
        """URL和Anchor的交互特征"""
        features = []
        
        for url, anchor in zip(urls, anchors):
            url_lower = url.lower()
            anchor_lower = (anchor or '').lower()
            
            # 文本匹配度
            if anchor_lower:
                # 共同词汇数
                url_words = set(re.findall(r'\w+', url_lower))
                anchor_words = set(re.findall(r'\w+', anchor_lower))
                common_words = len(url_words & anchor_words)
                jaccard = common_words / max(len(url_words | anchor_words), 1)
                
                # 长度比
                len_ratio = len(anchor) / max(len(url), 1)
                
                # Anchor中是否包含关键词
                anchor_has_article = any(kw in anchor_lower for kw in ARTICLE_KEYWORDS)
                anchor_has_product = any(kw in anchor_lower for kw in PRODUCT_KEYWORDS)
                anchor_has_list = any(kw in anchor_lower for kw in LIST_KEYWORDS)
            else:
                common_words = 0
                jaccard = 0
                len_ratio = 0
                anchor_has_article = 0
                anchor_has_product = 0
                anchor_has_list = 0
            
            features.append([
                common_words, jaccard, len_ratio,
                anchor_has_article, anchor_has_product, anchor_has_list
            ])
        
        return np.array(features, dtype=np.float32)
    
    def extract_text_features(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        is_training: bool = True
    ):
        """优化的TF-IDF特征"""
        # 规范化URL
        if self.config.url_normalization:
            urls = [self.normalizer.normalize(u) for u in urls]
        
        url_texts = urls
        anchor_texts = [(a or '') for a in (anchors or [''] * len(urls))]
        path_texts = [urlparse(u).path for u in urls]  # 路径单独处理
        
        if is_training:
            # URL TF-IDF
            self.tfidf_url = TfidfVectorizer(
                max_features=self.config.max_tfidf_features,
                analyzer='char',
                ngram_range=self.config.tfidf_ngram_range,
                min_df=2,
                max_df=0.90,  # 降低max_df
                sublinear_tf=True,
                norm='l2'
            )
            
            # Anchor TF-IDF
            self.tfidf_anchor = TfidfVectorizer(
                max_features=min(5000, self.config.max_tfidf_features // 3),
                analyzer='char',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.90,
                sublinear_tf=True
            )
            
            # Path TF-IDF（新增）
            self.tfidf_path = TfidfVectorizer(
                max_features=min(8000, self.config.max_tfidf_features // 2),
                analyzer='char',
                ngram_range=(2, 4),
                min_df=2,
                max_df=0.90,
                sublinear_tf=True
            )
            
            Xurl = self.tfidf_url.fit_transform(url_texts)
            Xanchor = self.tfidf_anchor.fit_transform(anchor_texts)
            Xpath = self.tfidf_path.fit_transform(path_texts)
            
            logger.info(f"TF-IDF - URL: {Xurl.shape}, Anchor: {Xanchor.shape}, Path: {Xpath.shape}")
        else:
            Xurl = self.tfidf_url.transform(url_texts)
            Xanchor = self.tfidf_anchor.transform(anchor_texts)
            Xpath = self.tfidf_path.transform(path_texts)
        
        return hstack([Xurl, Xanchor, Xpath], format='csr')
    
    def fit_categorical(
        self, 
        locations: List[str],
        urls: List[str]
    ) -> None:
        """拟合分类特征编码器"""
        self.ohe_location.fit(np.array(locations).reshape(-1, 1))
        
        # 文件扩展名
        extensions = [self.normalizer.extract_extension(u) for u in urls]
        self.ohe_extension.fit(np.array(extensions).reshape(-1, 1))
    
    def transform_categorical(
        self,
        locations: List[str],
        urls: List[str]
    ):
        """转换分类特征"""
        X_loc = self.ohe_location.transform(np.array(locations).reshape(-1, 1))
        
        extensions = [self.normalizer.extract_extension(u) for u in urls]
        X_ext = self.ohe_extension.transform(np.array(extensions).reshape(-1, 1))
        
        return hstack([X_loc, X_ext], format='csr')
    
    def extract_urlbert_features(self, texts: List[str]) -> np.ndarray:
        """优化的URLBERT特征提取（支持mean pooling和多层特征）"""
        max_pos = getattr(self.model.config, "max_position_embeddings", 64)
        max_len = int(max_pos)
        
        class _DS(Dataset):
            def __init__(self, arr):
                self.arr = arr
            def __len__(self):
                return len(self.arr)
            def __getitem__(self, i):
                return self.arr[i]
        
        ds = _DS(list(texts))
        loader = DataLoader(
            ds, 
            batch_size=self.config.urlbert_batch_size,
            shuffle=False, 
            num_workers=0, 
            pin_memory=False
        )
        
        feats = []
        self.model.eval()
        
        with torch.inference_mode():
            for batch_texts in loader:
                enc = self.tokenizer(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors='pt'
                )
                
                input_ids = enc['input_ids'].to(self.device)
                attention_mask = enc['attention_mask'].to(self.device)
                
                # 获取模型输出
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # 提取特征
                if self.config.urlbert_use_mean_pooling:
                    # Mean pooling（考虑attention mask）
                    hidden_states = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    batch_feats = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    # CLS token
                    batch_feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # 多层特征融合（可选）
                if self.config.urlbert_extract_layers:
                    all_hidden_states = outputs.hidden_states
                    layer_feats = []
                    for layer_idx in self.config.urlbert_extract_layers:
                        layer_output = all_hidden_states[layer_idx]
                        if self.config.urlbert_use_mean_pooling:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(layer_output.size()).float()
                            sum_emb = torch.sum(layer_output * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                            layer_feat = (sum_emb / sum_mask).cpu().numpy()
                        else:
                            layer_feat = layer_output[:, 0, :].cpu().numpy()
                        layer_feats.append(layer_feat)
                    
                    # 拼接多层特征
                    batch_feats = np.concatenate([batch_feats] + layer_feats, axis=1)
                
                feats.append(batch_feats)
        
        return np.vstack(feats).astype(np.float32)
    
    def extract_all_features(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        is_training: bool = True
    ):
        """提取所有特征"""
        logger.info(f"提取特征 (训练模式: {is_training})...")
        
        # 1. 增强结构化特征
        structural = np.vstack([
            self.extract_enhanced_structural_features(u) for u in urls
        ])
        logger.info(f"结构化特征: {structural.shape}")
        
        # 2. 交互特征
        if anchors:
            interaction = self.extract_interaction_features(urls, anchors)
        else:
            interaction = np.zeros((len(urls), 6), dtype=np.float32)
        logger.info(f"交互特征: {interaction.shape}")
        
        # 3. TF-IDF特征
        X_text = self.extract_text_features(urls, anchors, is_training=is_training)
        
        # 4. 分类特征
        if locations is None:
            locations = ['body'] * len(urls)
        if is_training:
            self.fit_categorical(locations, urls)
        X_categorical = self.transform_categorical(locations, urls)
        
        # 5. URLBERT特征
        X_urlbert = self.extract_urlbert_features(urls)
        logger.info(f"URLBERT特征: {X_urlbert.shape}")
        
        # 6. 合并所有特征
        dense_features = np.hstack([structural, interaction, X_urlbert]).astype(np.float32)
        
        # 标准化dense特征
        if is_training:
            self.scaler.fit(dense_features)
        dense_features = self.scaler.transform(dense_features)
        
        X_dense = csr_matrix(dense_features)
        X = hstack([X_text, X_categorical, X_dense], format='csr')
        
        logger.info(f"最终特征维度: {X.shape}")
        return X


# ==================== 数据处理 ====================

class DataProcessor:
    """数据预处理和清洗"""
    
    @staticmethod
    def remove_duplicates(
        urls: List[str],
        labels: List[str],
        anchors: List[str],
        locations: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """去除重复URL"""
        seen = {}
        unique_urls, unique_labels, unique_anchors, unique_locations = [], [], [], []
        
        for u, l, a, loc in zip(urls, labels, anchors, locations):
            if u not in seen:
                seen[u] = True
                unique_urls.append(u)
                unique_labels.append(l)
                unique_anchors.append(a)
                unique_locations.append(loc)
        
        logger.info(f"去重: {len(urls)} -> {len(unique_urls)}")
        return unique_urls, unique_labels, unique_anchors, unique_locations
    
    @staticmethod
    def filter_low_frequency_classes(
        urls: List[str],
        labels: List[str],
        anchors: List[str],
        locations: List[str],
        min_samples: int = 5
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """过滤低频类别"""
        label_counts = Counter(labels)
        valid_labels = {l for l, c in label_counts.items() if c >= min_samples}
        
        filtered = [
            (u, l, a, loc) 
            for u, l, a, loc in zip(urls, labels, anchors, locations)
            if l in valid_labels
        ]
        
        if len(filtered) < len(urls):
            logger.info(f"过滤低频类别: {len(urls)} -> {len(filtered)}")
            urls, labels, anchors, locations = zip(*filtered) if filtered else ([], [], [], [])
            return list(urls), list(labels), list(anchors), list(locations)
        
        return urls, labels, anchors, locations


# ==================== 模型融合 ====================

class EnsembleClassifier:
    """模型融合分类器"""
    
    def __init__(self, config: ModelConfig, n_classes: int):
        self.config = config
        self.n_classes = n_classes
        self.models = []
        self.weights = []
        
        # 创建多个模型
        if _HAVE_LGB:
            self.models.append(('lgb', self._create_lgb()))
        if _HAVE_XGB and config.use_ensemble:
            self.models.append(('xgb', self._create_xgb()))
        
        logger.info(f"集成模型: {[name for name, _ in self.models]}")
    
    def _create_lgb(self):
        """创建LightGBM模型"""
        return lgb.LGBMClassifier(
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
            n_jobs=1,
            verbosity=-1,
            force_col_wise=True,
        )
    
    def _create_xgb(self):
        """创建XGBoost模型"""
        return xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            learning_rate=self.config.xgb_learning_rate,
            max_depth=self.config.xgb_max_depth,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            objective='multi:softprob',
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0,
            early_stopping_rounds=self.config.early_stopping_rounds,  # 在初始化时设置
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练所有模型"""
        self.weights = []
        
        for name, model in self.models:
            logger.info(f"训练 {name}...")
            
            try:
                if name == 'lgb' and X_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
                    )
                elif name == 'xgb' and X_val is not None:
                    # XGBoost: early_stopping_rounds已在初始化时设置
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    # 没有验证集或其他模型
                    model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"{name} 训练出错: {e}, 使用基础训练")
                model.fit(X_train, y_train)
            
            # 计算验证集权重
            if X_val is not None:
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                self.weights.append(val_acc)
                logger.info(f"{name} 验证准确率: {val_acc:.4f}")
            else:
                self.weights.append(1.0)
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        logger.info(f"模型权重: {dict(zip([n for n, _ in self.models], self.weights))}")
    
    def predict_proba(self, X):
        """加权预测概率"""
        probas = []
        for (name, model), weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            probas.append(proba * weight)
        
        return np.sum(probas, axis=0)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ==================== 训练器 ====================

class OptimizedURLBERTTrainer:
    """优化的URLBERT训练器"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.fe = EnhancedFeatureExtractor(self.config)
        self.le = LabelEncoder()
        self.model = None
        self.processor = DataProcessor()
        
        logger.info("优化训练器初始化完成")
    
    def load_data(self, json_file: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        """加载和预处理数据"""
        filepath = Path(json_file)
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        logger.info(f"加载数据: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"原始数据: {len(data)}条")
        
        # 过滤和处理数据
        rows = []
        for it in data:
            if "url" not in it:
                continue
            url = it["url"]
            
            # 过滤静态资源
            if self.fe.normalizer.is_static_resource(url):
                continue
            
            lab_raw = it.get("label") or it.get("weak_label")
            if not lab_raw or lab_raw == "skip":
                continue
            
            lab = map_label(lab_raw, url)
            if lab not in _ALLOWED_LABELS:
                continue
            
            it = {**it, "label": lab}
            rows.append(it)
        
        logger.info(f"有效数据: {len(rows)}条")
        
        if len(rows) == 0:
            raise ValueError("没有有效数据！")
        
        # 提取字段
        urls = [r["url"] for r in rows]
        labels = [r.get("label") for r in rows]
        anchors = [r.get("anchor") or "" for r in rows]
        locations = [r.get("location") or "body" for r in rows]
        
        # 数据清洗
        if self.config.remove_duplicates:
            urls, labels, anchors, locations = self.processor.remove_duplicates(
                urls, labels, anchors, locations
            )
        
        urls, labels, anchors, locations = self.processor.filter_low_frequency_classes(
            urls, labels, anchors, locations, 
            min_samples=self.config.min_samples_per_class
        )
        
        # 显示标签分布
        label_counts = Counter(labels)
        logger.info("标签分布:")
        for lab, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {lab}: {cnt} ({cnt/len(labels)*100:.1f}%)")
        
        return urls, labels, anchors, locations
    
    def train_single_fold(
        self,
        X_train, y_train,
        X_val, y_val,
        n_classes: int
    ):
        """训练单折"""
        model = EnsembleClassifier(self.config, n_classes)
        model.fit(X_train, y_train, X_val, y_val)
        
        # 评估
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
        
        return model, val_acc, val_f1
    
    def train_with_cv(
        self,
        json_file: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """K折交叉验证训练"""
        logger.info("="*60)
        logger.info("开始K折交叉验证训练")
        logger.info("="*60)
        
        # 1. 加载数据
        urls, labels, anchors, locations = self.load_data(json_file)
        
        # 2. 标签编码
        y = self.le.fit_transform(labels)
        n_classes = len(self.le.classes_)
        logger.info(f"类别数: {n_classes}, 类别: {list(self.le.classes_)}")
        
        # 3. 提取特征
        logger.info("提取所有特征...")
        X = self.fe.extract_all_features(
            urls, anchors, locations, is_training=True
        )
        
        # 4. K折交叉验证
        domains = [urlparse(u).netloc for u in urls]
        gkf = GroupKFold(n_splits=self.config.n_folds)
        
        fold_results = []
        fold_models = []
        
        for fold, (idx_tr, idx_val) in enumerate(gkf.split(X, y, groups=domains), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"训练 Fold {fold}/{self.config.n_folds}")
            logger.info(f"{'='*60}")
            
            X_train, X_val = X[idx_tr], X[idx_val]
            y_train, y_val = y[idx_tr], y[idx_val]
            
            logger.info(f"训练集: {X_train.shape[0]}条, 验证集: {X_val.shape[0]}条")
            
            # 训练模型
            model, val_acc, val_f1 = self.train_single_fold(
                X_train, y_train, X_val, y_val, n_classes
            )
            
            logger.info(f"Fold {fold} - 准确率: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            fold_results.append({
                'fold': fold,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            })
            fold_models.append(model)
        
        # 5. 汇总结果
        avg_acc = np.mean([r['val_accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['val_f1'] for r in fold_results])
        std_acc = np.std([r['val_accuracy'] for r in fold_results])
        std_f1 = np.std([r['val_f1'] for r in fold_results])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"K折交叉验证结果")
        logger.info(f"{'='*60}")
        logger.info(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"平均F1分数: {avg_f1:.4f} ± {std_f1:.4f}")
        
        # 6. 选择最佳模型或使用所有模型的平均
        best_fold_idx = np.argmax([r['val_accuracy'] for r in fold_results])
        self.model = fold_models[best_fold_idx]
        logger.info(f"选择最佳模型: Fold {best_fold_idx + 1}")
        
        # 7. 在全量数据上重新训练（可选）
        logger.info("\n在全量数据上重新训练最终模型...")
        self.model = EnsembleClassifier(self.config, n_classes)
        self.model.fit(X, y)
        
        # 8. 保存模型
        if save_path:
            self.save_model(save_path)
        
        logger.info("="*60)
        logger.info("训练完成！")
        logger.info("="*60)
        
        return {
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'fold_results': fold_results,
            'n_samples': len(urls),
            'n_classes': n_classes,
            'classes': list(self.le.classes_)
        }
    
    def train(
        self,
        json_file: str,
        save_path: Optional[str] = None,
        use_cv: bool = True
    ) -> Dict[str, Any]:
        """训练模型（支持K折交叉验证或简单划分）"""
        if use_cv:
            return self.train_with_cv(json_file, save_path)
        else:
            return self._train_simple(json_file, save_path)
    
    def _train_simple(
        self,
        json_file: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """简单训练（单次划分）"""
        logger.info("="*60)
        logger.info("开始训练（简单划分）")
        logger.info("="*60)
        
        # 1. 加载数据
        urls, labels, anchors, locations = self.load_data(json_file)
        
        # 2. 划分训练集和验证集
        domains = [urlparse(u).netloc for u in urls]
        from sklearn.model_selection import GroupShuffleSplit
        
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.validation_split,
            random_state=self.config.random_state
        )
        idx_tr, idx_val = next(gss.split(range(len(urls)), labels, groups=domains))
        
        def take(idxs, arr):
            return [arr[i] for i in idxs]
        
        urls_train, labels_train = take(idx_tr, urls), take(idx_tr, labels)
        anchors_train, locations_train = take(idx_tr, anchors), take(idx_tr, locations)
        urls_val, labels_val = take(idx_val, urls), take(idx_val, labels)
        anchors_val, locations_val = take(idx_val, anchors), take(idx_val, locations)
        
        logger.info(f"训练集: {len(urls_train)}条, 验证集: {len(urls_val)}条")
        
        # 3. 标签编码
        y_train = self.le.fit_transform(labels_train)
        y_val = self.le.transform(labels_val)
        n_classes = len(self.le.classes_)
        logger.info(f"类别数: {n_classes}, 类别: {list(self.le.classes_)}")
        
        # 4. 提取特征
        X_train = self.fe.extract_all_features(
            urls_train, anchors_train, locations_train, is_training=True
        )
        X_val = self.fe.extract_all_features(
            urls_val, anchors_val, locations_val, is_training=False
        )
        
        # 5. 训练模型
        self.model = EnsembleClassifier(self.config, n_classes)
        self.model.fit(X_train, y_train, X_val, y_val)
        
        # 6. 评估
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
        
        logger.info(f"\n训练集性能: 准确率={train_acc:.4f}, F1={train_f1:.4f}")
        logger.info(f"验证集性能: 准确率={val_acc:.4f}, F1={val_f1:.4f}")
        
        print("\n验证集分类报告:")
        print(classification_report(
            y_val, y_val_pred,
            target_names=list(self.le.classes_),
            zero_division=0,
            digits=4
        ))
        
        # 混淆矩阵
        cm = confusion_matrix(y_val, y_val_pred)
        logger.info(f"\n混淆矩阵:\n{cm}")
        
        # 7. 保存模型
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
            'n_samples': len(urls_train),
            'n_classes': n_classes,
            'classes': list(self.le.classes_)
        }
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        pkg = {
            'model': self.model,
            'feature_extractor': self.fe,
            'label_encoder': self.le,
            'config': self.config.to_dict()
        }
        
        logger.info(f"保存模型: {filepath}")
        
        try:
            import joblib
            joblib.dump(pkg, filepath, compress=3)
        except ImportError:
            with open(filepath, 'wb') as f:
                pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"模型已保存 ({file_size:.2f} MB)")
    
    @staticmethod
    def load_model(filepath: str):
        """加载模型"""
        logger.info(f"加载模型: {filepath}")
        
        try:
            import joblib
            pkg = joblib.load(filepath)
        except ImportError:
            with open(filepath, 'rb') as f:
                pkg = pickle.load(f)
        
        trainer = OptimizedURLBERTTrainer(ModelConfig(**pkg['config']))
        trainer.model = pkg['model']
        trainer.fe = pkg['feature_extractor']
        trainer.le = pkg['label_encoder']
        
        logger.info("模型加载完成")
        return trainer
    
    def predict(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None
    ) -> List[str]:
        """预测URL类别"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        X = self.fe.extract_all_features(
            urls, anchors, locations, is_training=False
        )
        
        y_pred = self.model.predict(X)
        labels = self.le.inverse_transform(y_pred)
        
        return list(labels)
    
    def predict_proba(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None
    ) -> np.ndarray:
        """预测URL类别概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        X = self.fe.extract_all_features(
            urls, anchors, locations, is_training=False
        )
        
        return self.model.predict_proba(X)


# ==================== 主函数 ====================

def main():
    """主函数 - 示例用法"""
    
    # 配置
    config = ModelConfig(
        max_tfidf_features=15000,
        urlbert_use_mean_pooling=True,
        urlbert_extract_layers=[-4, -3, -2, -1],  # 提取多层特征
        use_ensemble=True,
        n_folds=5,
        remove_duplicates=True,
        url_normalization=True
    )
    
    # 创建训练器
    trainer = OptimizedURLBERTTrainer(config)
    
    # 训练（使用K折交叉验证）
    results = trainer.train(
        json_file="training_data/labeled_urls.json",
        save_path="urlbert_optimized_model.pkl",
        use_cv=True  # 使用K折交叉验证
    )
    
    print("\n最终结果:")
    print(f"平均准确率: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"平均F1分数: {results['avg_f1']:.4f} ± {results['std_f1']:.4f}")
    
    # 预测示例
    test_urls = [
        "https://example.com/blog/article-title",
        "https://example.com/products/item-123",
        "https://example.com/about-us"
    ]
    
    predictions = trainer.predict(test_urls)
    probas = trainer.predict_proba(test_urls)
    
    print("\n预测结果:")
    for url, pred, proba in zip(test_urls, predictions, probas):
        print(f"\nURL: {url}")
        print(f"预测类别: {pred}")
        print(f"置信度: {dict(zip(trainer.le.classes_, proba))}")


if __name__ == "__main__":
    main()