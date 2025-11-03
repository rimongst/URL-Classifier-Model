# -*- coding: utf-8 -*-
"""
- Enhanced feature engineering with URL patterns and statistical features
- URLBERT multi-layer feature extraction with mean pooling
- Ensemble model (LightGBM + XGBoost) with K-fold cross-validation
- Improved performance and cleaner code structure
"""

import json
import pickle
import logging
import re
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse, parse_qs, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.sparse import hstack, csr_matrix
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Environment settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold

# Tree models
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

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class ModelConfig:
    """Model configuration with optimized defaults"""
    # TF-IDF parameters
    max_tfidf_features: int = 10000  # Reduced from 15000
    tfidf_ngram_range: Tuple[int, int] = (1, 4)  # Reduced from (1, 5)
    
    # URLBERT parameters
    urlbert_model_name: str = "CrabInHoney/urlbert-tiny-base-v4"
    urlbert_device: Optional[str] = None
    urlbert_max_length: int = 128
    urlbert_batch_size: int = 64  # Increased for better performance
    urlbert_use_mean_pooling: bool = True
    urlbert_extract_layers: Optional[List[int]] = None  # e.g., [-4, -3, -2, -1]
    
    # Training parameters
    random_state: int = 42
    n_folds: int = 5
    use_ensemble: bool = True
    
    # LightGBM parameters
    lgb_n_estimators: int = 400  # Reduced from 500
    lgb_learning_rate: float = 0.05
    lgb_max_depth: int = 7  # Reduced from 8
    lgb_num_leaves: int = 60  # Reduced from 80
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 1.5  # Reduced from 2.0
    lgb_reg_lambda: float = 2.0  # Reduced from 3.0
    
    # XGBoost parameters
    xgb_n_estimators: int = 400
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # Data processing
    remove_duplicates: bool = True
    url_normalization: bool = True
    min_samples_per_class: int = 5
    early_stopping_rounds: int = 50
    
    n_jobs: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==================== Constants ====================

CATEGORIES = {
    'article_page': 'Article Page',
    'article_list': 'Article List',
    'product_detail': 'Product Detail',
    'product_list': 'Product List',
    'brand_info': 'Brand Info',
    'irrelevant': 'Irrelevant'
}

_ALLOWED_LABELS = frozenset(CATEGORIES.keys())
MERGE_TO_IRRELEVANT = {"irrelevant", "legal", "account", "commerce"}
IRRELEVANT_PATTERN = re.compile(
    r"/(privacy|terms|legal|mentions[-_]?legales|rgpd|gdpr|"
    r"login|signin|signup|register|account|my[-_]?account|"
    r"cart|checkout|panier|paiement|basket)\b",
    re.I
)

# URL pattern keywords
ARTICLE_KEYWORDS = {'article', 'post', 'blog', 'news', 'story', 'read'}
PRODUCT_KEYWORDS = {'product', 'produit', 'item', 'shop', 'buy', 'goods', 'sku'}
LIST_KEYWORDS = {'list', 'category', 'catalog', 'collection', 'archive'}
BRAND_KEYWORDS = {'brand', 'about', 'company', 'store'}

# File extensions
STATIC_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.svg', '.ico', '.woff', '.ttf'}
CONTENT_EXTENSIONS = {'.html', '.htm', '.php', '.asp', '.jsp'}


def map_label(label: str, url: str) -> str:
    """Map label to standard categories"""
    if label in MERGE_TO_IRRELEVANT or IRRELEVANT_PATTERN.search(url):
        return "irrelevant"
    return label


# ==================== URL Utilities ====================

class URLNormalizer:
    """URL normalization and validation utilities"""
    
    TRACKING_PARAMS = {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'fbclid', 'gclid', 'msclkid', '_ga', 'ref', 'source'
    }
    
    @staticmethod
    def normalize(url: str) -> str:
        """Normalize URL by removing tracking params and standardizing format"""
        try:
            url = unquote(url).lower()
            parsed = urlparse(url)
            
            if parsed.query:
                params = parse_qs(parsed.query)
                filtered = {k: v for k, v in params.items() if k not in URLNormalizer.TRACKING_PARAMS}
                query_str = '&'.join(f"{k}={v[0]}" for k, v in filtered.items()) if filtered else ''
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if query_str:
                    url += f"?{query_str}"
            
            # Remove trailing slash (except for root)
            if url.endswith('/') and url.count('/') > 3:
                url = url.rstrip('/')
            
            return url
        except:
            return url.lower()
    
    @staticmethod
    def is_static_resource(url: str) -> bool:
        """Check if URL is a static resource"""
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in STATIC_EXTENSIONS)
    
    @staticmethod
    def extract_extension(url: str) -> str:
        """Extract file extension from URL"""
        path = urlparse(url).path
        return path.split('.')[-1].lower() if '.' in path else ''


# ==================== Feature Extraction ====================

class FeatureExtractor:
    """Enhanced feature extraction with multiple feature types"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.normalizer = URLNormalizer()
        
        # Vectorizers
        self.tfidf_url = None
        self.tfidf_anchor = None
        self.ohe_location = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        self.ohe_extension = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        self.scaler = StandardScaler(with_mean=False)
        
        # URLBERT
        logger.info(f"Loading URLBERT: {config.urlbert_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.urlbert_model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(config.urlbert_model_name)
        
        self.device = torch.device(
            config.urlbert_device if config.urlbert_device 
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"URLBERT device: {self.device}")
    
    def extract_structural_features(self, url: str) -> np.ndarray:
        """Extract structural features from URL (vectorized for batch processing)"""
        try:
            parsed = urlparse(url)
            path, query = parsed.path, parsed.query
            
            # Length features
            url_len, path_len, query_len = len(url), len(path), len(query)
            
            # Path analysis
            segments = [s for s in path.split('/') if s]
            num_segments = len(segments)
            avg_seg_len = np.mean([len(s) for s in segments]) if segments else 0
            max_seg_len = max([len(s) for s in segments]) if segments else 0
            
            # Query parameters
            num_params = len(parse_qs(query))
            
            # Character counts (vectorized)
            char_counts = [
                sum(c.isdigit() for c in url),
                sum(c.isalpha() for c in url),
                url.count('-'), url.count('_'), url.count('.'),
                url.count('/'), url.count('='), url.count('&')
            ]
            
            # Ratios
            digit_ratio = char_counts[0] / max(url_len, 1)
            letter_ratio = char_counts[1] / max(url_len, 1)
            special_ratio = (char_counts[2] + char_counts[3]) / max(url_len, 1)
            
            # Domain features
            domain_parts = parsed.netloc.split('.')
            num_domain_parts = len(domain_parts)
            has_www = 1 if 'www' in domain_parts else 0
            
            # Entropy
            url_entropy = entropy([url.count(c) for c in set(url)]) if len(set(url)) > 1 else 0
            
            # Keyword matching
            url_lower = url.lower()
            keyword_flags = [
                any(kw in url_lower for kw in kws)
                for kws in [ARTICLE_KEYWORDS, PRODUCT_KEYWORDS, LIST_KEYWORDS, BRAND_KEYWORDS]
            ]
            
            # Extension features
            ext = self.normalizer.extract_extension(url)
            has_content_ext = 1 if ext in ['html', 'htm', 'php', 'asp', 'jsp'] else 0
            is_static = 1 if self.normalizer.is_static_resource(url) else 0
            
            features = [
                url_len, path_len, query_len,
                num_segments, avg_seg_len, max_seg_len, num_params,
                *char_counts,
                digit_ratio, letter_ratio, special_ratio,
                num_domain_parts, has_www, url_entropy,
                *keyword_flags, has_content_ext, is_static
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return np.zeros(24, dtype=np.float32)
    
    def extract_interaction_features(self, urls: List[str], anchors: List[str]) -> np.ndarray:
        """Extract interaction features between URL and anchor text"""
        features = []
        
        for url, anchor in zip(urls, anchors):
            if not anchor:
                features.append([0, 0, 0, 0, 0, 0])
                continue
            
            url_lower, anchor_lower = url.lower(), anchor.lower()
            
            # Word overlap
            url_words = set(re.findall(r'\w+', url_lower))
            anchor_words = set(re.findall(r'\w+', anchor_lower))
            common = len(url_words & anchor_words)
            jaccard = common / max(len(url_words | anchor_words), 1)
            
            # Length ratio
            len_ratio = len(anchor) / max(len(url), 1)
            
            # Keyword flags in anchor
            keyword_flags = [
                any(kw in anchor_lower for kw in kws)
                for kws in [ARTICLE_KEYWORDS, PRODUCT_KEYWORDS, LIST_KEYWORDS]
            ]
            
            features.append([common, jaccard, len_ratio, *keyword_flags])
        
        return np.array(features, dtype=np.float32)
    
    def extract_tfidf_features(
        self, 
        urls: List[str], 
        anchors: Optional[List[str]] = None,
        is_training: bool = True
    ):
        """Extract TF-IDF features from URLs and anchors"""
        if self.config.url_normalization:
            urls = [self.normalizer.normalize(u) for u in urls]
        
        anchor_texts = [(a or '') for a in (anchors or [''] * len(urls))]
        
        if is_training:
            # URL TF-IDF
            self.tfidf_url = TfidfVectorizer(
                max_features=self.config.max_tfidf_features,
                analyzer='char',
                ngram_range=self.config.tfidf_ngram_range,
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )
            
            # Anchor TF-IDF
            self.tfidf_anchor = TfidfVectorizer(
                max_features=min(3000, self.config.max_tfidf_features // 3),
                analyzer='char',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )
            
            X_url = self.tfidf_url.fit_transform(urls)
            X_anchor = self.tfidf_anchor.fit_transform(anchor_texts)
            logger.info(f"TF-IDF - URL: {X_url.shape}, Anchor: {X_anchor.shape}")
        else:
            X_url = self.tfidf_url.transform(urls)
            X_anchor = self.tfidf_anchor.transform(anchor_texts)
        
        return hstack([X_url, X_anchor], format='csr')
    
    def extract_categorical_features(
        self,
        locations: List[str],
        urls: List[str],
        is_training: bool = True
    ):
        """Extract categorical features"""
        if is_training:
            self.ohe_location.fit(np.array(locations).reshape(-1, 1))
            extensions = [self.normalizer.extract_extension(u) for u in urls]
            self.ohe_extension.fit(np.array(extensions).reshape(-1, 1))
        
        X_loc = self.ohe_location.transform(np.array(locations).reshape(-1, 1))
        extensions = [self.normalizer.extract_extension(u) for u in urls]
        X_ext = self.ohe_extension.transform(np.array(extensions).reshape(-1, 1))
        
        return hstack([X_loc, X_ext], format='csr')
    
    def extract_urlbert_features(self, texts: List[str]) -> np.ndarray:
        """Extract URLBERT embeddings with optional multi-layer fusion"""
        max_len = min(self.config.urlbert_max_length, 
                     getattr(self.model.config, "max_position_embeddings", 64))
        
        class _Dataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, i):
                return self.data[i]
        
        loader = DataLoader(
            _Dataset(texts),
            batch_size=self.config.urlbert_batch_size,
            shuffle=False,
            num_workers=0
        )
        
        features = []
        self.model.eval()
        
        with torch.inference_mode():
            for batch in loader:
                encoded = self.tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=bool(self.config.urlbert_extract_layers)
                )
                
                # Mean pooling or CLS token
                if self.config.urlbert_use_mean_pooling:
                    hidden = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                    sum_emb = torch.sum(hidden * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    batch_feat = (sum_emb / sum_mask).cpu().numpy()
                else:
                    batch_feat = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Multi-layer fusion (if specified)
                if self.config.urlbert_extract_layers:
                    layer_feats = []
                    for idx in self.config.urlbert_extract_layers:
                        layer_out = outputs.hidden_states[idx]
                        if self.config.urlbert_use_mean_pooling:
                            mask_exp = attention_mask.unsqueeze(-1).expand(layer_out.size()).float()
                            layer_feat = torch.sum(layer_out * mask_exp, dim=1)
                            layer_feat = layer_feat / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
                        else:
                            layer_feat = layer_out[:, 0, :]
                        layer_feats.append(layer_feat.cpu().numpy())
                    
                    batch_feat = np.concatenate([batch_feat] + layer_feats, axis=1)
                
                features.append(batch_feat)
        
        return np.vstack(features).astype(np.float32)
    
    def extract_all_features(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        is_training: bool = True
    ):
        """Extract and combine all feature types"""
        logger.info(f"Extracting features (training mode: {is_training})...")
        
        # Default values
        if anchors is None:
            anchors = [''] * len(urls)
        if locations is None:
            locations = ['body'] * len(urls)
        
        # 1. Structural features (parallelized)
        with ThreadPoolExecutor(max_workers=4) as executor:
            structural = np.vstack(list(executor.map(self.extract_structural_features, urls)))
        logger.info(f"Structural features: {structural.shape}")
        
        # 2. Interaction features
        interaction = self.extract_interaction_features(urls, anchors)
        logger.info(f"Interaction features: {interaction.shape}")
        
        # 3. TF-IDF features
        X_text = self.extract_tfidf_features(urls, anchors, is_training)
        
        # 4. Categorical features
        X_cat = self.extract_categorical_features(locations, urls, is_training)
        
        # 5. URLBERT features
        X_bert = self.extract_urlbert_features(urls)
        logger.info(f"URLBERT features: {X_bert.shape}")
        
        # 6. Combine all features
        dense_features = np.hstack([structural, interaction, X_bert]).astype(np.float32)
        
        if is_training:
            self.scaler.fit(dense_features)
        dense_features = self.scaler.transform(dense_features)
        
        X_dense = csr_matrix(dense_features)
        X = hstack([X_text, X_cat, X_dense], format='csr')
        
        logger.info(f"Final feature shape: {X.shape}")
        return X


# ==================== Data Processing ====================

class DataProcessor:
    """Data preprocessing and cleaning utilities"""
    
    @staticmethod
    def load_and_clean(
        json_file: str,
        normalizer: URLNormalizer,
        remove_duplicates: bool = True,
        min_samples: int = 5
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Load and clean data from JSON file"""
        filepath = Path(json_file)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"Raw data: {len(data)} entries")
        
        # Filter and process
        rows = []
        for item in data:
            if "url" not in item:
                continue
            
            url = item["url"]
            
            # Skip static resources
            if normalizer.is_static_resource(url):
                continue
            
            # Get label
            label_raw = item.get("label") or item.get("weak_label")
            if not label_raw or label_raw == "skip":
                continue
            
            label = map_label(label_raw, url)
            if label not in _ALLOWED_LABELS:
                continue
            
            rows.append({
                "url": url,
                "label": label,
                "anchor": item.get("anchor") or "",
                "location": item.get("location") or "body"
            })
        
        logger.info(f"Valid data: {len(rows)} entries")
        
        if len(rows) == 0:
            raise ValueError("No valid data found!")
        
        # Extract fields
        urls = [r["url"] for r in rows]
        labels = [r["label"] for r in rows]
        anchors = [r["anchor"] for r in rows]
        locations = [r["location"] for r in rows]
        
        # Remove duplicates
        if remove_duplicates:
            urls, labels, anchors, locations = DataProcessor._remove_duplicates(
                urls, labels, anchors, locations
            )
        
        # Filter low-frequency classes
        urls, labels, anchors, locations = DataProcessor._filter_low_freq(
            urls, labels, anchors, locations, min_samples
        )
        
        # Show label distribution
        label_counts = Counter(labels)
        logger.info("Label distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
        
        return urls, labels, anchors, locations
    
    @staticmethod
    def _remove_duplicates(
        urls: List[str],
        labels: List[str],
        anchors: List[str],
        locations: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Remove duplicate URLs"""
        seen = set()
        result = [[], [], [], []]
        
        for u, l, a, loc in zip(urls, labels, anchors, locations):
            if u not in seen:
                seen.add(u)
                result[0].append(u)
                result[1].append(l)
                result[2].append(a)
                result[3].append(loc)
        
        logger.info(f"Deduplication: {len(urls)} -> {len(result[0])}")
        return tuple(result)
    
    @staticmethod
    def _filter_low_freq(
        urls: List[str],
        labels: List[str],
        anchors: List[str],
        locations: List[str],
        min_samples: int
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Filter low-frequency classes"""
        label_counts = Counter(labels)
        valid_labels = {l for l, c in label_counts.items() if c >= min_samples}
        
        result = [[], [], [], []]
        for u, l, a, loc in zip(urls, labels, anchors, locations):
            if l in valid_labels:
                result[0].append(u)
                result[1].append(l)
                result[2].append(a)
                result[3].append(loc)
        
        if len(result[0]) < len(urls):
            logger.info(f"Low-frequency filter: {len(urls)} -> {len(result[0])}")
        
        return tuple(result)


# ==================== Ensemble Model ====================

class EnsembleClassifier:
    """Weighted ensemble of gradient boosting models"""
    
    def __init__(self, config: ModelConfig, n_classes: int):
        self.config = config
        self.n_classes = n_classes
        self.models = []
        self.weights = []
        
        # Create models
        if _HAVE_LGB:
            self.models.append(('lgb', self._create_lgb()))
        if _HAVE_XGB and config.use_ensemble:
            self.models.append(('xgb', self._create_xgb()))
        
        logger.info(f"Ensemble models: {[name for name, _ in self.models]}")
    
    def _create_lgb(self):
        """Create LightGBM classifier"""
        return lgb.LGBMClassifier(
            n_estimators=self.config.lgb_n_estimators,
            learning_rate=self.config.lgb_learning_rate,
            max_depth=self.config.lgb_max_depth,
            num_leaves=self.config.lgb_num_leaves,
            subsample=self.config.lgb_subsample,
            colsample_bytree=self.config.lgb_colsample_bytree,
            reg_alpha=self.config.lgb_reg_alpha,
            reg_lambda=self.config.lgb_reg_lambda,
            objective="multiclass",
            class_weight="balanced",
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=-1,
            force_col_wise=True
        )
    
    def _create_xgb(self):
        """Create XGBoost classifier"""
        return xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            learning_rate=self.config.xgb_learning_rate,
            max_depth=self.config.xgb_max_depth,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            objective='multi:softprob',
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models with validation"""
        self.weights = []
        
        for name, model in self.models:
            logger.info(f"Training {name}...")
            
            try:
                if name == 'lgb' and X_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
                    )
                elif name == 'xgb' and X_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"{name} training error: {e}, using basic training")
                model.fit(X_train, y_train)
            
            # Calculate weights based on validation accuracy
            if X_val is not None:
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                self.weights.append(val_acc)
                logger.info(f"{name} validation accuracy: {val_acc:.4f}")
            else:
                self.weights.append(1.0)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        logger.info(f"Model weights: {dict(zip([n for n, _ in self.models], self.weights))}")
    
    def predict_proba(self, X):
        """Weighted probability prediction"""
        probas = [model.predict_proba(X) * weight 
                  for (_, model), weight in zip(self.models, self.weights)]
        return np.sum(probas, axis=0)
    
    def predict(self, X):
        """Class prediction"""
        return np.argmax(self.predict_proba(X), axis=1)


# ==================== Trainer ====================

class URLBERTTrainer:
    """Main trainer class with K-fold cross-validation"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.fe = FeatureExtractor(self.config)
        self.le = LabelEncoder()
        self.model = None
        
        logger.info("Trainer initialized")
    
    def train(
        self,
        json_file: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train model with K-fold cross-validation"""
        logger.info("="*60)
        logger.info("Starting K-fold cross-validation training")
        logger.info("="*60)
        
        # 1. Load and clean data
        urls, labels, anchors, locations = DataProcessor.load_and_clean(
            json_file,
            self.fe.normalizer,
            self.config.remove_duplicates,
            self.config.min_samples_per_class
        )
        
        # 2. Encode labels
        y = self.le.fit_transform(labels)
        n_classes = len(self.le.classes_)
        logger.info(f"Classes: {n_classes}, Labels: {list(self.le.classes_)}")
        
        # 3. Extract features once
        logger.info("Extracting all features...")
        X = self.fe.extract_all_features(urls, anchors, locations, is_training=True)
        
        # 4. K-fold cross-validation
        domains = [urlparse(u).netloc for u in urls]
        gkf = GroupKFold(n_splits=self.config.n_folds)
        
        fold_results = []
        fold_models = []
        
        for fold, (idx_tr, idx_val) in enumerate(gkf.split(X, y, groups=domains), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Fold {fold}/{self.config.n_folds}")
            logger.info(f"{'='*60}")
            
            X_train, X_val = X[idx_tr], X[idx_val]
            y_train, y_val = y[idx_tr], y[idx_val]
            
            logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
            
            # Train model
            model = EnsembleClassifier(self.config, n_classes)
            model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate
            y_val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            
            logger.info(f"Fold {fold} - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            fold_results.append({'fold': fold, 'accuracy': val_acc, 'f1': val_f1})
            fold_models.append(model)
        
        # 5. Summary
        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_acc = np.std([r['accuracy'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])
        
        logger.info(f"\n{'='*60}")
        logger.info("K-fold Cross-Validation Results")
        logger.info(f"{'='*60}")
        logger.info(f"Avg Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"Avg F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        
        # 6. Select best model
        best_idx = np.argmax([r['accuracy'] for r in fold_results])
        self.model = fold_models[best_idx]
        logger.info(f"Selected best model from Fold {best_idx + 1}")
        
        # 7. Retrain on full data
        logger.info("\nRetraining on full dataset...")
        self.model = EnsembleClassifier(self.config, n_classes)
        self.model.fit(X, y)
        
        # 8. Save model
        if save_path:
            self.save_model(save_path)
        
        logger.info("="*60)
        logger.info("Training completed!")
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
    
    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        pkg = {
            'model': self.model,
            'feature_extractor': self.fe,
            'label_encoder': self.le,
            'config': self.config.to_dict()
        }
        
        logger.info(f"Saving model: {filepath}")
        
        try:
            import joblib
            joblib.dump(pkg, filepath, compress=3)
        except ImportError:
            with open(filepath, 'wb') as f:
                pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Model saved ({file_size:.2f} MB)")
    
    @staticmethod
    def load_model(filepath: str):
        """Load trained model"""
        logger.info(f"Loading model: {filepath}")
        
        try:
            import joblib
            pkg = joblib.load(filepath)
        except ImportError:
            with open(filepath, 'rb') as f:
                pkg = pickle.load(f)
        
        trainer = URLBERTTrainer(ModelConfig(**pkg['config']))
        trainer.model = pkg['model']
        trainer.fe = pkg['feature_extractor']
        trainer.le = pkg['label_encoder']
        
        logger.info("Model loaded successfully")
        return trainer
    
    def predict(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None
    ) -> List[str]:
        """Predict URL categories"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = self.fe.extract_all_features(urls, anchors, locations, is_training=False)
        y_pred = self.model.predict(X)
        return list(self.le.inverse_transform(y_pred))
    
    def predict_proba(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None
    ) -> np.ndarray:
        """Predict URL category probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = self.fe.extract_all_features(urls, anchors, locations, is_training=False)
        return self.model.predict_proba(X)


# ==================== Main ====================

def main():
    """Example usage"""
    
    # Configuration
    config = ModelConfig(
        max_tfidf_features=10000,
        urlbert_use_mean_pooling=True,
        urlbert_extract_layers=[-4, -3, -2, -1],
        use_ensemble=True,
        n_folds=5
    )
    
    # Create trainer
    trainer = URLBERTTrainer(config)
    
    # Train with K-fold CV
    results = trainer.train(
        json_file="training_data/labeled_urls.json",
        save_path="urlbert_optimized_model.pkl"
    )
    
    print("\nFinal Results:")
    print(f"Avg Accuracy: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Avg F1 Score: {results['avg_f1']:.4f} ± {results['std_f1']:.4f}")
    
    # Prediction example
    test_urls = [
        "https://example.com/blog/article-title",
        "https://example.com/products/item-123",
        "https://example.com/about-us"
    ]
    
    predictions = trainer.predict(test_urls)
    probas = trainer.predict_proba(test_urls)
    
    print("\nPredictions:")
    for url, pred, proba in zip(test_urls, predictions, probas):
        print(f"\nURL: {url}")
        print(f"Predicted: {pred}")
        print(f"Confidence: {dict(zip(trainer.le.classes_, proba))}")


if __name__ == "__main__":
    main()