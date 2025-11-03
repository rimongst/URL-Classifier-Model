# -*- coding: utf-8 -*-
"""
URLBERT URL Classifier
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import TruncatedSVD

# Tree models
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Transformers
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset

# Simple logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class ModelConfig:
    """Model configuration optimized for accuracy"""
    # TF-IDF parameters (balanced for accuracy)
    max_tfidf_features: int = 12000
    tfidf_ngram_range: Tuple[int, int] = (1, 5)
    
    # URLBERT parameters
    urlbert_model_name: str = "CrabInHoney/urlbert-tiny-base-v4"
    urlbert_device: Optional[str] = None
    urlbert_max_length: int = 128
    urlbert_batch_size: int = 64
    urlbert_use_mean_pooling: bool = True
    urlbert_extract_layers: Optional[List[int]] = None
    
    # *** OPTIMIZATION: Add SVD for BERT embeddings ***
    bert_embedding_dim: Optional[int] = 128
    
    # Training parameters
    random_state: int = 42
    n_folds: int = 5
    use_ensemble: bool = True
    
    # LightGBM parameters (optimized for accuracy)
    lgb_n_estimators: int = 500
    lgb_learning_rate: float = 0.03
    lgb_max_depth: int = 8
    lgb_num_leaves: int = 70
    lgb_subsample: float = 0.85
    lgb_colsample_bytree: float = 0.85
    lgb_reg_alpha: float = 1.0
    lgb_reg_lambda: float = 1.5
    lgb_min_child_samples: int = 20
    
    # XGBoost parameters
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 7
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    
    # Data processing
    remove_duplicates: bool = True
    url_normalization: bool = True
    min_samples_per_class: int = 5
    early_stopping_rounds: int = 50
    
    n_jobs: int = -1
    verbose: int = 0  # Logging verbosity: 0=minimal, 1=normal, 2=detailed
    
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

ALLOWED_LABELS = frozenset(CATEGORIES.keys())
MERGE_TO_IRRELEVANT = {"irrelevant", "legal", "account", "commerce"}
IRRELEVANT_PATTERN = re.compile(
    r"/(privacy|terms|legal|mentions[-_]?legales|rgpd|gdpr|"
    r"login|signin|signup|register|account|my[-_]?account|"
    r"cart|checkout|panier|paiement|basket)\b", re.I
)

# URL pattern keywords (expanded for better accuracy)
ARTICLE_KEYWORDS = {'article', 'post', 'blog', 'news', 'story', 'read', 'actualite', 'billet'}
PRODUCT_KEYWORDS = {'product', 'produit', 'item', 'shop', 'buy', 'goods', 'sku', 'achat'}
LIST_KEYWORDS = {'list', 'category', 'catalog', 'collection', 'archive', 'categorie', 'liste'}
BRAND_KEYWORDS = {'brand', 'about', 'company', 'store', 'marque', 'entreprise'}

STATIC_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.svg', '.ico', '.woff', '.ttf'}


def map_label(label: str, url: str) -> str:
    """Map label to standard categories"""
    if label in MERGE_TO_IRRELEVANT or IRRELEVANT_PATTERN.search(url):
        return "irrelevant"
    return label


# ==================== URL Utilities ====================

class URLNormalizer:
    """Fast URL normalization"""
    
    TRACKING_PARAMS = frozenset({
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'fbclid', 'gclid', 'msclkid', '_ga', 'ref', 'source'
    })
    
    @staticmethod
    def normalize(url: str) -> str:
        """Normalize URL"""
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
            
            if url.endswith('/') and url.count('/') > 3:
                url = url.rstrip('/')
            
            return url
        except:
            return url.lower()
    
    @staticmethod
    def is_static_resource(url: str) -> bool:
        """Check if URL is static resource"""
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in STATIC_EXTENSIONS)
    
    @staticmethod
    def extract_extension(url: str) -> str:
        """Extract file extension"""
        path = urlparse(url).path
        return path.split('.')[-1].lower() if '.' in path else ''


# ==================== Feature Extraction ====================

class FeatureExtractor:
    """High-performance feature extraction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.normalizer = URLNormalizer()
        
        # Vectorizers
        self.tfidf_url = None
        self.tfidf_anchor = None
        self.tfidf_path = None
        
        # *** OPTIMIZATION: Replace OHE with LabelEncoder for categorical features ***
        self.le_location = LabelEncoder()
        self.le_extension = LabelEncoder()
        self.le_location_map_ = {}
        self.le_extension_map_ = {}
        self.categorical_feature_indices_ = None # Store indices for LGBM
        
        self.scaler = StandardScaler(with_mean=False)
        
        # *** OPTIMIZATION: Add SVD for BERT embedding reduction ***
        self.svd_bert = None
        if self.config.bert_embedding_dim and self.config.bert_embedding_dim > 0:
            self.svd_bert = TruncatedSVD(
                n_components=self.config.bert_embedding_dim,
                random_state=self.config.random_state
            )
        
        # URLBERT
        if config.verbose > 0:
            logger.info(f"Loading URLBERT: {config.urlbert_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.urlbert_model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(config.urlbert_model_name)
        
        self.device = torch.device(
            config.urlbert_device if config.urlbert_device 
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model.to(self.device)
        self.model.eval()
        
        if config.verbose > 0:
            logger.info(f"Device: {self.device}")
    
    def extract_structural_features(self, url: str) -> np.ndarray:
        """Extract 28 structural features from URL"""
        try:
            parsed = urlparse(url)
            path, query = parsed.path, parsed.query
            
            # Length features (3)
            url_len, path_len, query_len = len(url), len(path), len(query)
            
            # Path analysis (3)
            segments = [s for s in path.split('/') if s]
            num_segments = len(segments)
            avg_seg_len = np.mean([len(s) for s in segments]) if segments else 0
            max_seg_len = max([len(s) for s in segments]) if segments else 0
            
            # Query parameters (1)
            num_params = len(parse_qs(query))
            
            # Character counts (9)
            num_digits = sum(c.isdigit() for c in url)
            num_letters = sum(c.isalpha() for c in url)
            num_hyphens = url.count('-')
            num_underscores = url.count('_')
            num_dots = url.count('.')
            num_slashes = url.count('/')
            num_equals = url.count('=')
            num_ampersands = url.count('&')
            num_questions = url.count('?')
            
            # Ratios (3)
            digit_ratio = num_digits / max(url_len, 1)
            letter_ratio = num_letters / max(url_len, 1)
            special_ratio = (num_hyphens + num_underscores) / max(url_len, 1)
            
            # Domain features (2)
            domain_parts = parsed.netloc.split('.')
            num_domain_parts = len(domain_parts)
            has_www = 1 if 'www' in domain_parts else 0
            
            # Entropy (1)
            url_entropy = entropy([url.count(c) for c in set(url)]) if len(set(url)) > 1 else 0
            
            # Keyword matching (4)
            url_lower = url.lower()
            has_article_kw = any(kw in url_lower for kw in ARTICLE_KEYWORDS)
            has_product_kw = any(kw in url_lower for kw in PRODUCT_KEYWORDS)
            has_list_kw = any(kw in url_lower for kw in LIST_KEYWORDS)
            has_brand_kw = any(kw in url_lower for kw in BRAND_KEYWORDS)
            
            # Extension features (2)
            ext = self.normalizer.extract_extension(url)
            has_content_ext = 1 if ext in ['html', 'htm', 'php', 'asp', 'jsp'] else 0
            is_static = 1 if self.normalizer.is_static_resource(url) else 0
            
            # Total: 28 features
            features = [
                url_len, path_len, query_len,
                num_segments, avg_seg_len, max_seg_len, num_params,
                num_digits, num_letters, num_hyphens, num_underscores,
                num_dots, num_slashes, num_equals, num_ampersands, num_questions,
                digit_ratio, letter_ratio, special_ratio,
                num_domain_parts, has_www, url_entropy,
                has_article_kw, has_product_kw, has_list_kw, has_brand_kw,
                has_content_ext, is_static
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            if self.config.verbose > 1:
                logger.warning(f"Feature extraction failed: {e}")
            return np.zeros(28, dtype=np.float32)
    
    def extract_interaction_features(self, urls: List[str], anchors: List[str]) -> np.ndarray:
        """Extract URL-anchor interaction features"""
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
            
            # Keyword flags
            flags = [any(kw in anchor_lower for kw in kws)
                    for kws in [ARTICLE_KEYWORDS, PRODUCT_KEYWORDS, LIST_KEYWORDS]]
            
            features.append([common, jaccard, len_ratio, *flags])
        
        return np.array(features, dtype=np.float32)
    
    def extract_tfidf_features(
        self, 
        urls: List[str], 
        anchors: Optional[List[str]] = None,
        is_training: bool = True
    ):
        """Extract TF-IDF features (3 separate vectorizers for better accuracy)"""
        if self.config.url_normalization:
            urls = [self.normalizer.normalize(u) for u in urls]
        
        anchor_texts = [(a or '') for a in (anchors or [''] * len(urls))]
        path_texts = [urlparse(u).path for u in urls]
        
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
            
            # Path TF-IDF (separate for better accuracy)
            self.tfidf_path = TfidfVectorizer(
                max_features=min(6000, self.config.max_tfidf_features // 2),
                analyzer='char',
                ngram_range=(2, 4),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )
            
            # Anchor TF-IDF
            self.tfidf_anchor = TfidfVectorizer(
                max_features=min(3000, self.config.max_tfidf_features // 4),
                analyzer='char',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )
            
            X_url = self.tfidf_url.fit_transform(urls)
            X_path = self.tfidf_path.fit_transform(path_texts)
            X_anchor = self.tfidf_anchor.fit_transform(anchor_texts)
            
            if self.config.verbose > 1:
                logger.info(f"TF-IDF - URL:{X_url.shape} Path:{X_path.shape} Anchor:{X_anchor.shape}")
        else:
            X_url = self.tfidf_url.transform(urls)
            X_path = self.tfidf_path.transform(path_texts)
            X_anchor = self.tfidf_anchor.transform(anchor_texts)
        
        return hstack([X_url, X_path, X_anchor], format='csr')
    
    def extract_categorical_features(
        self,
        locations: List[str],
        urls: List[str],
        is_training: bool = True
    ) -> np.ndarray:
        """
        *** OPTIMIZATION: Extract categorical features as label-encoded integers ***
        Uses a map for robust handling of unknown values during inference.
        """
        locations_arr = np.array(locations)
        extensions = [self.normalizer.extract_extension(u) for u in urls]
        extensions_arr = np.array(extensions)
        
        if is_training:
            # Fit LabelEncoders
            self.le_location.fit(locations_arr)
            self.le_extension.fit(extensions_arr)
            
            # Store maps for fast, safe transformation (handles unknowns)
            # 0 is reserved for "unknown"
            self.le_location_map_ = {cls: i+1 for i, cls in enumerate(self.le_location.classes_)}
            self.le_extension_map_ = {cls: i+1 for i, cls in enumerate(self.le_extension.classes_)}
        
        # Transform using maps, .get(key, 0) maps unknown keys to 0
        loc_encoded = [self.le_location_map_.get(loc, 0) for loc in locations_arr]
        ext_encoded = [self.le_extension_map_.get(ext, 0) for ext in extensions_arr]

        return np.vstack([loc_encoded, ext_encoded]).T.astype(np.float32)

    def extract_urlbert_features(self, texts: List[str]) -> np.ndarray:
        """Extract URLBERT embeddings"""
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
                
                # Mean pooling or CLS
                if self.config.urlbert_use_mean_pooling:
                    hidden = outputs.last_hidden_state
                    mask_exp = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                    sum_emb = torch.sum(hidden * mask_exp, dim=1)
                    sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
                    batch_feat = (sum_emb / sum_mask).cpu().numpy()
                else:
                    batch_feat = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Multi-layer fusion
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
        """Extract and combine all features"""
        if self.config.verbose > 0:
            logger.info(f"Extracting features...")
        
        if anchors is None:
            anchors = [''] * len(urls)
        if locations is None:
            locations = ['body'] * len(urls)
        
        # 1. Structural features (DENSE)
        structural = np.vstack([self.extract_structural_features(u) for u in urls])
        
        # 2. Interaction features (DENSE)
        interaction = self.extract_interaction_features(urls, anchors)
        
        # 3. TF-IDF features (SPARSE)
        X_text = self.extract_tfidf_features(urls, anchors, is_training)
        
        # 4. URLBERT features (DENSE)
        X_bert_raw = self.extract_urlbert_features(urls)
        
        # 5. *** OPTIMIZATION: Reduce BERT embedding dimensions *** (DENSE)
        X_bert = X_bert_raw
        if self.svd_bert:
            if is_training:
                if self.config.verbose > 1:
                    logger.info(f"Fitting SVD to BERT embeddings ({X_bert_raw.shape[1]} -> {self.config.bert_embedding_dim})")
                self.svd_bert.fit(X_bert_raw)
            X_bert = self.svd_bert.transform(X_bert_raw)
        
        # 6. Combine DENSE numerical features
        dense_numerical = np.hstack([structural, interaction, X_bert]).astype(np.float32)
        
        if is_training:
            self.scaler.fit(dense_numerical)
        dense_numerical = self.scaler.transform(dense_numerical)
        
        X_dense_numerical_sparse = csr_matrix(dense_numerical)
        
        # 7. *** OPTIMIZATION: Get LabelEncoded categorical features *** (DENSE)
        X_cat_dense = self.extract_categorical_features(locations, urls, is_training)
        X_cat_sparse = csr_matrix(X_cat_dense)
        
        # 8. *** OPTIMIZATION: Store categorical feature indices for LGBM ***
        n_tfidf = X_text.shape[1]
        n_dense_num = X_dense_numerical_sparse.shape[1]
        
        # Indices are relative to the *final* hstacked matrix
        self.categorical_feature_indices_ = [
            n_tfidf + n_dense_num,     # First categorical feature (location)
            n_tfidf + n_dense_num + 1  # Second categorical feature (extension)
        ]
        
        # 9. Combine all
        X = hstack([X_text, X_dense_numerical_sparse, X_cat_sparse], format='csr')
        
        if self.config.verbose > 0:
            logger.info(f"Feature shape: {X.shape}")
            if self.config.verbose > 1:
                logger.info(f"  TF-IDF feats: {n_tfidf}")
                logger.info(f"  Dense/Scaled feats: {n_dense_num} (BERT reduced to {X_bert.shape[1]})")
                logger.info(f"  Categorical feats: {X_cat_sparse.shape[1]} at indices {self.categorical_feature_indices_}")

        return X


# ==================== Data Processing ====================

class DataProcessor:
    """Fast data processing"""
    
    @staticmethod
    def load_and_clean(
        json_file: str,
        normalizer: URLNormalizer,
        remove_duplicates: bool = True,
        min_samples: int = 5,
        verbose: int = 0
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Load and clean data"""
        filepath = Path(json_file)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if verbose > 0:
            logger.info(f"Loading data from: {filepath.name}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Filter and process
        rows = []
        for item in data:
            if "url" not in item:
                continue
            
            url = item["url"]
            
            if normalizer.is_static_resource(url):
                continue
            
            label_raw = item.get("label") or item.get("weak_label")
            if not label_raw or label_raw == "skip":
                continue
            
            label = map_label(label_raw, url)
            if label not in ALLOWED_LABELS:
                continue
            
            rows.append({
                "url": url,
                "label": label,
                "anchor": item.get("anchor") or "",
                "location": item.get("location") or "body"
            })
        
        if len(rows) == 0:
            raise ValueError("No valid data found!")
        
        # Extract fields
        urls = [r["url"] for r in rows]
        labels = [r["label"] for r in rows]
        anchors = [r["anchor"] for r in rows]
        locations = [r["location"] for r in rows]
        
        # Remove duplicates
        if remove_duplicates:
            seen = set()
            result = [[], [], [], []]
            for u, l, a, loc in zip(urls, labels, anchors, locations):
                if u not in seen:
                    seen.add(u)
                    result[0].append(u)
                    result[1].append(l)
                    result[2].append(a)
                    result[3].append(loc)
            urls, labels, anchors, locations = result
        
        # Filter low-frequency classes
        label_counts = Counter(labels)
        valid_labels = {l for l, c in label_counts.items() if c >= min_samples}
        result = [[], [], [], []]
        for u, l, a, loc in zip(urls, labels, anchors, locations):
            if l in valid_labels:
                result[0].append(u)
                result[1].append(l)
                result[2].append(a)
                result[3].append(loc)
        urls, labels, anchors, locations = result
        
        # Show distribution
        if verbose > 0:
            label_counts = Counter(labels)
            logger.info(f"Loaded {len(urls)} samples with {len(label_counts)} classes")
            if verbose > 1:
                for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                    logger.info(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
        
        return urls, labels, anchors, locations


# ==================== Ensemble Model ====================

class EnsembleClassifier:
    """Weighted ensemble classifier"""
    
    def __init__(self, config: ModelConfig, n_classes: int):
        self.config = config
        self.n_classes = n_classes
        self.models = []
        self.weights = []
        
        if HAS_LGB:
            self.models.append(('lgb', self._create_lgb()))
        if HAS_XGB and config.use_ensemble:
            self.models.append(('xgb', self._create_xgb()))
    
    def _create_lgb(self):
        """Create LightGBM"""
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
            force_col_wise=True
        )
    
    def _create_xgb(self):
        """Create XGBoost"""
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
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, categorical_feature: List[int] = None):
        """
        Train all models
        *** OPTIMIZATION: Pass 'categorical_feature' list to LGBM ***
        """
        self.weights = []
        
        for name, model in self.models:
            if self.config.verbose > 1:
                logger.info(f"  Training {name}...")
            
            # Prepare fit parameters
            fit_params = {}
            if X_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
            
            try:
                if name == 'lgb':
                    if X_val is not None:
                        fit_params['callbacks'] = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
                    # Pass categorical feature indices *only* to LGBM
                    fit_params['categorical_feature'] = categorical_feature
                    model.fit(X_train, y_train, **fit_params)
                
                elif name == 'xgb':
                    if X_val is not None:
                        fit_params['verbose'] = False # XGB specific
                    model.fit(X_train, y_train, **fit_params)
                
                else:
                    # For any other model
                    model.fit(X_train, y_train)

            except Exception as e:
                if self.config.verbose > 1:
                    logger.warning(f"  {name} error: {e}")
                # Fallback to simple fit
                model.fit(X_train, y_train)
            
            # Calculate weights
            if X_val is not None:
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                self.weights.append(val_acc)
                if self.config.verbose > 1:
                    logger.info(f"  {name} acc: {val_acc:.4f}")
            else:
                self.weights.append(1.0)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def predict_proba(self, X):
        """Weighted prediction"""
        probas = [model.predict_proba(X) * weight 
                  for (_, model), weight in zip(self.models, self.weights)]
        return np.sum(probas, axis=0)
    
    def predict(self, X):
        """Predict classes"""
        return np.argmax(self.predict_proba(X), axis=1)


# ==================== Trainer ====================

class URLBERTTrainer:
    """Main trainer with clean logging"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.fe = FeatureExtractor(self.config)
        self.le = LabelEncoder()
        self.model = None
    
    def train(
        self,
        json_file: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train with K-fold CV"""
        if self.config.verbose > 0:
            logger.info("="*60)
            logger.info("Training URLBERT Classifier")
            logger.info("="*60)
        
        # Load data
        urls, labels, anchors, locations = DataProcessor.load_and_clean(
            json_file,
            self.fe.normalizer,
            self.config.remove_duplicates,
            self.config.min_samples_per_class,
            self.config.verbose
        )
        
        # Encode labels
        y = self.le.fit_transform(labels)
        n_classes = len(self.le.classes_)
        
        if self.config.verbose > 0:
            logger.info(f"Classes: {n_classes}")
        
        # Extract features
        X = self.fe.extract_all_features(urls, anchors, locations, is_training=True)
        # *** OPTIMIZATION: Get fitted categorical indices from the feature extractor ***
        cat_indices = self.fe.categorical_feature_indices_
        
        # K-fold CV
        domains = [urlparse(u).netloc for u in urls]
        gkf = GroupKFold(n_splits=self.config.n_folds)
        
        fold_results = []
        fold_models = []
        
        for fold, (idx_tr, idx_val) in enumerate(gkf.split(X, y, groups=domains), 1):
            if self.config.verbose > 0:
                logger.info(f"\nFold {fold}/{self.config.n_folds}")
            
            X_train, X_val = X[idx_tr], X[idx_val]
            y_train, y_val = y[idx_tr], y[idx_val]
            
            # Train
            model = EnsembleClassifier(self.config, n_classes)
            # *** OPTIMIZATION: Pass indices to the fit method ***
            model.fit(X_train, y_train, X_val, y_val, categorical_feature=cat_indices)
            
            # Evaluate
            y_val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            
            if self.config.verbose > 0:
                logger.info(f"  Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            fold_results.append({'fold': fold, 'accuracy': val_acc, 'f1': val_f1})
            fold_models.append(model)
        
        # Summary
        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_acc = np.std([r['accuracy'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])
        
        if self.config.verbose > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Avg Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
            logger.info(f"Avg F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
            logger.info(f"{'='*60}")
        
        # Select best model
        best_idx = np.argmax([r['accuracy'] for r in fold_results])
        self.model = fold_models[best_idx]
        
        # Retrain on full data
        if self.config.verbose > 0:
            logger.info("\nRetraining on full dataset...")
        self.model = EnsembleClassifier(self.config, n_classes)
        # *** OPTIMIZATION: Pass indices to the final fit method ***
        self.model.fit(X, y, categorical_feature=cat_indices)
        
        # Save
        if save_path:
            self.save_model(save_path)
        
        if self.config.verbose > 0:
            logger.info("Training completed!")
        
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
        """Save model"""
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
        
        try:
            import joblib
            joblib.dump(pkg, filepath, compress=3)
        except ImportError:
            with open(filepath, 'wb') as f:
                pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.config.verbose > 0:
            file_size = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Model saved: {filepath.name} ({file_size:.1f} MB)")
    
    @staticmethod
    def load_model(filepath: str):
        """Load model"""
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
        
        return trainer
    
    def predict(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None
    ) -> List[str]:
        """Predict categories"""
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
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = self.fe.extract_all_features(urls, anchors, locations, is_training=False)
        return self.model.predict_proba(X)


# ==================== Main ====================

def main():
    """Example usage"""
    
    # Configuration
    config = ModelConfig(
        max_tfidf_features=12000,
        urlbert_use_mean_pooling=True,
        urlbert_extract_layers=[-4, -3, -2, -1],
        bert_embedding_dim=128,  # <-- New optimization parameter
        use_ensemble=True,
        n_folds=5,
        verbose=1  # 0=minimal, 1=normal, 2=detailed
    )
    
    # Train
    trainer = URLBERTTrainer(config)
    results = trainer.train(
        json_file="training_data/labeled_urls.json",
        save_path="urlbert_model.pkl"
    )
    
    print(f"\n✓ Accuracy: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"✓ F1 Score: {results['avg_f1']:.4f} ± {results['std_f1']:.4f}")
    
    # Predict
    test_urls = [
        "https://example.com/blog/article-title",
        "https://example.com/products/item-123"
    ]
    
    predictions = trainer.predict(test_urls)
    probas = trainer.predict_proba(test_urls)
    
    print("\nPredictions:")
    for url, pred, proba in zip(test_urls, predictions, probas):
        print(f"\n{url}")
        print(f"  → {pred} ({max(proba):.2%})")


if __name__ == "__main__":
    main()