# -*- coding: utf-8 -*-
"""
URLBERT URL Classifier - Stacking Ensemble Version
Enhanced with 47-dimensional expert features
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
from sklearn.linear_model import LogisticRegression

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
    # TF-IDF parameters
    max_tfidf_features: int = 12000
    tfidf_ngram_range: Tuple[int, int] = (1, 5)
    
    # URLBERT parameters
    urlbert_model_name: str = "CrabInHoney/urlbert-tiny-base-v4"
    urlbert_device: Optional[str] = None
    urlbert_max_length: int = 64
    urlbert_batch_size: int = 64
    urlbert_use_mean_pooling: bool = True
    urlbert_extract_layers: Optional[List[int]] = None  # e.g., [-4, -3, -2, -1]
    
    bert_embedding_dim: Optional[int] = 128
    
    # Training parameters
    random_state: int = 42
    n_folds: int = 5
    use_ensemble: bool = True
    
    # LightGBM parameters
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
    
    # Meta-model parameters
    meta_C: float = 1.0
    meta_max_iter: int = 1000
    
    # Data processing
    remove_duplicates: bool = True
    url_normalization: bool = True
    min_samples_per_class: int = 5
    early_stopping_rounds: int = 50
    
    n_jobs: int = -1
    verbose: int = 0
    
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
    """
    Implements the 47-dimension expert feature set + URLBERT + TF-IDF
    
    Feature dimensions:
    - 47 expert features (dense)
    - 128 BERT embeddings (dense, after SVD)
    - ~22,000 TF-IDF features (sparse)
    """
    
    # Path Keywords (compiled regex for performance)
    RE_PDETAIL = re.compile(r'/(produit|product|item|sku|p/)', re.I)
    RE_PLIST = re.compile(r'/(categorie|category|collection|catalog|catalogue)', re.I)
    RE_ARTICLE = re.compile(r'/(blog|article|conseils|noticias|actualites|publications)', re.I)
    RE_BRAND = re.compile(r'/(about|qui-sommes-nous|sobre-nosotros|our-story|our-values|our-history)', re.I)
    RE_LEGAL = re.compile(r'/(legal|privacy|terms|conditions|mentions[-_]?legales|avisos[-_]?legales|rgpd|gdpr|cookies)', re.I)
    RE_COMMERCE = re.compile(r'/(cart|checkout|basket|panier|paiement|login|signin|register|account|my[-_]?account)', re.I)
    RE_DATE_PATH = re.compile(r'/\d{4}[/-]\d{2}[/-]\d{2}')
    RE_LONG_NUM = re.compile(r'\d{4,}')
    RE_ENDS_NUM = re.compile(r'/\d{1,3}/?$')
    RE_MULTILANG = re.compile(r'/(fr|es|en|pt|it)[-_/]', re.I)
    
    # Anchor Keywords (enhanced with more terms)
    RE_ANCHOR_PRICE = re.compile(
        r'(£|€|\$|rrp|price|prix|ml|gummies|tabs|sachets|bags|softgel|chewable|mg|g\b)',
        re.I
    )
    RE_ANCHOR_ACTION = re.compile(
        r'(add to basket|add to cart|buy|acheter|ajouter|comprar|añadir|basket|cart)',
        re.I
    )
    RE_ANCHOR_DATE = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}')

    def __init__(self, config: ModelConfig):
        self.config = config
        self.normalizer = URLNormalizer()
        
        # Vectorizers
        self.tfidf_url = None
        self.tfidf_anchor = None
        self.tfidf_path = None
        
        # Scaler for dense features
        self.scaler = StandardScaler(with_mean=False)
        
        # SVD for BERT embedding reduction
        self.svd_bert = None
        if self.config.bert_embedding_dim and self.config.bert_embedding_dim > 0:
            self.svd_bert = TruncatedSVD(
                n_components=self.config.bert_embedding_dim,
                random_state=self.config.random_state
            )
        
        # Categorical feature indices (for LGBM)
        self.categorical_feature_indices_ = []  # Currently no categorical features
        
        # URLBERT
        if config.verbose > 0:
            logger.info(f"Loading URLBERT: {config.urlbert_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.urlbert_model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(config.urlbert_model_name)
        
        model_max_length = getattr(self.model.config, 'max_position_embeddings', 512)
        self.actual_max_length = min(config.urlbert_max_length, model_max_length)
        
        if self.actual_max_length != config.urlbert_max_length and config.verbose > 0:
            logger.info(f"Adjusted max_length from {config.urlbert_max_length} to {self.actual_max_length}")
        
        self.device = torch.device(
            config.urlbert_device if config.urlbert_device 
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model.to(self.device)
        self.model.eval()
        
        if config.verbose > 0:
            logger.info(f"Device: {self.device}")

    def extract_47_features(self, url: str, anchor: str, location: str) -> np.ndarray:
        """
        Extract 47 expert features based on data analysis
        
        Returns:
            np.ndarray: 47-dimensional feature vector
        """
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            query = parsed.query.lower()
            anchor = anchor.lower()
            location = location.strip().lower()  # Fixed: use strip().lower()
            
            # ===== 1. Base Structural Features (15 dims) =====
            url_length = len(url)
            path_length = len(path)
            query_length = len(query)
            
            segments = [s for s in path.split('/') if s]
            path_depth = len(segments)
            avg_segment_length = np.mean([len(s) for s in segments]) if segments else 0
            max_segment_length = max([len(s) for s in segments]) if segments else 0
            
            n_query_params = len(parse_qs(query))
            
            digit_ratio = sum(c.isdigit() for c in url) / max(1, url_length)
            n_hyphens = url.count('-')
            n_underscores = url.count('_')
            n_slashes = url.count('/')
            n_dots = url.count('.')
            
            has_long_number = 1.0 if self.RE_LONG_NUM.search(url) else 0.0
            ends_with_slash = 1.0 if path.endswith('/') else 0.0
            is_homepage = 1.0 if path in ['', '/'] else 0.0
            
            base_features = [
                url_length, path_length, query_length,
                path_depth, avg_segment_length, max_segment_length,
                n_query_params, digit_ratio, n_hyphens, n_underscores,
                n_slashes, n_dots, has_long_number, ends_with_slash, is_homepage
            ]
            
            # ===== 2. Path Semantic Features (18 dims) =====
            # Product Detail (3 dims)
            has_product_detail = 1.0 if self.RE_PDETAIL.search(path) else 0.0
            has_category = 1.0 if self.RE_PLIST.search(path) else 0.0
            has_list_exclude = has_category  # Used for exclusion logic
            
            # Optimized scoring based on data analysis
            product_detail_score = (
                has_product_detail * 2.0 +        # 70% coverage - main signal
                (1.0 - has_category) * 1.5 +      # Exclude list keywords
                (1.0 if 2 <= path_depth <= 3 else 0.0)  # Typical depth
            )
            
            # Product List (3 dims)
            has_list_indicator = has_category
            product_list_score = (
                has_category * 2.0 +               # 23% coverage - important
                has_list_indicator * 0.5 +
                (1.0 if path_depth >= 3 else 0.0) # Usually deeper
            )
            
            # Article Page (4 dims)
            has_blog = 1.0 if self.RE_ARTICLE.search(path) else 0.0
            has_date_in_path = 1.0 if self.RE_DATE_PATH.search(path) else 0.0
            has_article_keywords = has_blog
            article_page_score = (
                has_blog * 1.5 +                   # 35% coverage
                has_date_in_path * 2.0             # Strong signal (3%)
            )
            
            # Article List (3 dims)
            ends_with_number = 1.0 if self.RE_ENDS_NUM.search(path) else 0.0
            has_list_pattern = 1.0 if (has_category or ends_with_number) else 0.0
            article_list_score = (
                has_blog * 1.5 +                   # 55% coverage
                has_category * 0.8 +               # 20% coverage
                ends_with_number * 0.6             # 21% coverage (pagination)
            )
            
            # Brand Info (2 dims)
            has_about = 1.0 if self.RE_BRAND.search(path) else 0.0
            is_root = is_homepage
            
            # Legal/Irrelevant (2 dims)
            has_legal = 1.0 if self.RE_LEGAL.search(path) else 0.0
            has_commerce = 1.0 if self.RE_COMMERCE.search(path) else 0.0
            
            # NEW: Multilingual indicator (1 dim) - completes 47 features
            is_multilingual = 1.0 if self.RE_MULTILANG.search(path) else 0.0
            
            semantic_features = [
                has_product_detail, has_list_exclude, product_detail_score,
                has_category, has_list_indicator, product_list_score,
                has_blog, has_date_in_path, has_article_keywords, article_page_score,
                ends_with_number, has_list_pattern, article_list_score,
                has_about, is_root,
                has_legal, has_commerce, is_multilingual
            ]  # 18 dims
            
            # ===== 3. Anchor Features (8 dims) =====
            anchor_length = len(anchor)
            anchor_words = anchor.split()
            anchor_word_count = len(anchor_words)
            
            anchor_has_price = 1.0 if self.RE_ANCHOR_PRICE.search(anchor) else 0.0
            anchor_has_action = 1.0 if self.RE_ANCHOR_ACTION.search(anchor) else 0.0
            anchor_has_date = 1.0 if self.RE_ANCHOR_DATE.search(anchor) else 0.0
            anchor_is_title_like = 1.0 if (anchor_word_count >= 3 and not anchor_has_price) else 0.0
            anchor_is_short_nav = 1.0 if (1 <= anchor_word_count <= 3 and anchor_length < 30) else 0.0
            anchor_is_single_word = 1.0 if anchor_word_count == 1 else 0.0
            
            anchor_features = [
                anchor_length, anchor_word_count,
                anchor_has_price, anchor_has_action,
                anchor_has_date, anchor_is_title_like,
                anchor_is_short_nav, anchor_is_single_word
            ]
            
            # ===== 4. Location Features (3 dims - OHE) =====
            # Fixed: strict matching
            loc_header = 1.0 if location == 'header' else 0.0
            loc_body = 1.0 if location == 'body' else 0.0
            loc_footer = 1.0 if location == 'footer' else 0.0
            
            location_features = [loc_header, loc_body, loc_footer]
            
            # ===== 5. Cross Features (3 dims) =====
            header_x_product = loc_header * (has_product_detail + has_category)
            body_x_blog_x_long_anchor = loc_body * has_blog * (1.0 if anchor_length > 30 else 0.0)
            footer_x_legal = loc_footer * has_legal
            
            cross_features = [header_x_product, body_x_blog_x_long_anchor, footer_x_legal]
            
            # ===== Combine All (47 dims) =====
            all_47_features = np.concatenate([
                base_features,      # 15
                semantic_features,  # 18
                anchor_features,    # 8
                location_features,  # 3
                cross_features      # 3
            ])
            
            assert len(all_47_features) == 47, f"Feature dimension error: {len(all_47_features)}"
            
            return all_47_features.astype(np.float32)
        
        except Exception as e:
            if self.config.verbose > 0:
                logger.warning(f"Feature extraction failed")
                logger.warning(f"  URL: {url[:100]}...")
                logger.warning(f"  Error: {type(e).__name__}: {str(e)}")
            return np.zeros(47, dtype=np.float32)

    def extract_tfidf_features(
        self,
        urls: List[str],
        anchors: Optional[List[str]],
        is_training: bool
    ):
        """Extract TF-IDF features"""
        # URL TF-IDF
        if is_training:
            self.tfidf_url = TfidfVectorizer(
                max_features=self.config.max_tfidf_features,
                ngram_range=self.config.tfidf_ngram_range,
                analyzer='char',
                min_df=2,
                dtype=np.float32
            )
            url_tfidf = self.tfidf_url.fit_transform(urls)
        else:
            url_tfidf = self.tfidf_url.transform(urls)
        
        # Anchor TF-IDF
        if anchors and any(anchor.strip() for anchor in anchors):
            try:
                if is_training:
                    self.tfidf_anchor = TfidfVectorizer(
                        max_features=self.config.max_tfidf_features // 2,
                        ngram_range=(1, 3),
                        min_df=2,
                        dtype=np.float32
                    )
                    anchor_tfidf = self.tfidf_anchor.fit_transform(anchors)
                else:
                    if self.tfidf_anchor is not None:
                        anchor_tfidf = self.tfidf_anchor.transform(anchors)
                    else:
                        anchor_tfidf = csr_matrix((len(urls), 0), dtype=np.float32)
            except ValueError:
                if is_training:
                    self.tfidf_anchor = None
                anchor_tfidf = csr_matrix((len(urls), 0), dtype=np.float32)
        else:
            if is_training:
                self.tfidf_anchor = None
            anchor_tfidf = csr_matrix((len(urls), 0), dtype=np.float32)
        
        # Path TF-IDF
        paths = [urlparse(u).path for u in urls]
        if is_training:
            self.tfidf_path = TfidfVectorizer(
                max_features=self.config.max_tfidf_features // 3,
                ngram_range=(1, 4),
                analyzer='char',
                min_df=2,
                dtype=np.float32
            )
            path_tfidf = self.tfidf_path.fit_transform(paths)
        else:
            path_tfidf = self.tfidf_path.transform(paths)
        
        return hstack([url_tfidf, anchor_tfidf, path_tfidf], format='csr')

    def extract_urlbert_embeddings(self, urls: List[str], is_training: bool) -> np.ndarray:
        """Extract URLBERT embeddings with optional multi-layer extraction"""
        class URLDataset(Dataset):
            def __init__(self, urls, tokenizer, max_length):
                self.urls = urls
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.urls)
            
            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.urls[idx],
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {k: v.squeeze(0) for k, v in encoding.items()}
        
        dataset = URLDataset(urls, self.tokenizer, self.actual_max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.urlbert_batch_size,
            shuffle=False,
            num_workers=0
        )
        
        embeddings = []
        
        with torch.inference_mode():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                model_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'output_hidden_states': True
                }
                
                if 'token_type_ids' in batch:
                    model_inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                
                outputs = self.model(**model_inputs)
                
                # Support multi-layer extraction
                if self.config.urlbert_extract_layers:
                    hidden_states = outputs.hidden_states
                    layer_embeddings = [hidden_states[i][:, 0, :] 
                                      for i in self.config.urlbert_extract_layers]
                    
                    if self.config.urlbert_use_mean_pooling:
                        last_hidden = outputs.last_hidden_state
                        mask = attention_mask.unsqueeze(-1)
                        masked_hidden = last_hidden * mask
                        sum_embeddings = torch.sum(masked_hidden, dim=1)
                        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                        mean_pooled = sum_embeddings / sum_mask
                        layer_embeddings.append(mean_pooled)
                    
                    batch_embedding = torch.cat(layer_embeddings, dim=1)
                else:
                    # Single layer extraction
                    if self.config.urlbert_use_mean_pooling:
                        last_hidden = outputs.last_hidden_state
                        mask = attention_mask.unsqueeze(-1)
                        masked_hidden = last_hidden * mask
                        sum_embeddings = torch.sum(masked_hidden, dim=1)
                        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                        batch_embedding = sum_embeddings / sum_mask
                    else:
                        batch_embedding = outputs.last_hidden_state[:, 0, :]
                
                embeddings.append(batch_embedding.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        # Apply SVD dimensionality reduction
        if self.svd_bert is not None:
            if is_training:
                embeddings = self.svd_bert.fit_transform(embeddings)
            else:
                embeddings = self.svd_bert.transform(embeddings)
        
        return embeddings.astype(np.float32)
    
    def extract_all_features(
        self,
        urls: List[str],
        anchors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        is_training: bool = False
    ) -> np.ndarray:
        """
        Extract and combine all features:
        [Scaled 47 Expert Features + Scaled BERT] | [Sparse TF-IDF]
        """
        
        # 1. Extract 47 Expert Features (Dense)
        if self.config.verbose > 1:
            logger.info("  Extracting 47 expert features...")
        features_47 = np.vstack([
            self.extract_47_features(u, a, l) 
            for u, a, l in zip(urls, anchors, locations)
        ])
        
        # 2. Extract URLBERT Embeddings (Dense)
        if self.config.verbose > 1:
            logger.info("  Extracting URLBERT embeddings...")
        bert = self.extract_urlbert_embeddings(urls, is_training)
        
        # 3. Extract TF-IDF Features (Sparse)
        if self.config.verbose > 1:
            logger.info("  Extracting TF-IDF features...")
        tfidf = self.extract_tfidf_features(urls, anchors, is_training)
        
        # 4. Combine and Scale Dense Features
        dense_features = np.hstack([features_47, bert])
        
        if is_training:
            scaled_dense = self.scaler.fit_transform(dense_features)
        else:
            scaled_dense = self.scaler.transform(dense_features)
        
        # 5. Combine All (Dense-Sparse)
        dense_sparse = csr_matrix(scaled_dense)
        X = hstack([dense_sparse, tfidf], format='csr')
        
        if self.config.verbose > 0 and is_training:
            logger.info(f"Total feature dimension: {X.shape[1]}")
            logger.info(f"  Dense (Expert+BERT): {scaled_dense.shape[1]} (47 + {bert.shape[1]})")
            logger.info(f"  Sparse (TF-IDF): {tfidf.shape[1]}")
        
        # Set categorical feature indices (empty for now)
        if is_training:
            self.categorical_feature_indices_ = []
        
        return X


# ==================== Data Processing ====================

class DataProcessor:
    """Data loading and cleaning"""
    
    @staticmethod
    def load_and_clean(
        json_file: str,
        normalizer: URLNormalizer,
        remove_duplicates: bool,
        min_samples_per_class: int,
        verbose: int
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Load and clean data"""
        if verbose > 0:
            logger.info(f"\nLoading data from: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if verbose > 0:
            logger.info(f"Raw samples: {len(data)}")
        
        urls, labels, anchors, locations = [], [], [], []
        
        for item in data:
            url = item.get('url', '').strip()
            label = item.get('label') or item.get('weak_label')
            if not url or not label:
                continue
            
            label = label.strip().lower()
            anchor = item.get('anchor', '').strip()
            location = item.get('location', 'body').strip().lower()
            
            url = normalizer.normalize(url)
            
            if normalizer.is_static_resource(url):
                continue
            
            label = map_label(label, url)
            
            if label not in ALLOWED_LABELS:
                continue
            
            urls.append(url)
            labels.append(label)
            anchors.append(anchor)
            locations.append(location)
        
        if verbose > 0:
            logger.info(f"After filtering: {len(urls)}")
        
        # Remove duplicates
        if remove_duplicates:
            seen = {}
            unique_indices = []
            for i, url in enumerate(urls):
                if url not in seen:
                    seen[url] = i
                    unique_indices.append(i)
            
            urls = [urls[i] for i in unique_indices]
            labels = [labels[i] for i in unique_indices]
            anchors = [anchors[i] for i in unique_indices]
            locations = [locations[i] for i in unique_indices]
            
            if verbose > 0:
                logger.info(f"After deduplication: {len(urls)}")
        
        # Filter by class frequency
        label_counts = Counter(labels)
        valid_labels = {label for label, count in label_counts.items() if count >= min_samples_per_class}
        
        indices = [i for i, label in enumerate(labels) if label in valid_labels]
        urls = [urls[i] for i in indices]
        labels = [labels[i] for i in indices]
        anchors = [anchors[i] for i in indices]
        locations = [locations[i] for i in indices]
        
        if verbose > 0:
            logger.info(f"Final samples: {len(urls)}")
            logger.info("Class distribution:")
            for label, count in sorted(Counter(labels).items()):
                logger.info(f"  {label}: {count}")
        
        return urls, labels, anchors, locations


# ==================== Stacking Ensemble ====================

class StackingEnsembleClassifier:
    """Stacking ensemble with out-of-fold predictions and meta-model"""
    
    def __init__(self, config: ModelConfig, n_classes: int):
        self.config = config
        self.n_classes = n_classes
        self.base_models = []
        self.meta_model = None
        
        # Initialize base models
        if HAS_LGB:
            self.base_models.append(('lgb', self._create_lgb()))
        if HAS_XGB and config.use_ensemble:
            self.base_models.append(('xgb', self._create_xgb()))
        
        # Initialize meta model
        self.meta_model = LogisticRegression(
            C=config.meta_C,
            max_iter=config.meta_max_iter,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )
    
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
    
    def fit_base_models(
        self, 
        X_train, 
        y_train, 
        X_val=None, 
        y_val=None, 
        categorical_feature: List[int] = None
    ):
        """Train base models (called during K-fold to generate OOF predictions)"""
        trained_models = []
        
        for name, model_template in self.base_models:
            # Create a fresh instance
            if name == 'lgb':
                model = self._create_lgb()
            elif name == 'xgb':
                model = self._create_xgb()
            else:
                continue
            
            if self.config.verbose > 1:
                logger.info(f"    Training {name}...")
            
            # Prepare fit parameters
            fit_params = {}
            if X_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
            
            try:
                if name == 'lgb':
                    if X_val is not None:
                        fit_params['callbacks'] = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
                    # Pass categorical feature indices
                    if categorical_feature:
                        fit_params['categorical_feature'] = categorical_feature
                    model.fit(X_train, y_train, **fit_params)
                
                elif name == 'xgb':
                    if X_val is not None:
                        fit_params['verbose'] = False
                    model.fit(X_train, y_train, **fit_params)
                
            except Exception as e:
                if self.config.verbose > 1:
                    logger.warning(f"    {name} error: {e}")
                model.fit(X_train, y_train)
            
            trained_models.append((name, model))
        
        return trained_models
    
    def fit(self, X, y, oof_predictions: np.ndarray, categorical_feature: List[int] = None):
        """
        Fit the stacking ensemble using out-of-fold predictions
        
        Args:
            X: Full training data
            y: Full training labels
            oof_predictions: Out-of-fold predictions [n_samples, n_base_models * n_classes]
            categorical_feature: Indices of categorical features
        """
        # Step 1: Retrain base models on full training data
        if self.config.verbose > 0:
            logger.info("\nRetraining base models on full dataset...")
        
        trained_models = []
        for name, model_template in self.base_models:
            if name == 'lgb':
                model = self._create_lgb()
            elif name == 'xgb':
                model = self._create_xgb()
            else:
                continue
            
            if self.config.verbose > 1:
                logger.info(f"  Training {name}...")
            
            try:
                if name == 'lgb' and categorical_feature:
                    model.fit(X, y, categorical_feature=categorical_feature)
                else:
                    model.fit(X, y)
            except Exception as e:
                if self.config.verbose > 1:
                    logger.warning(f"  {name} error: {e}")
                model.fit(X, y)
            
            trained_models.append((name, model))
        
        self.base_models = trained_models
        
        # Step 2: Train meta-model on OOF predictions
        if self.config.verbose > 0:
            logger.info("Training meta-model on OOF predictions...")
        
        self.meta_model.fit(oof_predictions, y)
        
        if self.config.verbose > 1:
            train_pred = self.meta_model.predict(oof_predictions)
            train_acc = accuracy_score(y, train_pred)
            logger.info(f"  Meta-model training accuracy: {train_acc:.4f}")
    
    def predict_proba(self, X):
        """Predict probabilities using stacking"""
        base_predictions = []
        for name, model in self.base_models:
            pred_proba = model.predict_proba(X)
            base_predictions.append(pred_proba)
        
        stacked_predictions = np.hstack(base_predictions)
        return self.meta_model.predict_proba(stacked_predictions)
    
    def predict(self, X):
        """Predict classes"""
        return np.argmax(self.predict_proba(X), axis=1)


# ==================== Trainer ====================

class URLBERTTrainer:
    """Main trainer with Stacking ensemble"""
    
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
        """Train with K-fold CV and Stacking"""
        if self.config.verbose > 0:
            logger.info("="*60)
            logger.info("Training URLBERT Stacking Classifier")
        
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
        cat_indices = self.fe.categorical_feature_indices_
        
        # K-fold CV
        domains = [urlparse(u).netloc for u in urls]
        gkf = GroupKFold(n_splits=self.config.n_folds)
        
        # Initialize OOF predictions
        n_base_models = 2 if (HAS_LGB and HAS_XGB and self.config.use_ensemble) else 1
        oof_predictions = np.zeros((len(y), n_base_models * n_classes), dtype=np.float32)
        
        fold_results = []
        
        for fold, (idx_tr, idx_val) in enumerate(gkf.split(X, y, groups=domains), 1):
            if self.config.verbose > 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fold {fold}/{self.config.n_folds}")
                logger.info(f"{'='*60}")
            
            X_train, X_val = X[idx_tr], X[idx_val]
            y_train, y_val = y[idx_tr], y[idx_val]
            
            # Create temporary ensemble for this fold
            temp_ensemble = StackingEnsembleClassifier(self.config, n_classes)
            
            # Train base models
            trained_models = temp_ensemble.fit_base_models(
                X_train, y_train, X_val, y_val, 
                categorical_feature=cat_indices  # Fixed: pass categorical features
            )
            
            # Generate OOF predictions
            fold_oof_preds = []
            for name, model in trained_models:
                pred_proba = model.predict_proba(X_val)
                fold_oof_preds.append(pred_proba)
                
                if self.config.verbose > 1:
                    val_pred = model.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                    logger.info(f"    {name} val accuracy: {val_acc:.4f}")
            
            fold_oof_stacked = np.hstack(fold_oof_preds)
            oof_predictions[idx_val] = fold_oof_stacked
            
            # Train temporary meta-model for validation
            temp_meta = LogisticRegression(
                C=self.config.meta_C,
                max_iter=self.config.meta_max_iter,
                multi_class='multinomial',
                solver='lbfgs',
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            temp_meta.fit(fold_oof_stacked, y_val)
            
            # Evaluate stacking on validation set
            y_val_pred = temp_meta.predict(fold_oof_stacked)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            
            if self.config.verbose > 0:
                logger.info(f"  Stacking Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            fold_results.append({'fold': fold, 'accuracy': val_acc, 'f1': val_f1})
        
        # Summary
        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_acc = np.std([r['accuracy'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])
        
        if self.config.verbose > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Cross-Validation Results")
            logger.info(f"{'='*60}")
            logger.info(f"Avg Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
            logger.info(f"Avg F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
            logger.info(f"{'='*60}")
        
        # Train final stacking model
        if self.config.verbose > 0:
            logger.info("\nTraining final stacking ensemble...")
        
        self.model = StackingEnsembleClassifier(self.config, n_classes)
        self.model.fit(X, y, oof_predictions, categorical_feature=cat_indices)
        
        # Save
        if save_path:
            self.save_model(save_path)
        
        if self.config.verbose > 0:
            logger.info("\n✓ Training completed!")
        
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
        urlbert_extract_layers=None,  # Set to [-4, -3, -2, -1] for multi-layer
        bert_embedding_dim=128,
        use_ensemble=True,
        n_folds=5,
        meta_C=1.0,
        meta_max_iter=1000,
        verbose=1  # 0=minimal, 1=normal, 2=detailed
    )
    
    # Train
    trainer = URLBERTTrainer(config)
    results = trainer.train(
        json_file="training_data/labeled_urls.json",
        save_path="urlbert_stacking_model.pkl"
    )
    
    print(f"\n{'='*60}")
    print(f"Final Results")
    print(f"{'='*60}")
    print(f"✓ Accuracy: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"✓ F1 Score: {results['avg_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"{'='*60}")
    
    # Example prediction
    test_urls = [
        "https://www.nutrivet.fr/produit/croquettes-chat-poisson/",
        "https://www.nutrivet.fr/categorie-produit/votre-chien/"
    ]
    test_anchors = ["", "Votre chien"]
    test_locations = ["header", "header"]
    
    predictions = trainer.predict(test_urls, test_anchors, test_locations)
    probas = trainer.predict_proba(test_urls, test_anchors, test_locations)
    
    print("\nExample Predictions:")
    for url, pred, proba in zip(test_urls, predictions, probas):
        print(f"\nURL: {url}")
        print(f"  → {pred} (confidence: {max(proba):.2%})")


if __name__ == "__main__":
    main()