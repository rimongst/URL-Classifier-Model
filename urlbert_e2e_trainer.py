# -*- coding: utf-8 -*-
"""
URLBERT End-to-End Two-Tower Classifier
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
from scipy.sparse import csr_matrix
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Environment settings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import TruncatedSVD

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

# Transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Simple logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class E2EConfig:
    """End-to-end model configuration"""
    # URLBERT parameters
    urlbert_model_name: str = "CrabInHoney/urlbert-tiny-base-v4"
    urlbert_device: Optional[str] = None
    urlbert_max_length: int = 64
    
    # Tower 2 (Tabular) parameters
    use_tfidf: bool = True
    tfidf_dim: int = 256  # Reduced dimensionality for TF-IDF via PCA
    max_tfidf_features: int = 8000
    tfidf_ngram_range: Tuple[int, int] = (1, 4)
    
    # Location Embedding Dim
    location_embed_dim: int = 16 # Dimension for the location embedding
    
    tabular_hidden_dim: int = 128  # MLP hidden dimension for tabular features
    tabular_output_dim: int = 64   # Final tabular embedding dimension
    tabular_dropout: float = 0.3
    
    # Fusion and classification head
    fusion_dropout: float = 0.2
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    tabular_learning_rate: float = 1e-3  # Higher LR for tabular tower
    num_epochs: int = 10
    warmup_epochs: int = 1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    use_focal_loss: bool = True
    focal_loss_gamma: float = 2.0
    
    # Cross-validation
    n_folds: int = 5
    random_state: int = 42
    
    # Data processing
    remove_duplicates: bool = True
    url_normalization: bool = True
    min_samples_per_class: int = 5
    
    # Logging
    verbose: int = 1
    
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


# ==================== Feature Extraction ====================

class FeatureExtractor:
    """Extract structural, interaction, and TF-IDF features"""
    
    def __init__(self, config: E2EConfig):
        self.config = config
        self.normalizer = URLNormalizer()
        
        # Vectorizers
        self.tfidf_url = None
        self.tfidf_pca = None
        self.scaler = StandardScaler()
        
        # LabelEncoder for location
        self.le_location = LabelEncoder()
    
    def extract_structural_features(self, url: str) -> np.ndarray:
        """Extract 28 structural features from URL"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            query = parsed.query
            
            # Basic counts
            url_length = len(url)
            path_length = len(path)
            query_length = len(query)
            
            # Path analysis
            path_segments = [s for s in path.split('/') if s]
            n_path_segments = len(path_segments)
            avg_segment_length = np.mean([len(s) for s in path_segments]) if path_segments else 0
            max_segment_length = max([len(s) for s in path_segments]) if path_segments else 0
            
            # Character statistics
            n_digits = sum(c.isdigit() for c in url)
            n_letters = sum(c.isalpha() for c in url)
            n_special = sum(not c.isalnum() for c in url)
            digit_ratio = n_digits / len(url) if len(url) > 0 else 0
            letter_ratio = n_letters / len(url) if len(url) > 0 else 0
            
            # Special characters
            n_hyphens = url.count('-')
            n_underscores = url.count('_')
            n_dots = url.count('.')
            n_slashes = url.count('/')
            
            # Query parameters
            n_query_params = len(parse_qs(query))
            
            # Entropy
            char_counts = Counter(url)
            char_probs = np.array([count / len(url) for count in char_counts.values()])
            url_entropy = entropy(char_probs)
            
            # Pattern matching
            has_article_kw = int(any(kw in url.lower() for kw in ARTICLE_KEYWORDS))
            has_product_kw = int(any(kw in url.lower() for kw in PRODUCT_KEYWORDS))
            has_list_kw = int(any(kw in url.lower() for kw in LIST_KEYWORDS))
            has_brand_kw = int(any(kw in url.lower() for kw in BRAND_KEYWORDS))
            
            # Additional features
            has_id = int(bool(re.search(r'\d{4,}', url)))
            has_date = int(bool(re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', url)))
            ends_with_slash = int(url.endswith('/'))
            is_homepage = int(path in ['', '/'])
            
            # Complexity measures
            unique_char_ratio = len(set(url)) / len(url) if len(url) > 0 else 0
            consonant_vowel_ratio = self._consonant_vowel_ratio(url)
            
            return np.array([
                url_length, path_length, query_length, n_path_segments,
                avg_segment_length, max_segment_length, n_digits, n_letters,
                n_special, digit_ratio, letter_ratio, n_hyphens, n_underscores,
                n_dots, n_slashes, n_query_params, url_entropy, has_article_kw,
                has_product_kw, has_list_kw, has_brand_kw, has_id, has_date,
                ends_with_slash, is_homepage, unique_char_ratio, consonant_vowel_ratio,
                len(parsed.netloc)
            ], dtype=np.float32)
        except:
            return np.zeros(28, dtype=np.float32)
    
    @staticmethod
    def _consonant_vowel_ratio(text: str) -> float:
        """Calculate consonant to vowel ratio"""
        vowels = set('aeiouAEIOU')
        text_letters = [c for c in text if c.isalpha()]
        if not text_letters:
            return 0.0
        n_vowels = sum(1 for c in text_letters if c in vowels)
        n_consonants = len(text_letters) - n_vowels
        return n_consonants / n_vowels if n_vowels > 0 else float(n_consonants)
    
    # *** NEW: Added interaction features from GBDT script ***
    def extract_interaction_features(self, urls: List[str], anchors: List[str]) -> np.ndarray:
        """Extract URL-anchor interaction features (6 features)"""
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
            
            # Keyword flags (3 features)
            flags = [any(kw in anchor_lower for kw in kws)
                    for kws in [ARTICLE_KEYWORDS, PRODUCT_KEYWORDS, LIST_KEYWORDS]]
            
            features.append([common, jaccard, len_ratio, *flags])
        
        return np.array(features, dtype=np.float32)

    def fit_tfidf(self, urls: List[str]):
        """Fit TF-IDF and PCA"""
        if not self.config.use_tfidf:
            return
        
        # TF-IDF
        self.tfidf_url = TfidfVectorizer(
            max_features=self.config.max_tfidf_features,
            ngram_range=self.config.tfidf_ngram_range,
            analyzer='char',
            min_df=2,
            dtype=np.float32
        )
        tfidf_matrix = self.tfidf_url.fit_transform(urls)
        
        # PCA for dimensionality reduction
        self.tfidf_pca = TruncatedSVD(
            n_components=self.config.tfidf_dim,
            random_state=self.config.random_state
        )
        self.tfidf_pca.fit(tfidf_matrix)
    
    def extract_tfidf_features(self, urls: List[str]) -> np.ndarray:
        """Extract reduced TF-IDF features"""
        if not self.config.use_tfidf or self.tfidf_url is None:
            return np.zeros((len(urls), 0), dtype=np.float32)
        
        tfidf_matrix = self.tfidf_url.transform(urls)
        tfidf_reduced = self.tfidf_pca.transform(tfidf_matrix)
        return tfidf_reduced.astype(np.float32)
    
    def extract_all_tabular_features(
        self,
        urls: List[str],
        anchors: List[str],
        locations: List[str],
        fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and combine all tabular features for the MLP.
        Returns:
            - X_tabular (for MLP): Scaled [structural, interaction, tfidf] features
            - location_ids (for Embedding): LabelEncoded location IDs
        """
        # 1. Structural features
        structural = np.vstack([self.extract_structural_features(u) for u in urls])
        
        # 2. *** NEW: Interaction features ***
        interaction = self.extract_interaction_features(urls, anchors)
        
        # 3. TF-IDF features
        if self.config.use_tfidf:
            if fit:
                self.fit_tfidf(urls)
            tfidf = self.extract_tfidf_features(urls)
            # Combine all features for the MLP
            features_for_mlp = np.hstack([structural, interaction, tfidf])
        else:
            features_for_mlp = np.hstack([structural, interaction])
        
        # 4. Scale MLP features
        if fit:
            features_for_mlp = self.scaler.fit_transform(features_for_mlp)
        else:
            features_for_mlp = self.scaler.transform(features_for_mlp)
        
        # 5. *** NEW: Handle Location features (for Embedding layer) ***
        locations_arr = np.array(locations)
        if fit:
            location_ids = self.le_location.fit_transform(locations_arr)
        else:
            # Handle unseen locations during inference
            location_ids = np.array([
                self.le_location.transform([loc])[0] if loc in self.le_location.classes_ else 0
                for loc in locations_arr
            ])
            
        return features_for_mlp.astype(np.float32), location_ids.astype(np.int64)


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
        """
        *** UPDATED: Load and clean data, returning anchors and locations ***
        """
        if verbose > 0:
            logger.info(f"\nLoading data from: {json_file}")
        
        # Load JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if verbose > 0:
            logger.info(f"Raw samples: {len(data)}")
        
        # Parse and filter
        urls, labels, anchors, locations = [], [], [], []
        
        for item in data:
            url = item.get('url', '').strip()
            label = item.get('label') or item.get('weak_label')
            
            if not url or not label:
                continue
            
            label = label.strip().lower()

            # Normalize URL
            url = normalizer.normalize(url)
            
            # Skip static resources
            if normalizer.is_static_resource(url):
                continue
            
            # Map label
            label = map_label(label, url)
            
            if label not in ALLOWED_LABELS:
                continue
            
            urls.append(url)
            labels.append(label)
            # Add anchor and location
            anchors.append(item.get('anchor', ''))
            locations.append(item.get('location', 'body'))
        
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


# ==================== PyTorch Dataset ====================

class URLDataset(Dataset):
    """
    *** UPDATED: Custom dataset now includes location_id ***
    """
    
    def __init__(
        self,
        urls: List[str],
        labels: np.ndarray,
        tabular_features: np.ndarray,
        location_ids: np.ndarray, # *** NEW ***
        tokenizer,
        max_length: int
    ):
        self.urls = urls
        self.labels = labels
        self.tabular_features = tabular_features
        self.location_ids = location_ids # *** NEW ***
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        # Tokenize URL
        encoding = self.tokenizer(
            self.urls[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'tabular_features': torch.tensor(self.tabular_features[idx], dtype=torch.float32),
            'location_id': torch.tensor(self.location_ids[idx], dtype=torch.long), # *** NEW ***
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==================== Focal Loss ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for multiclass classification.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== Two-Tower Model ====================

class TwoTowerURLClassifier(nn.Module):
    """
    *** UPDATED: Two-Tower architecture with Location Embedding ***
    - Tower 1: Fine-tunable URLBERT (Transformer)
    - Tower 2: Tabular features MLP + Location Embedding
    - Fusion: Concatenation + Classification head
    """
    
    def __init__(
        self,
        config: E2EConfig,
        n_classes: int,
        tabular_input_dim: int,
        n_locations: int # *** NEW ***
    ):
        super().__init__()
        self.config = config
        
        # Tower 1: URLBERT (Transformer)
        self.urlbert = AutoModel.from_pretrained(config.urlbert_model_name)
        self.urlbert_config = self.urlbert.config
        self.transformer_dim = self.urlbert_config.hidden_size
        
        # Tower 2: Tabular features
        # Part A: MLP for continuous features
        self.tabular_tower = nn.Sequential(
            nn.Linear(tabular_input_dim, config.tabular_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.tabular_dropout),
            nn.Linear(config.tabular_hidden_dim, config.tabular_output_dim),
            nn.ReLU(),
            nn.Dropout(config.tabular_dropout)
        )
        
        # Part B: Embedding for discrete 'location' feature
        self.location_embedding = nn.Embedding(
            n_locations, 
            config.location_embed_dim
        )
        
        # Fusion and Classification Head
        # fusion_dim now includes location_embed_dim 
        fusion_dim = (
            self.transformer_dim + 
            config.tabular_output_dim + 
            config.location_embed_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.fusion_dropout),
            nn.Linear(fusion_dim, n_classes)
        )
    
    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Helper function for Attention-Masked Mean Pooling"""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask

    def forward(
        self,
        input_ids,
        attention_mask,
        tabular_features,
        location_id
    ):
        # Tower 1: URLBERT embeddings (Mean Pooling)
        transformer_outputs = self.urlbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        transformer_embedding = self._mean_pooling(
            transformer_outputs.last_hidden_state,
            attention_mask
        )
        
        # Tower 2: Tabular embeddings
        # Part A: MLP output
        tabular_mlp_embedding = self.tabular_tower(tabular_features)
        # Part B: Location output
        location_embed = self.location_embedding(location_id)
        
        # Combine Tower 2 outputs 
        tabular_embedding = torch.cat([tabular_mlp_embedding, location_embed], dim=1)
        
        # Fusion: Concatenate Tower 1 and Tower 2
        fused = torch.cat([transformer_embedding, tabular_embedding], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


# ==================== Trainer ====================

class E2ETrainer:
    """End-to-end trainer with fine-tuning"""
    
    def __init__(self, config: Optional[E2EConfig] = None):
        self.config = config or E2EConfig()
        self.fe = FeatureExtractor(self.config)
        self.le = LabelEncoder() # For labels
        self.model = None
        self.tokenizer = None
        self.device = torch.device(
            self.config.urlbert_device if self.config.urlbert_device
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    def train(
        self,
        json_file: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train with K-fold CV"""
        if self.config.verbose > 0:
            logger.info("="*60)
            logger.info("Training Two-Tower E2E Classifier (with Anchor/Location)")
            logger.info("="*60)
            logger.info(f"Device: {self.device}")
        
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
        
        # Extract tabular features
        if self.config.verbose > 0:
            logger.info("\nExtracting tabular features (Structural, Interaction, TF-IDF)...")
        
        # Fit FeatureExtractor and get both feature sets 
        X_tabular, location_ids = self.fe.extract_all_tabular_features(
            urls, anchors, locations, fit=True
        )
        tabular_dim = X_tabular.shape[1]
        n_locations = len(self.fe.le_location.classes_)
        
        if self.config.verbose > 0:
            logger.info(f"Tabular (MLP) feature dimension: {tabular_dim}")
            logger.info(f"Location (Embedding) classes: {n_locations}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.urlbert_model_name,
            use_fast=False
        )
        
        # K-fold CV
        domains = [urlparse(u).netloc for u in urls]
        gkf = GroupKFold(n_splits=self.config.n_folds)
        
        fold_results = []
        
        for fold, (idx_tr, idx_val) in enumerate(gkf.split(urls, y, groups=domains), 1):
            if self.config.verbose > 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fold {fold}/{self.config.n_folds}")
                logger.info(f"{'='*60}")
            
            # Split data
            urls_train = [urls[i] for i in idx_tr]
            urls_val = [urls[i] for i in idx_val]
            y_train, y_val = y[idx_tr], y[idx_val]
            X_tab_train, X_tab_val = X_tabular[idx_tr], X_tabular[idx_val]
            loc_ids_train, loc_ids_val = location_ids[idx_tr], location_ids[idx_val]
            
            # Create datasets
            train_dataset = URLDataset(
                urls_train, y_train, X_tab_train, loc_ids_train,
                self.tokenizer, self.config.urlbert_max_length
            )
            val_dataset = URLDataset(
                urls_val, y_val, X_tab_val, loc_ids_val,
                self.tokenizer, self.config.urlbert_max_length
            )
            
            # Train model for this fold
            fold_acc, fold_f1 = self._train_fold(
                train_dataset, val_dataset, n_classes, tabular_dim, n_locations
            )
            
            fold_results.append({'fold': fold, 'accuracy': fold_acc, 'f1': fold_f1})
        
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
        
        # Train final model on full data
        if self.config.verbose > 0:
            logger.info("\nTraining final model on full dataset...")
        
        full_dataset = URLDataset(
            urls, y, X_tabular, location_ids,
            self.tokenizer, self.config.urlbert_max_length
        )
        self.model = self._train_final_model(
            full_dataset, n_classes, tabular_dim, n_locations
        )
        
        # Save
        if save_path:
            self.save_model(save_path)
        
        if self.config.verbose > 0:
            logger.info("\nTraining completed!")
        
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
    
    def _train_fold(
        self,
        train_dataset: URLDataset,
        val_dataset: URLDataset,
        n_classes: int,
        tabular_dim: int,
        n_locations: int
    ) -> Tuple[float, float]:
        """Train model for one fold"""
        # Create model
        model = TwoTowerURLClassifier(
            self.config, n_classes, tabular_dim, n_locations
        ).to(self.device)
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Optimizer with differential learning rates
        optimizer = AdamW([
            {'params': model.urlbert.parameters(), 'lr': self.config.learning_rate},
            # Add location_embedding to tabular LR group 
            {'params': model.tabular_tower.parameters(), 'lr': self.config.tabular_learning_rate},
            {'params': model.location_embedding.parameters(), 'lr': self.config.tabular_learning_rate},
            {'params': model.classifier.parameters(), 'lr': self.config.tabular_learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = len(train_loader) * self.config.warmup_epochs
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            # Add LR for location_embedding 
            max_lr=[
                self.config.learning_rate, 
                self.config.tabular_learning_rate, 
                self.config.tabular_learning_rate, 
                self.config.tabular_learning_rate
            ],
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos'
        )
        
        # Loss function
        criterion: nn.Module
        if self.config.use_focal_loss:
            if self.config.verbose > 0:
                logger.info(f"  Using Focal Loss (gamma={self.config.focal_loss_gamma})")
            criterion = FocalLoss(gamma=self.config.focal_loss_gamma)
        else:
            if self.config.verbose > 0:
                logger.info("  Using Cross Entropy Loss")
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                tabular_features = batch['tabular_features'].to(self.device)
                location_id = batch['location_id'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                logits = model(input_ids, attention_mask, tabular_features, location_id)
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validate
            val_acc, val_f1 = self._evaluate(model, val_loader)
            
            if self.config.verbose > 1:
                avg_train_loss = train_loss / len(train_loader)
                logger.info(
                    f"  Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Loss: {avg_train_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}, "
                    f"Val F1: {val_f1:.4f}"
                )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        # Final evaluation
        final_acc, final_f1 = self._evaluate(model, val_loader)
        
        if self.config.verbose > 0:
            logger.info(f"  Final - Acc: {final_acc:.4f}, F1: {final_f1:.4f}")
        
        return final_acc, final_f1
    
    def _train_final_model(
        self,
        full_dataset: URLDataset,
        n_classes: int,
        tabular_dim: int,
        n_locations: int
    ) -> TwoTowerURLClassifier:
        """Train final model on full dataset"""
        model = TwoTowerURLClassifier(
            self.config, n_classes, tabular_dim, n_locations
        ).to(self.device)
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        optimizer = AdamW([
            {'params': model.urlbert.parameters(), 'lr': self.config.learning_rate},
            {'params': model.tabular_tower.parameters(), 'lr': self.config.tabular_learning_rate},
            {'params': model.location_embedding.parameters(), 'lr': self.config.tabular_learning_rate},
            {'params': model.classifier.parameters(), 'lr': self.config.tabular_learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = len(train_loader) * self.config.warmup_epochs
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                self.config.learning_rate, 
                self.config.tabular_learning_rate, 
                self.config.tabular_learning_rate, 
                self.config.tabular_learning_rate
            ],
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos'
        )
        
        criterion: nn.Module
        if self.config.use_focal_loss:
            if self.config.verbose > 0:
                logger.info(f"  Using Focal Loss (gamma={self.config.focal_loss_gamma})")
            criterion = FocalLoss(gamma=self.config.focal_loss_gamma)
        else:
            if self.config.verbose > 0:
                logger.info("  Using Cross Entropy Loss")
            criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(self.config.num_epochs):
            if self.config.verbose > 1 and epoch % 5 == 0:
                 logger.info(f"  Final training epoch {epoch+1}/{self.config.num_epochs}")
                 
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                tabular_features = batch['tabular_features'].to(self.device)
                location_id = batch['location_id'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, tabular_features, location_id)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()
                scheduler.step()
        
        return model
    
    def _evaluate(
        self,
        model: TwoTowerURLClassifier,
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        """Evaluate model"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                tabular_features = batch['tabular_features'].to(self.device)
                location_id = batch['location_id'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = model(input_ids, attention_mask, tabular_features, location_id)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return acc, f1
    
    def save_model(self, filepath: str) -> None:
        """Save model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'label_encoder': self.le,
            'feature_extractor': self.fe,
            'n_classes': len(self.le.classes_),
            'tabular_dim': self.fe.scaler.n_features_in_,
            'n_locations': len(self.fe.le_location.classes_)
        }, filepath)
        
        if self.config.verbose > 0:
            file_size = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Model saved: {filepath.name} ({file_size:.1f} MB)")
    
    @staticmethod
    def load_model(filepath: str, device: Optional[str] = None):
        """Load model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        config = E2EConfig(**checkpoint['config'])
        trainer = E2ETrainer(config)
        
        if device:
            trainer.device = torch.device(device)
        
        trainer.le = checkpoint['label_encoder']
        trainer.fe = checkpoint['feature_extractor']
        
        # Recreate model
        trainer.model = TwoTowerURLClassifier(
            config,
            checkpoint['n_classes'],
            checkpoint['tabular_dim'],
            checkpoint['n_locations']
        )
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.to(trainer.device)
        trainer.model.eval()
        
        # Load tokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(
            config.urlbert_model_name,
            use_fast=False
        )
        
        return trainer

# ==================== Main ====================

def main():
    """Example usage"""
    
    # Configuration
    config = E2EConfig(
        urlbert_model_name="CrabInHoney/urlbert-tiny-base-v4",
        use_tfidf=True,
        tfidf_dim=256,
        location_embed_dim=16,
        tabular_hidden_dim=128,
        tabular_output_dim=64,
        batch_size=32,
        learning_rate=2e-5,
        tabular_learning_rate=1e-3,
        num_epochs=10,
        n_folds=5,
        use_focal_loss=True,
        focal_loss_gamma=2.0,
        verbose=1
    )
    
    # Train
    trainer = E2ETrainer(config)
    results = trainer.train(
        json_file="training_data/labeled_urls.json",
        save_path="urlbert_e2e_model.pth"
    )
    
    print(f"\n✓ Accuracy: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"✓ F1 Score: {results['avg_f1']:.4f} ± {results['std_f1']:.4f}")

if __name__ == "__main__":
    main()