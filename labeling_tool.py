# -*- coding: utf-8 -*-
"""
优化版多分类URL标注工具
10个实用分类：文章页、文章列表、产品详情、产品列表、品牌、社交、法律、账号、商务、无关
"""
from flask import Flask, render_template_string, request, jsonify
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

app = Flask(__name__)

DATA_DIR = Path('training_data')
DATA_DIR.mkdir(exist_ok=True)
LABELED_FILE = DATA_DIR / 'labeled_urls.json'
CACHE_FILE = DATA_DIR / 'collected_urls.json'

# ==================== 分类定义 ====================

CATEGORIES = {
    'article_page': {
        'name': '新闻/博客文章页',
        'color': '#28a745',
        'icon': '📄',
        'key': '1',
        'description': '单篇文章内容页（新闻、博客、深度报道）',
        'examples': [
            '/blog/2024/01/how-to-guide',
            '/news/company-announcement',
            '/article/industry-analysis'
        ]
    },
    'article_list': {
        'name': '文章列表页',
        'color': '#17a2b8',
        'icon': '📋',
        'key': '2',
        'description': '文章索引、博客归档、新闻列表',
        'examples': [
            '/blog/',
            '/news/archive',
            '/articles',
            '/category/technology'
        ]
    },
    'product_detail': {
        'name': '产品详情页',
        'color': '#fd7e14',
        'icon': '📦',
        'key': '3',
        'description': '单个产品的详细介绍页',
        'examples': [
            '/products/iphone-15-pro',
            '/services/consulting-detail',
            '/product/123'
        ]
    },
    'product_list': {
        'name': '产品列表页',
        'color': '#ffc107',
        'icon': '🛍️',
        'key': '4',
        'description': '产品目录、产品分类页、服务列表',
        'examples': [
            '/products/',
            '/products/category/electronics',
            '/services',
            '/shop'
        ]
    },
    'brand_info': {
        'name': '品牌信息页',
        'color': '#6f42c1',
        'icon': '🏢',
        'key': '5',
        'description': '关于我们、公司介绍、品牌故事、团队',
        'examples': [
            '/about',
            '/about-us',
            '/company',
            '/our-story',
            '/team'
        ]
    },
    'social_media': {
        'name': '社交媒体',
        'color': '#e83e8c',
        'icon': '🔗',
        'key': '6',
        'description': 'Facebook、Instagram、Twitter/X、LinkedIn等',
        'examples': [
            'facebook.com/brand',
            'twitter.com/brand',
            'instagram.com/brand'
        ]
    },
    
    'legal': {
        'name': '法律相关',
        'color': '#dc3545',
        'icon': '⚖️',
        'key': '7',
        'description': '隐私政策、使用条款、Cookie政策',
        'examples': [
            '/privacy',
            '/terms',
            '/cookie-policy',
            '/legal'
        ]
    },
    'account': {
        'name': '账号相关',
        'color': '#dc3545',
        'icon': '👤',
        'key': '8',
        'description': '登录、注册、个人中心、账户设置',
        'examples': [
            '/login',
            '/register',
            '/account',
            '/profile',
            '/my-account'
        ]
    },
    'commerce': {
        'name': '商务相关',
        'color': '#dc3545',
        'icon': '💳',
        'key': '9',
        'description': '购物车、结账、支付、订单流程',
        'examples': [
            '/cart',
            '/checkout',
            '/payment',
            '/order'
        ]
    },
    'irrelevant': {
        'name': '无关页面',
        'color': '#6c757d',
        'icon': '🚫',
        'key': '0',
        'description': '其他无关内容、错误页、API等',
        'examples': [
            '/404',
            '/api/',
            '/sitemap.xml',
            '/feed'
        ]
    },
    'skip': {
        'name': '跳过/不确定',
        'color': '#adb5bd',
        'icon': '⏭️',
        'key': 's',
        'description': '无法判断的URL',
        'examples': []
    }
}

def load_data():
    """加载数据"""
    labeled = []
    if LABELED_FILE.exists():
        with open(LABELED_FILE, 'r', encoding='utf-8') as f:
            labeled = json.load(f)

    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, list):
            cache = {}
            for i, site_data in enumerate(raw):
                key = site_data.get('site') or site_data.get('domain') or str(i)
                cache[key] = site_data
        elif isinstance(raw, dict):
            cache = raw
        else:
            cache = {}

    return labeled, cache


def save_labeled_data(data):
    """保存标注数据"""
    with open(LABELED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_unlabeled_urls(labeled_data, cache):
    """获取未标注的URL"""
    labeled_urls = {item['url'] for item in labeled_data}
    site_iter = cache.values() if isinstance(cache, dict) else cache

    unlabeled = []
    for site_data in site_iter:
        for url_info in site_data.get('urls', []):
            if url_info['url'] not in labeled_urls:
                unlabeled.append(url_info)

    return unlabeled

def _lookup_meta(cache, url):
    """从 cache 中根据 URL 查找 anchor/location（兼容 dict/list 结构）"""
    site_iter = cache.values() if isinstance(cache, dict) else cache
    for site_data in site_iter or []:
        for info in site_data.get('urls', []) or []:
            if info.get('url') == url:
                return {'anchor': info.get('anchor', ''), 'location': info.get('location', 'body')}
    return {'anchor': '', 'location': 'body'}


# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>URL多分类标注工具</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .header h1 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-top: 20px;
        }
        
        .stat-box {
            padding: 12px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 3px;
            font-size: 12px;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        
        .card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .url-display {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            word-break: break-all;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            border-left: 4px solid #667eea;
        }
        
        .hints {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #ffc107;
        }
        
        .hints h3 {
            color: #856404;
            margin-bottom: 10px;
            font-size: 14px;
        }
        
        .hints ul {
            list-style: none;
            padding-left: 0;
        }
        
        .hints li {
            color: #856404;
            margin: 5px 0;
            padding-left: 25px;
            position: relative;
            font-size: 13px;
        }
        
        .hints li:before {
            content: "💡";
            position: absolute;
            left: 0;
        }
        
        .category-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .category-btn {
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            background: white;
            text-align: left;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .category-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .category-btn.wanted {
            border-color: #28a745;
            background: #f0fff4;
        }
        
        .category-btn.unwanted {
            border-color: #dc3545;
            background: #fff5f5;
        }
        
        .category-icon {
            font-size: 24px;
        }
        
        .category-info {
            flex: 1;
        }
        
        .category-name {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 3px;
        }
        
        .category-key {
            font-size: 11px;
            color: #666;
            font-family: monospace;
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 3px;
        }
        
        .progress-bar {
            background: #e9ecef;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .sidebar {
            position: sticky;
            top: 20px;
            height: fit-content;
        }
        
        .help-section {
            background: #e7f3ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .help-section h3 {
            color: #004085;
            margin-bottom: 15px;
            font-size: 16px;
        }
        
        .help-category {
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #cce5ff;
        }
        
        .help-category:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        
        .help-title {
            font-weight: bold;
            color: #004085;
            margin-bottom: 4px;
            font-size: 13px;
        }
        
        .help-desc {
            font-size: 12px;
            color: #004085;
            margin-bottom: 4px;
        }
        
        .help-examples {
            font-size: 11px;
            color: #6c757d;
            font-style: italic;
        }
        
        .action-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .action-btn {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .btn-undo {
            background: #ffc107;
            color: white;
        }
        
        .btn-skip {
            background: #6c757d;
            color: white;
        }
        
        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .card {
            animation: fadeIn 0.3s ease-out;
        }
        
        @media (max-width: 968px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            .sidebar {
                position: static;
            }
            .category-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏷️ URL分类标注工具</h1>
            <p>优化版 - 10个实用分类</p>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-number" id="total-labeled">0</div>
                    <div class="stat-label">已标注</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="article-page-count">0</div>
                    <div class="stat-label">📄 文章页</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="article-list-count">0</div>
                    <div class="stat-label">📋 文章列表</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="product-detail-count">0</div>
                    <div class="stat-label">📦 产品详情</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="product-list-count">0</div>
                    <div class="stat-label">🛍️ 产品列表</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="brand-count">0</div>
                    <div class="stat-label">🏢 品牌</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="remaining">0</div>
                    <div class="stat-label">待标注</div>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div>
                <div class="card" id="labeling-card">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress" style="width: 0%">
                            <span id="progress-text">0%</span>
                        </div>
                    </div>
                    
                    <div id="url-section">
                        <h2 style="margin-bottom: 20px; color: #333;">当前URL</h2>
                        <div class="url-display" id="current-url">加载中...</div>
                        
                        <div class="hints" id="hints-box" style="display: none;">
                            <h3>💡 智能识别提示</h3>
                            <ul id="hints-list"></ul>
                        </div>
                        
                        <h3 style="margin-bottom: 15px; color: #333;">选择分类</h3>
                        
                        <div class="category-grid">
                            <div class="category-btn wanted" onclick="label('article_page')">
                                <span class="category-icon">📄</span>
                                <div class="category-info">
                                    <div class="category-name">文章页</div>
                                    <span class="category-key">按键: 1</span>
                                </div>
                            </div>
                            
                            <div class="category-btn wanted" onclick="label('article_list')">
                                <span class="category-icon">📋</span>
                                <div class="category-info">
                                    <div class="category-name">文章列表</div>
                                    <span class="category-key">按键: 2</span>
                                </div>
                            </div>
                            
                            <div class="category-btn wanted" onclick="label('product_detail')">
                                <span class="category-icon">📦</span>
                                <div class="category-info">
                                    <div class="category-name">产品详情</div>
                                    <span class="category-key">按键: 3</span>
                                </div>
                            </div>
                            
                            <div class="category-btn wanted" onclick="label('product_list')">
                                <span class="category-icon">🛍️</span>
                                <div class="category-info">
                                    <div class="category-name">产品列表</div>
                                    <span class="category-key">按键: 4</span>
                                </div>
                            </div>
                            
                            <div class="category-btn wanted" onclick="label('brand_info')">
                                <span class="category-icon">🏢</span>
                                <div class="category-info">
                                    <div class="category-name">品牌信息</div>
                                    <span class="category-key">按键: 5</span>
                                </div>
                            </div>
                            
                            <div class="category-btn wanted" onclick="label('social_media')">
                                <span class="category-icon">🔗</span>
                                <div class="category-info">
                                    <div class="category-name">社交媒体</div>
                                    <span class="category-key">按键: 6</span>
                                </div>
                            </div>
                            
                            <div class="category-btn unwanted" onclick="label('legal')">
                                <span class="category-icon">⚖️</span>
                                <div class="category-info">
                                    <div class="category-name">法律相关</div>
                                    <span class="category-key">按键: 7</span>
                                </div>
                            </div>
                            
                            <div class="category-btn unwanted" onclick="label('account')">
                                <span class="category-icon">👤</span>
                                <div class="category-info">
                                    <div class="category-name">账号相关</div>
                                    <span class="category-key">按键: 8</span>
                                </div>
                            </div>
                            
                            <div class="category-btn unwanted" onclick="label('commerce')">
                                <span class="category-icon">💳</span>
                                <div class="category-info">
                                    <div class="category-name">商务相关</div>
                                    <span class="category-key">按键: 9</span>
                                </div>
                            </div>
                            
                            <div class="category-btn unwanted" onclick="label('irrelevant')">
                                <span class="category-icon">🚫</span>
                                <div class="category-info">
                                    <div class="category-name">无关页面</div>
                                    <span class="category-key">按键: 0</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="action-buttons">
                            <button class="action-btn btn-undo" onclick="undo()">
                                ↶ 撤销 (U)
                            </button>
                            <button class="action-btn btn-skip" onclick="label('skip')">
                                ⏭️ 跳过 (S)
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="help-section">
                    <h3>📖 分类指南</h3>
                    
                    <div class="help-category">
                        <div class="help-title">📄 文章页 ✅</div>
                        <div class="help-desc">单篇文章：新闻、博客、深度报道</div>
                        <div class="help-examples">例: /blog/2024/how-to</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title">📋 文章列表 ✅</div>
                        <div class="help-desc">多篇文章索引：博客首页、归档</div>
                        <div class="help-examples">例: /blog/, /news/</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title">📦 产品详情 ✅</div>
                        <div class="help-desc">单个产品的详细页面</div>
                        <div class="help-examples">例: /products/iphone-15</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title">🛍️ 产品列表 ✅</div>
                        <div class="help-desc">产品目录、分类页</div>
                        <div class="help-examples">例: /products/, /shop/</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title">🏢 品牌信息 ✅</div>
                        <div class="help-desc">关于我们、公司介绍</div>
                        <div class="help-examples">例: /about, /company</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title">🔗 社交媒体 ✅</div>
                        <div class="help-desc">社交网络链接</div>
                        <div class="help-examples">例: facebook.com/brand</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title" style="color: #dc3545;">⚖️ 法律相关 ❌</div>
                        <div class="help-desc">隐私、条款、Cookie</div>
                        <div class="help-examples">例: /privacy, /terms</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title" style="color: #dc3545;">👤 账号相关 ❌</div>
                        <div class="help-desc">登录、注册</div>
                        <div class="help-examples">例: /login, /register</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title" style="color: #dc3545;">💳 商务相关 ❌</div>
                        <div class="help-desc">购物车、支付</div>
                        <div class="help-examples">例: /cart, /checkout</div>
                    </div>
                    
                    <div class="help-category">
                        <div class="help-title" style="color: #dc3545;">🚫 无关页面 ❌</div>
                        <div class="help-desc">其他无关内容</div>
                        <div class="help-examples">例: /404, /api/</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentUrl = null;
        let currentIndex = 0;
        let unlabeledUrls = [];
        let stats = {};
        
        async function loadData() {
            const response = await fetch('/api/data');
            const data = await response.json();
            unlabeledUrls = data.unlabeled;
            stats = data.stats;
            
            updateStats();
            loadNextUrl();
        }
        
        function loadNextUrl() {
            if (currentIndex >= unlabeledUrls.length) {
                showDoneMessage();
                return;
            }
            
            const urlInfo = unlabeledUrls[currentIndex];
            currentUrl = urlInfo.url;
            
            document.getElementById('current-url').textContent = currentUrl;
            showHints(urlInfo);
            
            const progress = ((currentIndex + 1) / unlabeledUrls.length) * 100;
            document.getElementById('progress').style.width = progress + '%';
            document.getElementById('progress-text').textContent = Math.round(progress) + '%';
        }
        
        function showHints(urlInfo) {
            const hints = [];
            const url = urlInfo.url.toLowerCase();
            const path = new URL(urlInfo.url).pathname;
            
            // 社交媒体
            if (/(facebook|twitter|instagram|linkedin|youtube)\.com/i.test(url)) {
                hints.push('🔗 社交媒体域名 → 标记为"社交媒体"(6)');
            }
            
            // 法律相关
            if (/(privacy|terms|cookie|legal|gdpr)/i.test(url)) {
                hints.push('⚖️ 法律关键词 → "法律相关"(7)');
            }
            
            // 账号相关
            if (/(login|register|signin|signup|account|profile)/i.test(url)) {
                hints.push('👤 账号关键词 → "账号相关"(8)');
            }
            
            // 商务相关
            if (/(cart|checkout|payment|buy|order)/i.test(url)) {
                hints.push('💳 商务关键词 → "商务相关"(9)');
            }
            
            // 品牌信息
            if (/(about|company|mission|story|team)/i.test(url)) {
                hints.push('🏢 品牌关键词 → 可能是"品牌信息"(5)');
            }
            
            // 产品相关
            if (/(product|service)/i.test(url)) {
                const depth = path.split('/').filter(s => s).length;
                if (depth >= 3 || /\/\d+/.test(path)) {
                    hints.push('📦 产品+深路径 → 可能是"产品详情"(3)');
                } else if (path.endsWith('/')) {
                    hints.push('🛍️ 产品+浅路径 → 可能是"产品列表"(4)');
                }
            }
            
            // 文章相关
            if (/(blog|news|article|post)/i.test(url)) {
                const hasDate = /\d{4}/.test(path);
                const depth = path.split('/').filter(s => s).length;
                
                if (hasDate && depth >= 4) {
                    hints.push('📄 含日期+深路径 → 可能是"文章页"(1)');
                } else if (path.endsWith('/') || depth <= 2) {
                    hints.push('📋 浅路径或以/结尾 → 可能是"文章列表"(2)');
                }
            }
            
            // 列表页特征
            if (/(archive|category|tag)/.test(url) || path.endsWith('/')) {
                hints.push('📋 列表页特征 → 检查是文章还是产品列表');
            }
            
            if (hints.length > 0) {
                document.getElementById('hints-box').style.display = 'block';
                document.getElementById('hints-list').innerHTML = hints.map(h => `<li>${h}</li>`).join('');
            } else {
                document.getElementById('hints-box').style.display = 'none';
            }
        }
        
        async function label(type) {
            if (!currentUrl) return;
            
            if (type !== 'skip') {
                const response = await fetch('/api/label', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url: currentUrl, label: type})
                });
                
                const result = await response.json();
                stats = result.stats;
                updateStats();
            }
            
            currentIndex++;
            loadNextUrl();
        }
        
        function updateStats() {
            document.getElementById('total-labeled').textContent = stats.total || 0;
            document.getElementById('article-page-count').textContent = stats.article_page || 0;
            document.getElementById('article-list-count').textContent = stats.article_list || 0;
            document.getElementById('product-detail-count').textContent = stats.product_detail || 0;
            document.getElementById('product-list-count').textContent = stats.product_list || 0;
            document.getElementById('brand-count').textContent = stats.brand_info || 0;
            document.getElementById('remaining').textContent = unlabeledUrls.length - currentIndex;
        }
        
        function showDoneMessage() {
            document.getElementById('url-section').innerHTML = `
                <div style="text-align: center; padding: 40px;">
                    <h2 style="color: #28a745; margin-bottom: 20px;">🎉 完成！</h2>
                    <p style="font-size: 18px; color: #666;">
                        您已完成所有URL的标注！<br>
                        共标注 <strong style="color: #667eea;">${stats.total}</strong> 条数据
                    </p>
                </div>
            `;
        }
        
        async function undo() {
            const response = await fetch('/api/undo', {method: 'POST'});
            const result = await response.json();
            
            if (result.success) {
                stats = result.stats;
                updateStats();
                currentIndex = Math.max(0, currentIndex - 1);
                loadNextUrl();
            }
        }
        
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            
            const keyMap = {
                '1': 'article_page',
                '2': 'article_list',
                '3': 'product_detail',
                '4': 'product_list',
                '5': 'brand_info',
                '6': 'social_media',
                '7': 'legal',
                '8': 'account',
                '9': 'commerce',
                '0': 'irrelevant',
                's': 'skip'
            };
            
            const key = e.key.toLowerCase();
            if (keyMap[key]) {
                label(keyMap[key]);
            } else if (key === 'u') {
                undo();
            }
        });
        
        loadData();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    labeled, cache = load_data()
    unlabeled = get_unlabeled_urls(labeled, cache)
    
    label_counts = Counter(item['label'] for item in labeled)
    stats = {
        'total': len(labeled),
        **{k: label_counts.get(k, 0) for k in CATEGORIES.keys()}
    }
    
    return jsonify({'unlabeled': unlabeled, 'stats': stats})

@app.route('/api/label', methods=['POST'])
def add_label():
    data = request.json
    labeled, cache = load_data()

    meta = _lookup_meta(cache, data['url'])
    labeled.append({
        'url': data['url'],
        'label': data['label'],
        'anchor': meta.get('anchor', ''),
        'location': meta.get('location', 'body'),
        'labeled_at': datetime.now().isoformat()
    })

    
    save_labeled_data(labeled)
    
    label_counts = Counter(item['label'] for item in labeled)
    stats = {
        'total': len(labeled),
        **{k: label_counts.get(k, 0) for k in CATEGORIES.keys()}
    }
    
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/undo', methods=['POST'])
def undo_label():
    labeled, cache = load_data()
    
    if labeled:
        labeled.pop()
        save_labeled_data(labeled)
    
    label_counts = Counter(item['label'] for item in labeled)
    stats = {
        'total': len(labeled),
        **{k: label_counts.get(k, 0) for k in CATEGORIES.keys()}
    }
    
    return jsonify({'success': True, 'stats': stats})

if __name__ == '__main__':
    print("\n打开浏览器: http://127.0.0.1:5000")
    print("\n按 Ctrl+C 停止")
    
    app.run(host='127.0.0.1', port=5000, debug=False)