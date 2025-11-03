# -*- coding: utf-8 -*-
"""
URL收集系统
"""
import json
import time
import random
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag
from pathlib import Path
from collections import deque
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re


class URLDataCollector:
    """核心类"""
    
    def __init__(self, data_dir='training_data', random_seed=None):
        base = Path(data_dir)
        base.mkdir(parents=True, exist_ok=True)
        self.data_file  = base / 'labeled_urls.json'
        self.cache_file = base / 'collected_urls.json'
        self.collected_data = []
        self.labeled_data   = []
        
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        self.load_labeled_data()
        self.load_cache()
    
    def load_labeled_data(self):
        """加载已标注数据"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.labeled_data = json.load(f)
            except Exception as e:
                print(f"加载标注数据失败: {e}")
                self.labeled_data = []
        else:
            self.labeled_data = []
    
    def load_cache(self):
        """加载采集缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.collected_data = json.load(f)
            except Exception as e:
                print(f"加载缓存失败: {e}")
                self.collected_data = []
        else:
            self.collected_data = []
    
    def get_statistics(self):
        """统计信息"""
        stats = {
            'total_labeled': len(self.labeled_data),
            'labels': {}
        }
        for item in self.labeled_data:
            label = item.get('label', 'unknown')
            stats['labels'][label] = stats['labels'].get(label, 0) + 1
        return stats

    def save_labeled_data(self):
        """保存标注数据"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.labeled_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存标注数据失败: {e}")
    
    def save_cache(self):
        """保存采集缓存"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def save_statistics(self, stats_file='stats.json'):
        """保存统计信息"""
        try:
            stats = self.get_statistics()
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存统计失败: {e}")

    def _is_irrelevant_url(self, url):
        """过滤常见无价值URL"""
        path = urlparse(url).path.lower()
        
        blacklist = [
            '/cart', '/sso', '/oauth', '/api', '/graphql', '/wp-json', '/wp-admin',
            '/sitemap', '/feed', '/xmlrpc', '/basket', '/panier', '/commande'
        ]
        for b in blacklist:
            if b in path:
                return True
        if re.search(r"\.(?:jpg|jpeg|png|gif|svg|ico|pdf|zip|rar|7z|mp4|webm|woff2?)$", path):
            return True
        return False

    def _pick_anchor_text(self, a):
        """优先 <a> 文本；退回 title/aria-label；再退回 <img alt>"""
        def _norm(s):
            import re
            return re.sub(r"\s+", " ", (s or "")).strip()
        t = _norm(a.get_text(" ", strip=True))
        if t:
            return t
        for attr in ("aria-label", "title"):
            if a.has_attr(attr):
                tt = _norm(a.get(attr))
                if tt:
                    return tt
        img = a.find("img", alt=True)
        if img:
            tt = _norm(img.get("alt"))
            if tt:
                return tt
        return ""
    
    def _classify_location(self, a):
        HEADER_HINTS = (
            "header","topbar","navbar","nav","menu","site-header","masthead","menubar",
            "main-nav","primary-nav","global-nav","mega-menu","supernav","top-nav"
        )
        FOOTER_HINTS = ("footer","site-footer","bottom","colophon","footnote","credits")
        for parent in a.parents:
            name = getattr(parent, "name", "").lower()
            if name in ("header","nav"):
                return "header"
            if name == "footer":
                return "footer"
            ident = " ".join([
                (parent.get("id") or "").lower(),
                " ".join([c.lower() for c in (parent.get("class") or [])])
            ]).strip()
            if any(h in ident for h in HEADER_HINTS):
                return "header"
            if any(h in ident for h in FOOTER_HINTS):
                return "footer"
        return "body"

    def _guess_url_type(self, url):
        """简单猜测URL类型（用于弱标签/置信度）"""
        path = urlparse(url).path.lower()
        if re.search(r"/(blog|news|press|stories|journal)/", path):
            return 'article'
        if re.search(r"/(category|categories|list|archive|tag)/", path):
            return 'listing'
        if re.search(r"/(product|products|shop|item)/", path):
            return 'product'
        return 'uncertain'

    def _guess_confidence(self, url):
        """返回一个[0,1]的置信度分数"""
        guess = self._guess_url_type(url)
        if guess == 'uncertain':
            return 0.3
        elif guess in ['article', 'listing']:
            return 0.6
        elif guess == 'product':
            return 0.5
        return 0.3

    def _build_driver(self, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1200,900")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(25)
        return driver
    
    def collect_urls_from_website(self, url, max_urls=50, headless=True, max_depth=3, urls_per_depth=10):
        """深度采集 + 每层限制 + 最终随机采样"""
        collected = {
            'site': url,
            'domain': urlparse(url).netloc,
            'collected_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'urls': []
        }
        
        driver = None
        try:
            driver = self._build_driver(headless=headless)
            print(f"\n正在采集: {url}")
            
            base_domain = urlparse(url).netloc
            all_found_urls = {}
            
            current_layer = [url]
            visited = set()
            
            for depth in range(max_depth):
                if not current_layer:
                    break
                
                print(f"  深度 {depth}: 处理 {len(current_layer)} 个URL")
                next_layer_candidates = []
                
                for current_url in current_layer:
                    if current_url in visited:
                        continue
                    visited.add(current_url)
                    
                    try:
                        driver.get(current_url)
                        time.sleep(1)
                        
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        
                        for a in soup.find_all('a', href=True):
                            href = a.get('href', '').strip()
                            if href.startswith('#'):
                                continue
                                
                            full_url = urljoin(current_url, href)
                            full_url, _frag = urldefrag(full_url)
                            
                            if (urlparse(full_url).netloc != base_domain or 
                                self._is_irrelevant_url(full_url) or 
                                full_url in all_found_urls):
                                continue
                            
                            all_found_urls[full_url] = {
                                'anchor': self._pick_anchor_text(a),
                                'location': self._classify_location(a),
                                'depth': depth + 1,
                                'source': current_url
                            }
                            
                            if depth + 1 < max_depth:
                                next_layer_candidates.append(full_url)
                                
                    except Exception as e:
                        print(f"    访问失败 {current_url}: {e}")
                        continue
                
                if next_layer_candidates and depth + 1 < max_depth:
                    if self.random_seed is not None:
                        random.seed(self.random_seed + depth)
                    
                    if len(next_layer_candidates) > urls_per_depth:
                        current_layer = random.sample(next_layer_candidates, urls_per_depth)
                    else:
                        current_layer = next_layer_candidates
                    print(f"    发现 {len(next_layer_candidates)} 个新URL，随机选择 {len(current_layer)} 个进入下一层")
                else:
                    current_layer = []
            
            if self.random_seed is not None:
                random.seed(self.random_seed)
            
            all_urls_list = list(all_found_urls.items())
            if len(all_urls_list) > max_urls:
                sampled_urls = random.sample(all_urls_list, max_urls)
            else:
                sampled_urls = all_urls_list
            
            for u, info in sampled_urls:
                item = {
                    'url': u,
                    'anchor': info.get('anchor', ''),
                    'location': info.get('location', 'body'),
                    'depth': info.get('depth', 0),
                    'domain': base_domain,
                    'guess': self._guess_url_type(u),
                    'confidence': self._guess_confidence(u),
                    'source': info.get('source', url)
                }
                collected['urls'].append(item)
            
            self.collected_data.append(collected)
            self.save_cache()
            
            print(f"采集完成: 总共找到 {len(all_found_urls)} 个URL，随机采样 {len(collected['urls'])} 条")
            return collected
            
        except Exception as e:
            print(f"采集失败: {e}")
            return collected
        finally:
            if driver:
                driver.quit()

    def collect_from_multiple_sites(self, site_list, urls_per_site=30, depth=3, urls_per_depth=10):
        print(f"\n开始批量采集 {len(site_list)} 个网站...")
        print("=" * 80)
        
        all_collected = []
        for i, site in enumerate(site_list, 1):
            if not site.startswith('http'):
                site = 'https://' + site
            print(f"\n[{i}/{len(site_list)}] 处理: {site}")
            try:
                result = self.collect_urls_from_website(
                    site, 
                    max_urls=urls_per_site, 
                    headless=True, 
                    max_depth=depth,
                    urls_per_depth=urls_per_depth
                )
                all_collected.append(result)
                time.sleep(random.uniform(0.8, 1.6))
            except Exception as e:
                print(f"  跳过 {site}: {e}")
                continue
        
        total_urls = sum(len(r['urls']) for r in all_collected)
        print(f"  网站数: {len(all_collected)}")
        print(f"  URL总数: {total_urls}")
        print(f"  缓存位置: {self.cache_file}")
        
        return all_collected


def main():
    collector = URLDataCollector()
    print("1) 从多个网站批量采集")
    print("2) 查看统计信息")
    print("3) 导出已标注数据")
    print("q) 退出")
    
    choice = input("选择操作: ").strip().lower()
    if choice == '1':
        print("粘贴网站首页URL（含 https://），输入空行结束：")
        sites = []
        while True:
            try:
                site = input().strip()
            except EOFError:
                break
            if not site:
                break
            sites.append(site)
        if not sites:
            print("未输入网站")
            return
        
        urls_per_site = input("每个网站抽样保留数量 max_urls (默认30): ").strip()
        urls_per_site = int(urls_per_site) if urls_per_site else 30
        
        depth = input("深度 depth (默认3): ").strip()
        depth = int(depth) if depth else 3
        
        urls_per_depth = input("每层选择URL数量 urls_per_depth (默认10): ").strip()
        urls_per_depth = int(urls_per_depth) if urls_per_depth else 10
        
        seed = input("随机种子 seed (可选，回车跳过): ").strip()
        if seed:
            collector.random_seed = int(seed)
            random.seed(collector.random_seed)
        
        collector.collect_from_multiple_sites(sites, urls_per_site, depth=depth, urls_per_depth=urls_per_depth)
        return
    
    elif choice == '2':
        stats = collector.get_statistics()
        print("\n统计信息：")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    
    elif choice == '3':
        collector.save_labeled_data()
        print("已导出已标注数据。")
        return
    
    elif choice == 'q':
        print("退出。")
        return
    
    else:
        print("无效选项。")
        return

if __name__ == '__main__':
    main()