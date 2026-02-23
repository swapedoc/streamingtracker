# streaming_tracker_v3.py - Two-Flow System: Watch Now + Discover
# Run with: python3 streaming_tracker_v3.py

"""
STREAMING TRACKER V3.0 - TWO-FLOW SYSTEM

FLOWS:
1. WATCH NOW: Trending content with YouTube/Reddit reviews + scoring
2. DISCOVER: Classics/Genres/Gems with basic availability check (no reviews/scoring)

NEW DATABASE TABLE NEEDED:
Run this SQL in Supabase:

CREATE TABLE IF NOT EXISTS discover_content (
    id BIGSERIAL PRIMARY KEY,
    tmdb_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    original_title TEXT,
    platform TEXT NOT NULL,
    content_type TEXT NOT NULL,
    release_year INTEGER,
    imdb_rating FLOAT,
    poster_path TEXT,
    overview TEXT,
    category TEXT NOT NULL,
    genre TEXT,
    popularity FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tmdb_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_discover_category ON discover_content(category);
CREATE INDEX IF NOT EXISTS idx_discover_platform ON discover_content(platform);
CREATE INDEX IF NOT EXISTS idx_discover_genre ON discover_content(genre);
"""

import os
import re
import time
import math
import asyncio
import requests
import numpy as np
import feedparser
import urllib.parse
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from bs4 import BeautifulSoup

# ============================================================================
# PROGRESS UTILITIES
# ============================================================================

class Spinner:
    """Lightweight spinner for silent waits — shows we're alive"""
    def __init__(self, message="Working"):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
    
    def _spin(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r   {frames[i % len(frames)]} {self.message}...")
            sys.stdout.flush()
            i += 1
            time.sleep(0.15)
    
    def start(self):
        self._thread.start()
        return self
    
    def stop(self, final_msg=None):
        self._stop.set()
        self._thread.join()
        if final_msg:
            sys.stdout.write(f"\r   ✅ {final_msg}\n")
        else:
            sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

def progress(current, total, label=""):
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = current / total * 100
    sys.stdout.write(f"\r   [{bar}] {current}/{total} ({pct:.0f}%) {label[:30]}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    TMDB_API_KEY = os.getenv('TMDB_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    # Platform mapping: Name -> TMDb Provider ID for India
    PLATFORMS = {
        'Netflix': 8,
        'Prime Video': 119,
        'Apple TV+': 350,
        'Jiohotstar': 2336,
        'JioCinema': 220
    }
    
    # Genre IDs for TMDb
    GENRES = {
        'Action': 28,
        'Horror': 27,
        'Comedy': 35,
        'Drama': 18,
        'Thriller': 53,
        'Sci-Fi': 878,
        'Romance': 10749
    }
    
    # WATCH NOW FLOW (with reviews & scoring)
    WATCH_NOW_TRENDING_LIMIT = 20
    WATCH_NOW_MAX_VIDEOS_PER_PLATFORM = 10
    
    # DISCOVER FLOW (no reviews, just availability)
    DISCOVER_CLASSICS_LIMIT = 120
    DISCOVER_GENRE_LIMIT = 80
    DISCOVER_UNDERDOG_LIMIT = 60
    DISCOVER_ENABLED_GENRES = ['Action', 'Horror', 'Thriller', 'Comedy', 'Drama' ,'Romance', 'Sci-Fi']
    DISCOVER_INDIAN_LIMIT = 120
    
    # Indian languages for TMDb filtering (Hindi only)
    INDIAN_LANGUAGES = ['hi']
    INDIAN_LANGUAGE_NAMES = {
        'hi': 'Hindi'
    }
    
    USE_TRANSCRIPTS = False   # transcripts too slow; title+desc is fast & nearly as good
    USE_REDDIT = True          # on by default; disable with --no-reddit
    USE_CRITICS = True         # on by default; disable with --no-critics

# ============================================================================
# TMDB INTEGRATION
# ============================================================================

class TMDbResolver:
    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
        # Shared session with automatic retry on connection errors
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self._session = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=10))
    
    def get_trending(self, media_type='all', time_window='week', limit=20) -> List[Dict]:
        """Get trending content from TMDb"""
        url = f"{self.base_url}/trending/{media_type}/{time_window}"
        params = {'api_key': self.api_key}
        
        for attempt in range(4):
            try:
                response = self._session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                if attempt == 3:
                    print(f"❌ TMDb trending error: {e}")
                    return []
                wait_time = attempt + 1
                print(f"   ⚡️ Retrying in {wait_time}s...")
                time.sleep(wait_time)

        try:
            trending = []
            current_year = datetime.now().year
            WATCH_NOW_MAX_AGE = 3   # exclude titles older than 3 years from Watch Now

            for item in data.get('results', [])[:limit]:
                if item.get('media_type') not in ['movie', 'tv']:
                    continue

                media_type = item['media_type']
                title = item.get('title') if media_type == 'movie' else item.get('name')
                release_year = self._extract_year(
                    item.get('release_date') if media_type == 'movie' else item.get('first_air_date')
                )

                # Skip old titles — they belong in Discover, not Watch Now
                if release_year and (current_year - release_year) > WATCH_NOW_MAX_AGE:
                    continue

                trending.append({
                    'tmdb_id': item['id'],
                    'title': title,
                    'original_title': item.get('original_title') or item.get('original_name'),
                    'content_type': media_type,
                    'release_year': release_year,
                    'poster_path': item.get('poster_path'),
                    'overview': item.get('overview'),
                    'popularity': item.get('popularity', 0),
                    'imdb_rating': item.get('vote_average'),
                    'category': 'trending'
                })

            print(f"✅ Found {len(trending)} trending titles (within {WATCH_NOW_MAX_AGE} years)")
            return trending

        except Exception as e:
            print(f"❌ Error parsing TMDb data: {e}")
            return []
    
    def get_trending_indian(self, media_type='movie', limit=20) -> List[Dict]:
        """
        Get genuinely trending Hindi content.
        Step 1: Pull from /trending/week filtered to original_language=hi (truly trending this week).
        Step 2: Top up via /discover with a 2-year recency cap so old evergreen titles
                like PK, Dangal etc. never appear in the Watch Now list.
        """
        print(f"\n🇮🇳 Fetching trending Hindi {media_type} content...")

        all_results = []
        seen_ids = set()
        current_year = datetime.now().year

        # Step 1: genuine trending endpoint, filter Hindi
        for attempt in range(3):
            try:
                response = self._session.get(
                    f"{self.base_url}/trending/{media_type}/week",
                    params={"api_key": self.api_key},
                    timeout=15
                )
                response.raise_for_status()
                for item in response.json().get("results", []):
                    if item.get("original_language") != "hi":
                        continue
                    if item["id"] in seen_ids:
                        continue
                    seen_ids.add(item["id"])
                    title = item.get("title") if media_type == "movie" else item.get("name")
                    release_date = item.get("release_date") if media_type == "movie" else item.get("first_air_date")
                    all_results.append({
                        "tmdb_id": item["id"],
                        "title": title,
                        "original_title": item.get("original_title") or item.get("original_name"),
                        "content_type": media_type,
                        "release_year": self._extract_year(release_date),
                        "poster_path": item.get("poster_path"),
                        "overview": item.get("overview"),
                        "popularity": item.get("popularity", 0),
                        "imdb_rating": item.get("vote_average"),
                        "category": "trending",
                        "language": "hi",
                        "language_name": "Hindi",
                        "is_indian": True
                    })
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"   ⚡️ Trending endpoint failed: {e}")

        # Step 2: top up with /discover but only last 2 years, for BOTH media types
        if len(all_results) < limit:
            min_date = f"{current_year - 2}-01-01"
            for mt in ["movie", "tv"]:
                if len(all_results) >= limit:
                    break
                discover_params = {
                    "api_key": self.api_key,
                    "sort_by": "popularity.desc",
                    "with_original_language": "hi",
                    "vote_count.gte": 20,
                    "with_watch_providers": "8|119|350|2336|220",
                    "watch_region": "IN",
                    "page": 1
                }
                if mt == "movie":
                    discover_params["primary_release_date.gte"] = min_date
                else:
                    # air_date.gte catches series with recent seasons (not just first_air_date)
                    discover_params["air_date.gte"] = min_date
                try:
                    response = self._session.get(
                        f"{self.base_url}/discover/{mt}",
                        params=discover_params,
                        timeout=15
                    )
                    response.raise_for_status()
                    for item in response.json().get("results", []):
                        if item["id"] in seen_ids or len(all_results) >= limit:
                            break
                        seen_ids.add(item["id"])
                        title = item.get("title") if mt == "movie" else item.get("name")
                        release_date = item.get("release_date") if mt == "movie" else item.get("first_air_date")
                        all_results.append({
                            "tmdb_id": item["id"],
                            "title": title,
                            "original_title": item.get("original_title") or item.get("original_name"),
                            "content_type": mt,
                            "release_year": self._extract_year(release_date),
                            "poster_path": item.get("poster_path"),
                            "overview": item.get("overview"),
                            "popularity": item.get("popularity", 0),
                            "imdb_rating": item.get("vote_average"),
                            "category": "trending",
                            "language": "hi",
                            "language_name": "Hindi",
                            "is_indian": True
                        })
                except Exception as e:
                    print(f"   ⚡️ Discover fallback failed ({mt}): {e}")

        print(f"✅ Found {len(all_results)} trending Hindi {media_type} titles")
        return all_results[:limit]

    def get_watch_providers(self, tmdb_id: int, media_type: str, retries=3) -> List[int]:
        """Get streaming platforms where content is available in India"""
        for attempt in range(retries):
            try:
                response = self._session.get(
                    f"{self.base_url}/{media_type}/{tmdb_id}/watch/providers",
                    params={'api_key': self.api_key},
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                
                india_data = data.get('results', {}).get('IN', {})
                providers = india_data.get('flatrate', [])
                provider_ids = [p['provider_id'] for p in providers]
                
                return provider_ids
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    return []
        return []
    
    def _extract_year(self, date_str):
        if date_str:
            try:
                return int(date_str.split('-')[0])
            except:
                pass
        return None

# ============================================================================
# DISCOVER FLOW - NO REVIEWS, JUST AVAILABILITY
# ============================================================================

class DiscoverFlow:
    """
    Async-first discover flow.
    All TMDb page fetches fire concurrently via aiohttp, then provider checks
    run in parallel via ThreadPoolExecutor — total discover time ~5-15s vs 2-3min.
    """

    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.tmdb = TMDbResolver()

    # ── Async fetch helpers ────────────────────────────────────────────────

    def _build_jobs(self) -> List[Dict]:
        """
        Build the full list of (media_type, params, category, meta) fetch jobs
        without hitting the network. Each job = one TMDb /discover page request.
        """
        import math
        jobs = []
        PAGES_PER_TYPE = 4   # 4 pages × 20 results = 80 per media_type per category

        # ── Classics ──────────────────────────────────────────────────────
        for mt in ['movie', 'tv']:
            base = {
                'api_key': self.api_key,
                'sort_by': 'vote_average.desc',
                'vote_average.gte': 8.0,
                'vote_count.gte': 500,
                'with_watch_providers': '8|119|350|2336|220',
                'watch_region': 'IN',
            }
            for page in range(1, PAGES_PER_TYPE + 1):
                jobs.append({'mt': mt, 'params': {**base, 'page': page},
                             'category': 'classics', 'genre': None})

        # ── Genres ────────────────────────────────────────────────────────
        for genre_name in Config.DISCOVER_ENABLED_GENRES:
            genre_id = Config.GENRES.get(genre_name)
            if not genre_id:
                continue
            for mt in ['movie', 'tv']:
                base = {
                    'api_key': self.api_key,
                    'with_genres': genre_id,
                    'sort_by': 'popularity.desc',
                    'vote_average.gte': 6.5,
                    'vote_count.gte': 100,
                    'with_watch_providers': '8|119|350|2336|220',
                    'watch_region': 'IN',
                }
                for page in range(1, PAGES_PER_TYPE + 1):
                    jobs.append({'mt': mt, 'params': {**base, 'page': page},
                                 'category': f'genre_{genre_name.lower()}',
                                 'genre': genre_name})

        # ── Hidden Gems ───────────────────────────────────────────────────
        six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        for mt in ['movie', 'tv']:
            date_key = 'primary_release_date.gte' if mt == 'movie' else 'first_air_date.gte'
            base = {
                'api_key': self.api_key,
                'region': 'IN',
                'sort_by': 'vote_average.desc',
                'vote_count.gte': 50,
                'vote_count.lte': 5000,
                'vote_average.gte': 7.0,
                date_key: six_months_ago,
                'with_watch_providers': '8|119|350|2336|220',
                'watch_region': 'IN',
            }
            for page in range(1, 3):   # 2 pages × 2 types = 60 gems max
                jobs.append({'mt': mt, 'params': {**base, 'page': page},
                             'category': 'underdog', 'genre': None})

        # ── Indian / Hindi ─────────────────────────────────────────────────
        min_date = f"{datetime.now().year - 3}-01-01"
        for mt in ['movie', 'tv']:
            base = {
                'api_key': self.api_key,
                'with_original_language': 'hi',
                'sort_by': 'popularity.desc',
                'vote_count.gte': 20,
                'with_watch_providers': '8|119|350|2336|220',
                'watch_region': 'IN',
            }
            if mt == 'movie':
                base['primary_release_date.gte'] = min_date
            else:
                base['air_date.gte'] = min_date
            for page in range(1, PAGES_PER_TYPE + 1):
                jobs.append({'mt': mt, 'params': {**base, 'page': page},
                             'category': 'indian', 'genre': 'Hindi'})

        return jobs

    async def _fetch_page(self, session, job: Dict, semaphore) -> List[Dict]:
        """Fetch one /discover page, return parsed items. Retries on SSL errors."""
        url = f"https://api.themoviedb.org/3/discover/{job['mt']}"
        async with semaphore:
            for attempt in range(3):
                try:
                    async with session.get(url, params=job['params'], timeout=20) as resp:
                        if resp.status != 200:
                            return []
                        data = await resp.json()
                        items = []
                        for item in data.get('results', []):
                            mt = job['mt']
                            title = item.get('title') if mt == 'movie' else item.get('name')
                            release_date = (item.get('release_date') if mt == 'movie'
                                            else item.get('first_air_date')) or '2020-01-01'
                            parsed = {
                                'tmdb_id': item['id'],
                                'title': title,
                                'original_title': item.get('original_title') or item.get('original_name'),
                                'content_type': mt,
                                'release_year': int(release_date[:4]),
                                'poster_path': item.get('poster_path'),
                                'overview': item.get('overview'),
                                'popularity': item.get('popularity', 0),
                                'imdb_rating': item.get('vote_average'),
                                'category': job['category'],
                                'genre': job['genre'],
                            }
                            if job['category'] == 'indian':
                                parsed['language'] = 'hi'
                            items.append(parsed)
                        return items
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(0.75 * (attempt + 1))
            return []

    async def _fetch_all_async(self, jobs: List[Dict]) -> List[Dict]:
        """
        Fire all discover-page fetches AND provider checks concurrently in one
        aiohttp session — eliminates the slow per-item sync thread-pool entirely.
        """
        import aiohttp
        import asyncio as aio

        # ── Phase 1: fetch all /discover pages ────────────────────────────
        page_sem = aio.Semaphore(20)   # slightly more aggressive — TMDb handles it fine
        connector = aiohttp.TCPConnector(ssl=False, limit=60)
        async with aiohttp.ClientSession(connector=connector) as session:
            page_tasks = [self._fetch_page(session, job, page_sem) for job in jobs]
            page_results = await aio.gather(*page_tasks, return_exceptions=True)

            raw_items: List[Dict] = []
            for r in page_results:
                if isinstance(r, list):
                    raw_items.extend(r)

            # Deduplicate per (tmdb_id, category) before provider checks
            seen_keys: set = set()
            unique_items: List[Dict] = []
            for item in raw_items:
                key = (item['tmdb_id'], item['category'])
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_items.append(item)

            # ── Phase 2: provider checks — all async, same session ─────────
            # TMDb /discover already filtered by with_watch_providers so almost
            # every item will pass — we just need to know *which* platforms.
            # Group by tmdb_id to avoid duplicate provider fetches across categories.
            tmdb_id_to_items: Dict[int, List[Dict]] = {}
            for item in unique_items:
                tmdb_id_to_items.setdefault(item['tmdb_id'], []).append(item)

            provider_sem = aio.Semaphore(40)   # 40 concurrent provider requests

            async def fetch_providers(tmdb_id: int, content_type: str) -> tuple:
                url = (f"https://api.themoviedb.org/3/{content_type}"
                       f"/{tmdb_id}/watch/providers")
                params = {'api_key': self.api_key}
                for attempt in range(3):
                    try:
                        async with provider_sem:
                            async with session.get(url, params=params, timeout=10) as resp:
                                if resp.status == 429:          # rate-limited
                                    await aio.sleep(2 ** attempt)
                                    continue
                                if resp.status != 200:
                                    return tmdb_id, []
                                data = await resp.json()
                                india = data.get('results', {}).get('IN', {})
                                ids = [p['provider_id']
                                       for p in india.get('flatrate', [])]
                                return tmdb_id, ids
                    except Exception:
                        if attempt < 2:
                            await aio.sleep(0.5 * (attempt + 1))
                return tmdb_id, []

            # One provider fetch per unique tmdb_id (not per category copy)
            unique_tmdb = list(tmdb_id_to_items.items())
            total = len(unique_tmdb)
            print(f"\n🔍 Checking providers for {total} unique titles (async, 40 concurrent)...")

            provider_tasks = [
                fetch_providers(tid, items[0]['content_type'])
                for tid, items in unique_tmdb
            ]
            provider_results = await aio.gather(*provider_tasks, return_exceptions=True)

            # Build provider map: tmdb_id -> [platform_name, ...]
            provider_map: Dict[int, List[str]] = {}
            for res in provider_results:
                if isinstance(res, tuple):
                    tid, pids = res
                    provider_map[tid] = [
                        name for name, pid in Config.PLATFORMS.items()
                        if pid in pids
                    ]

        # Expand: one entry per (item, platform)
        expanded: List[tuple] = []
        for item in unique_items:
            for platform in provider_map.get(item['tmdb_id'], []):
                expanded.append((item, platform))

        return expanded   # list of (item_dict, platform_name)

    def fetch_all_discover(self) -> List[Dict]:
        """Entry point — builds jobs, runs async fetch+provider check, deduplicates."""
        import asyncio
        jobs = self._build_jobs()
        print(f"   🚀 Firing {len(jobs)} TMDb page requests concurrently...")

        def _run(coro):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(asyncio.run, coro).result()
                return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)

        expanded = _run(self._fetch_all_async(jobs))

        # Count by category for logging
        from collections import Counter
        counts = Counter(item['category'] for item, _ in expanded)
        for cat, n in sorted(counts.items()):
            print(f"   📦 {cat}: {n} titles")
        print(f"   📦 Total: {len(expanded)} (item, platform) pairs")
        return expanded   # caller receives pre-expanded pairs

    def save_discover_content(self):
        """Run fully-async fetch+provider pipeline, then batch-save to DB."""
        print("\n" + "="*70)
        print("🔍 DISCOVER FLOW - Collecting Content (No Reviews)")
        print("="*70)

        expanded = self.fetch_all_discover()   # list of (item, platform)

        if not expanded:
            print("❌ No content fetched — check API key / network")
            return

        print(f"\n   💾 Batch-saving {len(expanded)} platform entries...")

        # Deduplicate by (tmdb_id, platform) — same title can appear across categories
        # Keep the highest-rated category occurrence (classics > genre > underdog)
        CATEGORY_RANK = {'classics': 0, 'underdog': 1, 'indian': 2}
        seen_pairs: dict = {}
        for item, platform in expanded:
            key = (item['tmdb_id'], platform)
            rank = CATEGORY_RANK.get(item['category'], 3)
            if key not in seen_pairs or rank < CATEGORY_RANK.get(seen_pairs[key][0]['category'], 3):
                seen_pairs[key] = (item, platform)
        rows = []
        for (item, platform) in seen_pairs.values():
            rows.append({
                'tmdb_id':         item['tmdb_id'],
                'title':           item['title'],
                'original_title':  item.get('original_title'),
                'platform':        platform,
                'content_type':    item['content_type'],
                'release_year':    item['release_year'],
                'imdb_rating':     item['imdb_rating'],
                'poster_path':     item['poster_path'],
                'overview':        item['overview'],
                'category':        item['category'],
                'genre':           item.get('genre'),
                'popularity':      item['popularity'],
            })

        BATCH = 100
        saved = 0
        errors = 0
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            try:
                self.db.table('discover_content').upsert(
                    chunk, on_conflict='tmdb_id,platform'
                ).execute()
                saved += len(chunk)
                progress(min(i + BATCH, len(rows)), len(rows), "saving…")
            except Exception as e:
                errors += len(chunk)
                print(f"\n  ⚡️ Batch error at row {i}: {e}")

        print(f"\n✅ Saved {saved} items to discover_content" +
              (f" ({errors} errors)" if errors else ""))
        print("="*70)
# ============================================================================
# SENTIMENT ANALYSIS - 3-TIER CASCADE SYSTEM
# ============================================================================

class SentimentAnalyzer:
    def __init__(self):
        # Always initialize VADER as final fallback
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Tier 1: Groq (Fastest & Most generous free tier)
        self.groq_client = None
        self.use_groq = False
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                self.use_groq = True
                print("✅ Tier 1: Groq AI enabled (Primary)")
            except ImportError:
                print("❌ Tier 1 Skipped: 'groq' library not found. Run: pip install groq")
            except Exception as e:
                print(f"❌ Tier 1 Skipped: Groq Error - {e}")
        else:
            print("⚡️ Tier 1 Skipped: Missing GROQ_API_KEY in .env")
        
        # Tier 2: Gemini (Good but rate limited)
        self.gemini_client = None
        self.use_gemini = False
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_key)
                self.use_gemini = True
                print("✅ Tier 2: Gemini Flash enabled (Backup)")
            except Exception as e:
                print(f"⚡️ Gemini init failed: {e}")
        
        # Tier 3: VADER (Always available)
        print("✅ Tier 3: VADER enabled (Final Fallback)")
    
    def analyze(self, text: str) -> Dict:
        if not text or len(text.strip()) < 10:
            return {'sentiment': 0, 'confidence': 0.0}
        
        # Try Tier 1: Groq
        if self.use_groq:
            result = self._groq_analyze(text)
            if result:
                return result
            print("    ⤵️ Groq failed, falling back to Gemini...")        
        # Try Tier 2: Gemini
        if self.use_gemini:
            result = self._gemini_analyze(text)
            if result:
                return result
        
        # Tier 3: VADER (Always works)
        return self._vader_analyze(text)
    
    def _groq_analyze(self, text: str) -> Optional[Dict]:
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze sentiment of this review. Return ONLY JSON:
{{"sentiment": -1 or 0 or 1, "confidence": 0.0 to 1.0}}

Review: {text[:2000]}"""
                }],
                temperature=0.3,
                max_tokens=50
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            import json
            result = json.loads(result_text)
            
            sentiment = result.get('sentiment', 0)
            confidence = result.get('confidence', 0.5)
            
            if sentiment not in [-1, 0, 1]:
                sentiment = 0
            confidence = max(0.0, min(1.0, float(confidence)))
            
            return {'sentiment': sentiment, 'confidence': confidence}
            
        except Exception as e:
            # Check if rate limited
            if '429' in str(e) or 'rate' in str(e).lower():
                print(f"  ⚡️ Groq rate limited, trying Gemini...")
                self.use_groq = False  # Disable for this session
            else:
                print(f"  ⚡️ Groq error: {str(e)[:100]}")
            return None
    
    def _gemini_analyze(self, text: str) -> Optional[Dict]:
        try:
            prompt = f"""Analyze the sentiment of this movie/TV review.

Review: {text[:4000]}

Return ONLY a JSON object with this exact format:
{{"sentiment": -1 or 0 or 1, "confidence": 0.0 to 1.0}}

Where:
- sentiment: -1 (negative), 0 (neutral), 1 (positive)
- confidence: how certain you are (0.0 = unsure, 1.0 = very certain)"""

            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            result_text = response.text.strip()
            
            import json
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(result_text)
            
            sentiment = result.get('sentiment', 0)
            confidence = result.get('confidence', 0.5)
            
            if sentiment not in [-1, 0, 1]:
                sentiment = 0
            confidence = max(0.0, min(1.0, float(confidence)))
            
            return {'sentiment': sentiment, 'confidence': confidence}
            
        except Exception as e:
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                print(f"  ⚡️ Gemini rate limited, using VADER...")
                self.use_gemini = False
            return None
    
    def _vader_analyze(self, text: str) -> Dict:
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 1
        elif compound <= -0.05:
            sentiment = -1
        else:
            sentiment = 0
        
        confidence = min(1.0, abs(compound))
        return {'sentiment': sentiment, 'confidence': confidence}

# ============================================================================
# REDDIT INGESTER
# ============================================================================

class RedditIngester:
    # Reddit requires this exact UA format or returns 429/403
    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        # Real browser UA — Reddit blocks bot-format UAs for unauthenticated JSON API requests
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # Noise phrases to filter out low-signal comments
        self.noise_phrases = [
            'edit:', 'source:', 'http', 'just here', 'this is the way',
            'lol', 'lmao', 'haha', 'same', 'this', '^', 'deleted', 'removed'
        ]

    def _get_subreddits(self, title: str, media_type: str, is_hindi: bool) -> List[str]:
        """Return ordered list of subreddits to search, best signal first."""
        title_lower = title.lower()
        # Anime detection — check for common anime title patterns
        anime_keywords = ['kaisen', 'jujutsu', 'demon slayer', 'attack on titan',
                          'one piece', 'naruto', 'dragon ball', 'my hero', 'chainsaw',
                          'frieren', 'spy x', 'vinland', 'bleach', 'hunter x']
        is_anime = any(k in title_lower for k in anime_keywords)

        if is_anime:
            return ['anime', 'Animesuggest', 'television', 'movies']
        if is_hindi:
            return ['bollywood', 'india', 'HindiMovies',
                    'television' if media_type == 'tv' else 'movies']
        if media_type == 'tv':
            return ['television', 'NetflixBestOf', 'PrimeVideo', 'TrueFilm']
        return ['movies', 'TrueFilm', 'worldcinema', 'MovieSuggestions']

    def _get_queries(self, title: str) -> List[str]:
        """Multiple queries — fallback ensures we find something even for short/generic titles."""
        return [
            f"{title} review",
            f"{title} discussion",
            f'"{title}"',   # exact match catches threads that just mention the title
        ]

    def _search_subreddit(self, subreddit: str, query: str, limit: int = 5) -> List[Dict]:
        """Search a subreddit via Reddit JSON API."""
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {'q': query, 'restrict_sr': 'on', 'sort': 'relevance',
                  't': 'all', 'limit': limit}   # t=all not t=year — older threads still relevant
        for attempt in range(2):
            try:
                resp = self._session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    return resp.json().get('data', {}).get('children', [])
                if resp.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                if resp.status_code in (403, 404):
                    return []
            except Exception:
                time.sleep(1)
        return []

    def _extract_comments(self, thread_id: str) -> List[Dict]:
        """Fetch top comments (depth=1) for speed."""
        url = f"https://www.reddit.com/comments/{thread_id}.json"
        params = {'limit': 10, 'depth': 1, 'sort': 'top'}
        for attempt in range(2):
            try:
                resp = self._session.get(url, params=params, timeout=8)
                if resp.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                if resp.status_code != 200 or len(resp.json()) < 2:
                    return []
                comments_raw = resp.json()[1].get('data', {}).get('children', [])
                extracted = []
                for comment in comments_raw[:8]:
                    c_data = comment.get('data', {})
                    body = c_data.get('body', '')
                    if (body and body not in ('[deleted]', '[removed]')
                            and len(body) > 30
                            and not any(p in body.lower() for p in self.noise_phrases)):
                        sent = self.sentiment.analyze(body)
                        extracted.append({
                            'text': body[:600],
                            'sentiment': sent['sentiment'],
                            'confidence': sent['confidence'],
                            'score': c_data.get('score', 0)
                        })
                return extracted
            except Exception:
                time.sleep(1)
        return []

    def get_reddit_discussions(self, title: str, media_type: str,
                               is_hindi: bool = False) -> List[Dict]:
        """
        Search up to 2 subreddits with up to 3 query fallbacks.
        Stops as soon as 4 threads with comments are found.
        """
        subreddits = self._get_subreddits(title, media_type, is_hindi)[:2]
        queries = self._get_queries(title)

        seen_ids = set()
        all_threads = []

        for subreddit in subreddits:
            if len(all_threads) >= 4:
                break
            print(f"     📡 r/{subreddit}...", end=' ', flush=True)
            sub_found = 0

            for query in queries:
                if len(all_threads) >= 4:
                    break
                posts = self._search_subreddit(subreddit, query, limit=3)
                for post in posts:
                    if len(all_threads) >= 4:
                        break
                    post_data = post['data']
                    thread_id = post_data.get('id')
                    thread_title = post_data.get('title', '')
                    permalink = post_data.get('permalink', '')

                    if thread_id in seen_ids:
                        continue
                    skip_keywords = ['trailer', 'ama', 'casting', 'renewed', 'cancelled', 'official']
                    if any(k in thread_title.lower() for k in skip_keywords):
                        continue

                    seen_ids.add(thread_id)
                    comments = self._extract_comments(thread_id)
                    if comments:
                        all_threads.append({
                            'title': thread_title,
                            'url': f"https://www.reddit.com{permalink}",
                            'subreddit': subreddit,
                            'comments': comments,
                            'upvotes': post_data.get('score', 0)
                        })
                        sub_found += 1

                time.sleep(0.3)

            print(f"{sub_found} threads" if sub_found else "none")

        total_comments = sum(len(t['comments']) for t in all_threads)
        if all_threads:
            print(f"     ✅ {len(all_threads)} threads, {total_comments} comments across {len(subreddits)} subreddits")
        else:
            print(f"     ⚡️ No Reddit discussions found")
        return all_threads

    def compute_reddit_score(self, threads: List[Dict]) -> float:
        if not threads: return 50.0
        # Weight comments by their upvote score so highly-upvoted opinions matter more
        weighted_sum = 0.0
        weight_total = 0.0
        for thread in threads:
            thread_weight = max(1, thread.get('upvotes', 1))
            for comment in thread['comments']:
                comment_weight = max(1, comment.get('score', 1)) * thread_weight
                weighted_sum += comment['sentiment'] * comment_weight
                weight_total += comment_weight
        if weight_total == 0: return 50.0
        avg = weighted_sum / weight_total
        return (avg + 1) * 50

    def save_reddit_reviews(self, content_id: int, threads: List[Dict]):
        count = 0
        for thread in threads:
            for comment in thread['comments']:
                review_data = {
                    'content_id': content_id,
                    'source': 'reddit',
                    'source_url': thread['url'],
                    'source_id': f"{thread['url']}_{hash(comment['text'][:10])}",
                    'reviewer': f"r/{thread.get('subreddit', 'reddit')}",
                    'review_text': comment['text'],
                    'sentiment_score': (comment['sentiment'] + 1) * 50,
                    'confidence': comment['confidence'],
                    'weighted_sentiment': comment['sentiment'] * comment['confidence']
                }
                try:
                    self.db.table('reviews').upsert(review_data, on_conflict='source,source_id').execute()
                    count += 1
                except: pass
        if count > 0:
            print(f"     💾 Saved {count} Reddit comments")


# ============================================================================
# CRITIC REVIEW SCRAPER
# ============================================================================

class CriticReviewScraper:
    """
    Scrapes professional critic reviews from:
      - Rotten Tomatoes  (Tomatometer % + Audience Score)
      - Decider          (Stream It / Skip It verdict)
      - RogerEbert.com   (0-4 stars)
      - Vulture          (letter grade)
      - Bollywood Hungama (Hindi content, 1-5 stars)
      - Metacritic       (0-100 metascore)

    Each scraper has:
      - 3 retries with exponential backoff
      - Detailed debug logging
      - Graceful fallback (returns None, never crashes)
    """

    def __init__(self, debug: bool = False):
        self.sentiment = SentimentAnalyzer()
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.debug = debug
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })

    def _log(self, msg: str):
        if self.debug:
            print(f"     [DEBUG] {msg}")

    def _get(self, url: str, timeout: int = 8) -> Optional[BeautifulSoup]:
        """Fetch a URL with 2 retries + exponential backoff. Returns soup or None."""
        for attempt in range(2):
            try:
                self._log(f"GET {url} (attempt {attempt+1})")
                resp = self.session.get(url, timeout=timeout, allow_redirects=True)
                self._log(f"→ {resp.status_code} ({len(resp.text)} chars)")

                if resp.status_code == 200:
                    return BeautifulSoup(resp.text, 'html.parser')
                elif resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    self._log(f"Rate limited, waiting {wait}s")
                    time.sleep(wait)
                elif resp.status_code == 403:
                    self._log("403 Forbidden — site blocking scraper")
                    return None
                elif resp.status_code == 404:
                    self._log("404 Not Found")
                    return None
                else:
                    self._log(f"Unexpected status {resp.status_code}")
                    time.sleep(2 * (attempt + 1))

            except requests.exceptions.Timeout:
                self._log(f"Timeout on attempt {attempt+1}")
                time.sleep(2 * (attempt + 1))
            except requests.exceptions.ConnectionError as e:
                self._log(f"Connection error: {str(e)[:80]}")
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                self._log(f"Unexpected error: {str(e)[:80]}")
                time.sleep(2)

        self._log(f"All attempts failed for {url}")
        return None

    def _safe_int(self, text: str) -> Optional[int]:
        """Extract first integer from a string safely"""
        m = re.search(r'(\d+)', text)
        return int(m.group(1)) if m else None

    def _safe_float(self, text: str) -> Optional[float]:
        m = re.search(r'(\d+(?:\.\d+)?)', text)
        return float(m.group(1)) if m else None

    # ------------------------------------------------------------------
    # DDG SEARCH HELPER — finds review URLs without hitting blocked search pages
    # ------------------------------------------------------------------
    def _ddg_find_url(self, query: str, must_contain: str) -> Optional[str]:
        """
        Use DuckDuckGo Lite (plain HTML, no JS, no bot-block) to find a URL
        whose href contains `must_contain`.  Returns the first match or None.
        """
        try:
            encoded = urllib.parse.quote_plus(query)
            ddg_url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
            soup = self._get(ddg_url, timeout=15)
            if not soup:
                return None
            for a in soup.find_all('a', href=True):
                href = a['href']
                # DDG lite wraps links — unwrap uddg= redirect if present
                if 'uddg=' in href:
                    href = urllib.parse.unquote(href.split('uddg=')[-1])
                if must_contain in href:
                    self._log(f"DDG found: {href}")
                    return href
        except Exception as e:
            self._log(f"DDG search error: {e}")
        return None

    # ------------------------------------------------------------------
    # ROTTEN TOMATOES
    # ------------------------------------------------------------------
    def scrape_rotten_tomatoes(self, title: str, media_type: str,
                                year: Optional[int] = None) -> Optional[Dict]:
        """
        Fetch RT Tomatometer via OMDB API (reliable, no scraping, free tier = 1000/day).
        Falls back to direct RT scraping if OMDB_API_KEY is not set.
        OMDB key: https://www.omdbapi.com/apikey.aspx (free)
        """
        omdb_key = os.getenv('OMDB_API_KEY')
        if omdb_key:
            return self._rt_via_omdb(title, media_type, year, omdb_key)
        return self._rt_via_scrape(title, media_type, year)

    def _rt_via_omdb(self, title: str, media_type: str,
                     year: Optional[int], api_key: str) -> Optional[Dict]:
        """Use OMDB API — returns RT Tomatometer + Audience Score cleanly."""
        try:
            params = {
                'apikey': api_key,
                't': title,
                'type': 'series' if media_type == 'tv' else 'movie',
                'tomatoes': 'true',
                'r': 'json',
            }
            if year:
                params['y'] = year
            resp = self.session.get('https://www.omdbapi.com/', params=params, timeout=8)
            data = resp.json()

            if data.get('Response') == 'False':
                # Try without year if year-match failed
                if year and 'not found' in data.get('Error','').lower():
                    params.pop('y', None)
                    resp = self.session.get('https://www.omdbapi.com/', params=params, timeout=8)
                    data = resp.json()
                if data.get('Response') == 'False':
                    self._log(f"OMDB: {data.get('Error', 'not found')}")
                    return None

            ratings = {r['Source']: r['Value'] for r in data.get('Ratings', [])}
            rt_raw = ratings.get('Rotten Tomatoes', '')   # e.g. "94%"
            tomatometer = self._safe_int(rt_raw)

            # OMDB also has Metascore but we only want RT here
            if tomatometer is None:
                self._log("OMDB: no RT rating in response")
                return None

            if tomatometer >= 70:
                sentiment, confidence = 1, min(1.0, tomatometer / 100)
            elif tomatometer <= 40:
                sentiment, confidence = -1, min(1.0, (50 - tomatometer) / 50)
            else:
                sentiment, confidence = 0, 0.5

            verdict = f"Tomatometer: {tomatometer}%"
            imdb_r = data.get('imdbRating', '')
            if imdb_r and imdb_r != 'N/A':
                verdict += f" | IMDb: {imdb_r}"

            print(f"   🍅 RT (OMDB): {verdict}")
            return {
                'source': 'rotten_tomatoes',
                'url': f"https://www.rottentomatoes.com/search?search={requests.utils.quote(title)}",
                'verdict': verdict,
                'sentiment': sentiment,
                'confidence': confidence,
                'text': verdict,
                'reviewer': 'Rotten Tomatoes',
            }
        except Exception as e:
            self._log(f"OMDB error: {e}")
            return None

    def _rt_via_scrape(self, title: str, media_type: str,
                       year: Optional[int] = None) -> Optional[Dict]:
        """
        Direct RT scrape fallback (used only when no OMDB key).
        RT now returns 403 for most requests without a valid browser session —
        this path will often return None. Set OMDB_API_KEY for reliable results.
        """
        slug = re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()
        slug = re.sub(r"\s+", "_", slug)

        if media_type == 'tv':
            candidates = [
                f"https://www.rottentomatoes.com/tv/{slug}",
                f"https://www.rottentomatoes.com/tv/{slug}_1",
            ]
        else:
            candidates = [
                f"https://www.rottentomatoes.com/m/{slug}",
                f"https://www.rottentomatoes.com/m/{slug}_{year}" if year else None,
            ]
        candidates = [c for c in candidates if c]

        soup = None
        used_url = None
        for url in candidates:
            soup = self._get(url)
            if soup and not soup.select_one('[data-qa="error-page"]'):
                used_url = url
                break

        if not soup:
            search_url = f"https://www.rottentomatoes.com/search?search={requests.utils.quote(title)}"
            search_soup = self._get(search_url)
            if search_soup:
                for a in search_soup.find_all('a', href=True):
                    href = a['href']
                    if (f'/m/{slug[:4]}' in href or f'/tv/{slug[:4]}' in href):
                        full_url = 'https://www.rottentomatoes.com' + href if href.startswith('/') else href
                        soup = self._get(full_url)
                        if soup:
                            used_url = full_url
                            break

        if not soup:
            self._log("RT scrape: no page found (likely 403 blocked)")
            print(f"   ⚡️ RT: blocked — add OMDB_API_KEY to .env for reliable scores")
            return None

        tomatometer = None
        audience_score = None

        for script in soup.find_all('script', type='application/ld+json'):
            try:
                import json as _json
                data = _json.loads(script.string or '')
                rating = data.get('aggregateRating', {})
                if rating:
                    val = self._safe_float(str(rating.get('ratingValue', '')))
                    if val:
                        tomatometer = int(val * 10) if val <= 10 else int(val)
                        break
            except Exception:
                pass

        if tomatometer is None:
            for selector in [
                '[data-qa="tomatometer"]', 'score-board[tomatometerscore]',
                'rt-text[slot="criticsScore"]', 'span.mop-ratings-wrap__percentage',
            ]:
                tag = soup.select_one(selector)
                if tag:
                    val = self._safe_int(tag.get_text(strip=True) or tag.get('tomatometerscore', '') or '')
                    if val is not None:
                        tomatometer = val
                        break

        for selector in [
            '[data-qa="audience-score"]', 'score-board[audiencescore]',
            'rt-text[slot="audienceScore"]',
        ]:
            tag = soup.select_one(selector)
            if tag:
                val = self._safe_int(tag.get_text(strip=True) or tag.get('audiencescore', '') or '')
                if val is not None:
                    audience_score = val
                    break

        if tomatometer is None and audience_score is None:
            return None

        primary = tomatometer if tomatometer is not None else audience_score
        combined = int(tomatometer * 0.6 + audience_score * 0.4) if (tomatometer and audience_score) else primary

        if combined >= 70:
            sentiment, confidence = 1, min(1.0, combined / 100)
        elif combined <= 40:
            sentiment, confidence = -1, min(1.0, (50 - combined) / 50)
        else:
            sentiment, confidence = 0, 0.5

        parts = []
        if tomatometer is not None: parts.append(f"Tomatometer: {tomatometer}%")
        if audience_score is not None: parts.append(f"Audience: {audience_score}%")
        verdict = " | ".join(parts)
        print(f"   🍅 RT (scraped): {verdict}")

        return {
            'source': 'rotten_tomatoes',
            'url': used_url,
            'verdict': verdict,
            'sentiment': sentiment,
            'confidence': confidence,
            'text': verdict,
            'reviewer': 'Rotten Tomatoes',
        }


    # ------------------------------------------------------------------
    # DECIDER — Stream It or Skip It
    # ------------------------------------------------------------------
    def scrape_decider(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        # Decider search is Cloudflare-blocked — use their RSS feed to find the review URL
        # RSS search endpoint works without JS: /feed/?s=<query>
        slug_plus = title.lower().replace(" ", "+")
        rss_url = f"https://decider.com/feed/?s={slug_plus}+stream+it+or+skip+it"
        rss_soup = self._get(rss_url)

        link = None
        if rss_soup:
            # RSS items have <link> tags with the full URL
            for item in rss_soup.find_all('item'):
                item_link = item.find('link')
                item_title = item.find('title')
                if item_link and item_title:
                    href = item_link.get_text(strip=True) or item_link.next_sibling
                    if isinstance(href, str) and 'stream-it-or-skip-it' in href:
                        title_words = [w for w in title.lower().split() if len(w) > 2]
                        if any(w in href.lower() for w in title_words[:2]):
                            link = href.strip()
                            self._log(f"Decider RSS link: {link}")
                            break

        if not link:
            self._log("Decider: no Stream It or Skip It link found in RSS feed")
            return None

        review_soup = self._get(link)
        if not review_soup:
            return None

        verdict_text = ''
        for tag in review_soup.find_all(['h2', 'h3', 'strong', 'div', 'p']):
            text = tag.get_text(separator=' ', strip=True).upper()
            if any(v in text for v in ['STREAM IT', 'SKIP IT', 'SOME STREAMS']):
                verdict_text = text[:100]
                self._log(f"Decider verdict text: {verdict_text}")
                break

        if not verdict_text:
            self._log("Decider: verdict phrase not found on review page")
            return None

        if 'SKIP IT' in verdict_text:
            sentiment, confidence = -1, 0.95
        elif 'SOME STREAMS' in verdict_text:
            sentiment, confidence = 0, 0.8
        else:
            sentiment, confidence = 1, 0.95

        content_div = review_soup.select_one('div.entry-content, div.post-content, article')
        body = ''
        if content_div:
            body = ' '.join(p.get_text(strip=True) for p in content_div.find_all('p')[:6])

        return {
            'source': 'decider',
            'url': link,
            'verdict': verdict_text[:80],
            'sentiment': sentiment,
            'confidence': confidence,
            'text': body[:800] or verdict_text,
            'reviewer': 'Decider'
        }

    # ------------------------------------------------------------------
    # ROGEREBERT.COM
    # ------------------------------------------------------------------
    def scrape_rogerebert(self, title: str, media_type: str) -> Optional[Dict]:
        # rogerebert.com uses predictable /reviews/<slug> URLs — try candidates directly
        slug = re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()
        slug = re.sub(r"\s+", "-", slug)
        # Try common slug variants
        year_suffix = ""  # can extend later if needed
        candidates = [
            f"https://www.rogerebert.com/reviews/{slug}",
            f"https://www.rogerebert.com/reviews/{slug}-film-review",
            f"https://www.rogerebert.com/reviews/the-{slug}" if not slug.startswith("the-") else None,
        ]
        candidates = [c for c in candidates if c]

        link = None
        for url in candidates:
            soup_check = self._get(url)
            if soup_check and not soup_check.select_one('[class*="error"]'):
                # Verify it looks like a review page (has review body or star rating)
                if soup_check.select_one('div.review-content, div[itemprop="reviewBody"], [class*="star-rating"], [aria-label*="star"]'):
                    link = url
                    self._log(f"RogerEbert direct URL hit: {url}")
                    break
            time.sleep(0.5)

        if not link:
            self._log("RogerEbert: no direct URL matched")
            return None

        review_soup = self._get(link)
        if not review_soup:
            return None

        stars = None
        for selector in [
            '[aria-label*="star"]', 'abbr[title*="star"]',
            '[class*="star-rating"]', '[class*="starRating"]'
        ]:
            tag = review_soup.select_one(selector)
            if tag:
                label = tag.get('aria-label', '') or tag.get('title', '') or tag.get_text()
                val = self._safe_float(label)
                if val is not None and 0 <= val <= 4:
                    stars = val
                    self._log(f"RogerEbert stars: {stars}")
                    break

        body_tag = review_soup.select_one('div.review-content, div[itemprop="reviewBody"], div.page-content')
        body = body_tag.get_text(strip=True)[:800] if body_tag else ''

        if stars is not None:
            if stars >= 3:
                sentiment, confidence = 1, min(1.0, stars / 4)
            elif stars <= 1.5:
                sentiment, confidence = -1, min(1.0, (2 - stars) / 2)
            else:
                sentiment, confidence = 0, 0.5
        elif body:
            r = self.sentiment.analyze(body[:1000])
            sentiment, confidence = r['sentiment'], r['confidence']
            self._log("RogerEbert: no star rating, used text sentiment")
        else:
            self._log("RogerEbert: no rating and no body text")
            return None

        return {
            'source': 'rogerebert',
            'url': link,
            'verdict': f"{stars}/4 stars" if stars is not None else 'Text only',
            'sentiment': sentiment,
            'confidence': confidence,
            'text': body,
            'reviewer': 'RogerEbert.com'
        }

    # ------------------------------------------------------------------
    # VULTURE
    # ------------------------------------------------------------------
    def scrape_vulture(self, title: str) -> Optional[Dict]:
        # Vulture review URLs follow predictable patterns — try candidates directly
        slug = re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()
        slug = re.sub(r"\s+", "-", slug)
        candidates = [
            f"https://www.vulture.com/article/{slug}-review",
            f"https://www.vulture.com/movies/{slug}-review",
            f"https://www.vulture.com/tv/{slug}-review",
            f"https://www.vulture.com/article/{slug}-movie-review",
            f"https://www.vulture.com/article/{slug}-tv-review",
        ]

        link = None
        review_soup = None
        for url in candidates:
            soup_check = self._get(url)
            if soup_check and soup_check.select_one(
                'div.article-content, section.article-body, div[class*="body"], div[class*="article"]'
            ):
                link = url
                review_soup = soup_check
                self._log(f"Vulture direct URL hit: {url}")
                break
            time.sleep(0.5)

        if not link or not review_soup:
            self._log("Vulture: no direct URL matched")
            return None

        grade_map = {
            'A+': (1, 1.0), 'A': (1, 0.95), 'A-': (1, 0.85),
            'B+': (1, 0.75), 'B': (1, 0.65), 'B-': (0, 0.55),
            'C+': (0, 0.55), 'C': (0, 0.5), 'C-': (-1, 0.55),
            'D+': (-1, 0.65), 'D': (-1, 0.75), 'D-': (-1, 0.85), 'F': (-1, 0.95)
        }

        grade = None
        for tag in review_soup.find_all(['span', 'div', 'p']):
            text = tag.get_text(strip=True)
            if re.match(r'^[A-F][+-]?$', text):
                grade = text
                self._log(f"Vulture grade: {grade}")
                break

        body_tag = review_soup.select_one('div.article-content, section.article-body, div[class*="body"]')
        body = body_tag.get_text(strip=True)[:800] if body_tag else ''

        if grade and grade in grade_map:
            sentiment, confidence = grade_map[grade]
        elif body:
            r = self.sentiment.analyze(body[:1000])
            sentiment, confidence = r['sentiment'], r['confidence']
            grade = 'N/A'
            self._log("Vulture: no grade found, used text sentiment")
        else:
            self._log("Vulture: no grade and no body")
            return None

        return {
            'source': 'vulture',
            'url': link,
            'verdict': f"Grade: {grade}",
            'sentiment': sentiment,
            'confidence': confidence,
            'text': body,
            'reviewer': 'Vulture'
        }

    # ------------------------------------------------------------------
    # BOLLYWOOD HUNGAMA
    # ------------------------------------------------------------------
    def scrape_bollywood_hungama(self, title: str) -> Optional[Dict]:
        slug = re.sub(r"[^a-z0-9\s-]", "", title.lower()).strip().replace(' ', '-')
        urls = [
            f"https://www.bollywoodhungama.com/movie/{slug}/review/",
            f"https://www.bollywoodhungama.com/search/?q={title.replace(' ', '+')}",
        ]

        for url in urls:
            soup = self._get(url)
            if not soup:
                continue

            stars = None
            for selector in [
                '[class*="rating"] span', '[class*="star-rating"]',
                'meta[itemprop="ratingValue"]', '[class*="review-score"]'
            ]:
                tag = soup.select_one(selector)
                if tag:
                    val = self._safe_float(tag.get('content', '') or tag.get_text())
                    if val and 0 < val <= 5:
                        stars = val
                        self._log(f"BollywoodHungama stars: {stars} from {selector}")
                        break

            if stars is None:
                self._log(f"BollywoodHungama: no rating found at {url}")
                continue

            if stars >= 3.5:
                sentiment, confidence = 1, min(1.0, stars / 5)
            elif stars <= 2.0:
                sentiment, confidence = -1, min(1.0, (3 - stars) / 3)
            else:
                sentiment, confidence = 0, 0.5

            body_tag = soup.select_one('div.review-body, div[class*="review-content"], article')
            body = body_tag.get_text(strip=True)[:800] if body_tag else ''

            return {
                'source': 'bollywood_hungama',
                'url': url,
                'verdict': f"{stars}/5 stars",
                'sentiment': sentiment,
                'confidence': confidence,
                'text': body,
                'reviewer': 'Bollywood Hungama'
            }
        return None

    # ------------------------------------------------------------------
    # METACRITIC
    # ------------------------------------------------------------------
    def scrape_metacritic(self, title: str, media_type: str) -> Optional[Dict]:
        content_type = 'tv' if media_type == 'tv' else 'movie'
        base_slug = re.sub(r"[^a-z0-9\s-]", "", title.lower()).strip().replace(' ', '-')

        # Build candidate slugs — Metacritic sometimes uses full subtitles for sequels
        candidates = [base_slug]

        # If title ends with a bare number (e.g. "Pushpa 2"), expand to common sequel patterns
        m = re.match(r'^(.+)-(\d+)$', base_slug)
        if m:
            stem, num = m.group(1), m.group(2)
            ordinals = {'2': 'two', '3': 'three', '4': 'four'}
            candidates += [
                f"{stem}-the-rule---part-{num}",  # Pushpa 2 (triple dash edge case)
                f"{stem}-the-rule-part-{num}",
                f"{stem}-part-{num}",
                f"{stem}-{ordinals.get(num, num)}",
            ]

        soup = None
        used_url = None
        for slug in candidates:
            url = f"https://www.metacritic.com/{content_type}/{slug}/"
            soup = self._get(url)
            if soup:
                used_url = url
                self._log(f"Metacritic matched: {url}")
                break
        if not soup:
            return None

        score = None
        for selector in [
            'div.c-siteReviewScore span',
            'span[class*="metascore"]',
            'div[class*="metascore_w"] span',
            '[data-v-app] span.c-siteReviewScore_background span',
            'span.c-siteReviewScore_background'
        ]:
            tag = soup.select_one(selector)
            if tag:
                val = self._safe_int(tag.get_text(strip=True))
                if val and 0 <= val <= 100:
                    score = val
                    self._log(f"Metacritic score: {score} from {selector}")
                    break

        if score is None:
            self._log("Metacritic: score not found — page may be JS-rendered")
            return None

        if score >= 61:
            sentiment, confidence = 1, min(1.0, (score - 60) / 40)
        elif score <= 40:
            sentiment, confidence = -1, min(1.0, (41 - score) / 41)
        else:
            sentiment, confidence = 0, 0.5

        return {
            'source': 'metacritic',
            'url': url,
            'verdict': f"Metascore: {score}/100",
            'sentiment': sentiment,
            'confidence': confidence,
            'text': f"Metacritic aggregated critic score: {score}/100",
            'reviewer': 'Metacritic'
        }

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def fetch_all(self, content_id: int, title: str, media_type: str,
                  year: Optional[int] = None, is_hindi: bool = False) -> int:
        """Run active scrapers (RT only), save to DB, return count saved.
        Bollywood Hungama, Metacritic, and Decider removed — too slow / unreliable.
        """

        scrapers = [
            ('Rotten Tomatoes', lambda: self.scrape_rotten_tomatoes(title, media_type, year)),
        ]

        saved = 0
        for name, fn in scrapers:
            try:
                result = fn()
                if not result:
                    print(f"   📰 {name}: not found")
                    continue

                review_data = {
                    'content_id': content_id,
                    'source': result['source'],
                    'source_url': result['url'],
                    'source_id': f"{result['source']}_{content_id}_{result.get('url','')[-20:].replace('/','-')}",
                    'reviewer': result['reviewer'],
                    'review_text': result.get('text', '')[:800],
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'weighted_sentiment': result['sentiment'] * result['confidence']
                }
                self.db.table('reviews').upsert(review_data, on_conflict='source,source_id').execute()
                icon = '👍' if result['sentiment'] == 1 else '👎' if result['sentiment'] == -1 else '🤷'
                print(f"   📰 {name}: {result['verdict']} {icon}")
                saved += 1

            except Exception as e:
                print(f"   ❌ {name} FAILED: {str(e)[:80]}")
                self._log(f"{name} full error: {e}")

        return saved

# ============================================================================
# SCORING ENGINE
# ============================================================================

class ScoringEngine:
    @staticmethod
    def youtube_weight(views: int, subscribers: int, comments: int) -> float:
        """Combined authority + engagement weight"""
        view_weight = min(1.0, math.log10(views + 1) / 6)
        sub_weight = min(1.0, math.log10(subscribers + 1) / 6)
        authority = view_weight * sub_weight
        
        engagement_boost = min(0.3, math.log10(comments + 1) / 10)
        return min(1.0, authority + engagement_boost)
    
    @staticmethod
    def normalize_imdb(rating: float) -> float:
        if rating is None or rating < 0:
            return 50
        return max(0, min(100, (rating - 5) * 20))
    
    @staticmethod
    def get_dynamic_weights(release_year: int) -> Dict:
        """Dynamic weights based on content age (Recency Decay)"""
        current_year = 2026
        age = current_year - release_year if release_year else 10
        
        if age <= 1:
            return {'youtube': 0.65, 'imdb': 0.35}
        elif age <= 3:
            return {'youtube': 0.50, 'imdb': 0.50}
        elif age <= 5:
            return {'youtube': 0.40, 'imdb': 0.60}
        else:
            return {'youtube': 0.30, 'imdb': 0.70}
    
    @staticmethod
    def get_category(release_year: int) -> str:
        """Categorize content as Trending or Catalog"""
        current_year = 2026
        age = current_year - release_year if release_year else 10
        
        if age <= 2:
            return "trending"
        else:
            return "catalog"
    
    @staticmethod
    def get_label(score: float) -> str:
        if score >= 80:
            return "🔥 Must Watch"
        elif score >= 65:
            return "👍 Worth Your Time"
        elif score >= 50:
            return "🤷 Genre Fans Only"
        return "💤 Skip"

# ============================================================================
# YOUTUBE INGESTER
# ============================================================================

class YouTubeIngester:
    def __init__(self):
        self.api_key = Config.YOUTUBE_API_KEY
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.tmdb = TMDbResolver()
        self.sentiment = SentimentAnalyzer()
        self.scoring = ScoringEngine()
        self.reddit = RedditIngester() if Config.USE_REDDIT else None
        self.critic = CriticReviewScraper()
    
    def search_videos(self, query: str, max_results: int = 5) -> List[Dict]:
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={
                    'part': 'snippet',
                    'q': query,
                    'type': 'video',
                    'maxResults': max_results,
                    'key': self.api_key,
                    'order': 'relevance'
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            return [{
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel': item['snippet']['channelTitle'],
                'channel_id': item['snippet']['channelId']
            } for item in data.get('items', [])]
        except Exception as e:
            print(f"❌ YouTube search error: {e}")
            return []
    
    def get_video_stats(self, video_id: str, channel_id: str) -> Dict:
        try:
            response = requests.get(
                f"{self.base_url}/videos",
                params={'part': 'statistics', 'id': video_id, 'key': self.api_key},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get('items'):
                return {}
            
            stats = data['items'][0]['statistics']
            
            time.sleep(0.5)
            channel_response = requests.get(
                f"{self.base_url}/channels",
                params={'part': 'statistics', 'id': channel_id, 'key': self.api_key},
                timeout=10
            )
            channel_response.raise_for_status()
            channel_data = channel_response.json()
            
            subscribers = 0
            if channel_data.get('items'):
                subscribers = int(channel_data['items'][0]['statistics'].get('subscriberCount', 0))
            
            return {
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
                'subscribers': subscribers
            }
        except Exception as e:
            print(f"  ⚡️ Stats error: {e}")
            return {}
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        if not Config.USE_TRANSCRIPTS:
            return None
        
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([entry['text'] for entry in transcript_list])
            return transcript_text[:4000]
        except:
            return None
    
    def process_trending_content(self, trending_data: Dict, platform: str, platform_id: int):
        title = trending_data['title']
        tmdb_id = trending_data['tmdb_id']
        media_type = trending_data['content_type']
        
        print(f"\n🎬 {title} ({media_type})")
        
        providers = self.tmdb.get_watch_providers(tmdb_id, media_type)
        
        if not providers or platform_id not in providers:
            print(f"   ⚡️ Not available on {platform}, skipping for accuracy")
            return
        
        print(f"   ✅ Confirmed on {platform}")
        
        # For Hindi content, search with Hindi keyword for more relevant reviews
        if trending_data.get('is_indian'):
            query = f"{title} Hindi review {platform}"
        else:
            query = f"{title} {platform} review"
        
        print(f"   🔍 Searching: {query}")
        
        videos = self.search_videos(query, max_results=2)
        
        if not videos:
            print(f"   ⚡️ No videos found")
            return
        
        content_data = {
            'tmdb_id': tmdb_id,
            'title': title,
            'original_title': trending_data['original_title'],
            'platform': platform,
            'content_type': media_type,
            'release_year': trending_data['release_year'],
            'imdb_rating': trending_data['imdb_rating'],
            'poster_path': trending_data['poster_path'],
            'overview': trending_data['overview'],
            'discovery_source': trending_data.get('category', 'trending')
        }
        
        try:
            content_result = self.db.table('content').upsert(content_data, on_conflict='tmdb_id').execute()
            content_id = content_result.data[0]['id']
        except Exception as e:
            print(f"   ❌ DB error: {e}")
            return
        
        # ── YouTube: process videos sequentially (same API key, rate sensitive)
        for video in videos:
            print(f"     📺 {video['title'][:50]}...")
            stats = self.get_video_stats(video['video_id'], video['channel_id'])
            if not stats:
                continue
            transcript = self.get_transcript(video['video_id'])
            if transcript:
                review_text = transcript
                print(f"     📝 Transcript ({len(transcript)} chars)")
            else:
                review_text = f"{video['title']} {video['description']}"
                print(f"     📝 Title + description")
            sentiment_result = self.sentiment.analyze(review_text)
            print(f"     💭 Sentiment: {sentiment_result['sentiment']} (conf: {sentiment_result['confidence']:.2f})")
            youtube_weight = self.scoring.youtube_weight(
                stats.get('views', 0), stats.get('subscribers', 0), stats.get('comments', 0)
            )
            weighted_sentiment = sentiment_result['sentiment'] * sentiment_result['confidence'] * youtube_weight
            review_data = {
                'content_id': content_id,
                'source': 'youtube',
                'source_url': f"https://youtube.com/watch?v={video['video_id']}",
                'source_id': video['video_id'],
                'reviewer': video['channel'],
                'reviewer_subscribers': stats.get('subscribers', 0),
                'review_text': review_text[:1000],
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'views': stats.get('views', 0),
                'likes': stats.get('likes', 0),
                'comments_count': stats.get('comments', 0),
                'youtube_weight': youtube_weight,
                'weighted_sentiment': weighted_sentiment
            }
            try:
                self.db.table('reviews').upsert(review_data, on_conflict='source,source_id').execute()
                print(f"     💾 Saved")
            except Exception as e:
                print(f"     ❌ DB error: {e}")
            time.sleep(0.2)  # reduced from 0.5

        # ── Reddit + TMDb reviews + Critics run CONCURRENTLY (all hit different servers)
        is_hindi = trending_data.get('is_indian', False)

        def run_reddit():
            if not self.reddit:
                return
            print(f"   📡 Reddit...", flush=True)
            threads = self.reddit.get_reddit_discussions(title, media_type, is_hindi=is_hindi)
            if threads:
                score = self.reddit.compute_reddit_score(threads)
                print(f"   📊 Reddit: {score:.1f} ({len(threads)} threads)")
                self.reddit.save_reddit_reviews(content_id, threads)
            else:
                print(f"   ⚡️ Reddit: no discussions found")

        def run_tmdb_reviews():
            self._fetch_tmdb_reviews(content_id, tmdb_id, media_type)

        def run_critics():
            # RT via scraping only — Decider/Metacritic/BollywoodHungama removed for speed
            count = self.critic.fetch_all(content_id, title, media_type,
                                          year=trending_data.get('release_year'),
                                          is_hindi=is_hindi)
            if count == 0:
                print(f"   ⚡️ Critics: no scraper reviews found")

        print(f"   🔀 Running Reddit + TMDb reviews + Critics in parallel...")
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(run_reddit), ex.submit(run_tmdb_reviews), ex.submit(run_critics)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"   ❌ Parallel task error: {str(e)[:80]}")
    
    def _fetch_decider_youtube(self, content_id: int, title: str):
        """
        Fetch Decider's 'Stream It or Skip It' verdict via YouTube search.
        Decider posts these as YouTube videos — verdict is in the video title itself.
        No scraping needed, uses the existing YouTube API key.
        """
        query = f'"{title}" "stream it or skip it" decider'
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={
                    'part': 'snippet',
                    'q': query,
                    'type': 'video',
                    'maxResults': 3,
                    'key': self.api_key,
                    'order': 'relevance'
                },
                timeout=15
            )
            response.raise_for_status()
            items = response.json().get('items', [])
        except Exception as e:
            print(f"   ⚡️ Decider YouTube search error: {e}")
            return

        for item in items:
            video_title = item['snippet']['title']
            description = item['snippet']['description']
            channel = item['snippet']['channelTitle']
            video_id = item['id']['videoId']

            # Must be from Decider's own channel and contain the verdict phrase
            if 'decider' not in channel.lower() and 'decider' not in video_title.lower():
                continue

            title_upper = video_title.upper()
            if 'STREAM IT' in title_upper:
                verdict, sentiment, confidence = 'Stream It', 1, 0.95
            elif 'SKIP IT' in title_upper:
                verdict, sentiment, confidence = 'Skip It', -1, 0.95
            elif 'SOME STREAMS' in title_upper:
                verdict, sentiment, confidence = 'Some Streams', 0, 0.80
            else:
                continue  # Not a verdict video

            review_data = {
                'content_id':        content_id,
                'source':            'decider',
                'source_url':        f"https://youtube.com/watch?v={video_id}",
                'source_id':         f"decider_{video_id}",
                'reviewer':          'Decider',
                'review_text':       f"{verdict} — {video_title}",
                'sentiment':         sentiment,
                'confidence':        confidence,
                'weighted_sentiment': sentiment * confidence,
            }
            try:
                self.db.table('reviews').upsert(review_data, on_conflict='source,source_id').execute()
                icon = '👍' if sentiment == 1 else '👎' if sentiment == -1 else '🤷'
                print(f"   📰 Decider (YouTube): {verdict} {icon}")
            except Exception as e:
                print(f"   ❌ Decider save error: {e}")
            return  # Only need the first matching verdict

    def _fetch_tmdb_reviews(self, content_id: int, tmdb_id: int, media_type: str):
        """Fetch user reviews from TMDB — free, no extra API key needed"""
        try:
            resp = requests.get(
                f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/reviews",
                params={'api_key': self.tmdb.api_key, 'language': 'en-US', 'page': 1},
                timeout=10
            )
            if resp.status_code != 200:
                return
            reviews = resp.json().get('results', [])[:8]  # up to 8 reviews
            if not reviews:
                return

            count = 0
            for review in reviews:
                text = review.get('content', '')
                if len(text) < 50:
                    continue
                author = review.get('author', 'TMDb User')
                review_id = review.get('id', '')
                rating = review.get('author_details', {}).get('rating')

                # Use rating as sentiment hint if available (1-10 scale)
                if rating is not None:
                    if rating >= 7:
                        sent = {'sentiment': 1, 'confidence': min(1.0, (rating - 5) / 5)}
                    elif rating <= 4:
                        sent = {'sentiment': -1, 'confidence': min(1.0, (5 - rating) / 5)}
                    else:
                        sent = {'sentiment': 0, 'confidence': 0.3}
                else:
                    sent = self.sentiment.analyze(text[:1000])

                review_data = {
                    'content_id': content_id,
                    'source': 'tmdb',
                    'source_url': review.get('url', ''),
                    'source_id': f"tmdb_{review_id}",
                    'reviewer': author,
                    'review_text': text[:800],
                    'sentiment': sent['sentiment'],
                    'confidence': sent['confidence'],
                    'weighted_sentiment': sent['sentiment'] * sent['confidence']
                }
                try:
                    self.db.table('reviews').upsert(review_data, on_conflict='source,source_id').execute()
                    count += 1
                except Exception:
                    pass

            if count:
                print(f"   📝 Saved {count} TMDb reviews")
        except Exception as e:
            print(f"   ⚡️ TMDb reviews error: {e}")

    def run(self):
        print("\n" + "="*70)
        print("🚀 WATCH NOW FLOW - Processing Trending with Reviews")
        print("="*70)
        
        # Get global trending content
        spinner = Spinner("Fetching global trending").start()
        trending_all = self.tmdb.get_trending('all', 'week', limit=Config.WATCH_NOW_TRENDING_LIMIT)
        spinner.stop(f"Global trending: {len(trending_all)} titles")
        
        # Get Hindi trending content and merge
        spinner = Spinner("Fetching Hindi trending").start()
        indian_trending = []
        for media_type in ['movie', 'tv']:
            indian_trending.extend(self.tmdb.get_trending_indian(media_type, limit=10))
        spinner.stop(f"Hindi trending: {len(indian_trending)} titles")
        
        # Interleave Hindi titles into global list so they aren't pushed past the per-platform cap.
        # Pattern: every 3rd slot gets a Hindi title → guarantees ~3 Hindi per 10 processed.
        seen_ids = {item['tmdb_id'] for item in trending_all}
        unique_hindi = [i for i in indian_trending if i['tmdb_id'] not in seen_ids]
        
        interleaved = []
        hindi_iter = iter(unique_hindi)
        hindi_slot = 3  # insert a Hindi title every N global titles
        hi_count = 0
        for i, item in enumerate(trending_all):
            interleaved.append(item)
            if (i + 1) % hindi_slot == 0:
                try:
                    interleaved.append(next(hindi_iter))
                    hi_count += 1
                except StopIteration:
                    pass
        # Append any remaining Hindi titles not yet inserted
        for item in hindi_iter:
            interleaved.append(item)
            hi_count += 1
        
        trending_all = interleaved
        print(f"\n📦 Total trending (global + Hindi): {len(trending_all)} titles ({hi_count} Hindi interleaved)")
        
        if not trending_all:
            print("❌ No trending content discovered")
            return
        
        processed = 0
        platform_list = list(Config.PLATFORMS.items())
        total_platforms = len(platform_list)

        # Pre-fetch provider data ONCE per title (instead of once per platform = 5x speedup)
        print(f"\n🔍 Pre-fetching provider availability for {len(trending_all)} titles...")
        provider_cache = {}
        for i, item in enumerate(trending_all):
            progress(i + 1, len(trending_all), item['title'])
            provider_cache[item['tmdb_id']] = self.tmdb.get_watch_providers(
                item['tmdb_id'], item['content_type']
            )
        print()  # newline after progress bar

        # Monkey-patch get_watch_providers to use cache during this run
        original_get_providers = self.tmdb.get_watch_providers
        def cached_get_providers(tmdb_id, media_type, retries=3):
            if tmdb_id in provider_cache:
                return provider_cache[tmdb_id]
            return original_get_providers(tmdb_id, media_type, retries)
        self.tmdb.get_watch_providers = cached_get_providers

        # Skip platforms that have zero matches in the entire trending list — no point iterating
        active_platforms = []
        for platform, platform_id in platform_list:
            matches = sum(1 for item in trending_all
                         if platform_id in (provider_cache.get(item['tmdb_id']) or []))
            if matches > 0:
                active_platforms.append((platform, platform_id))
                print(f"   ✅ {platform}: {matches} potential matches")
            else:
                print(f"   ⏭️  {platform}: 0 matches — skipping entirely")
        print()

        def process_platform(args):
            platform, platform_id = args
            idx = active_platforms.index(args) + 1
            print(f"\n{'='*70}")
            print(f"🎬 {platform.upper()} ({idx}/{len(active_platforms)} active platforms)")
            print(f"{'='*70}")
            confirmed = 0  # count only titles actually on this platform
            # Only iterate titles we know are on this platform (from provider_cache)
            platform_titles = [
                item for item in trending_all
                if platform_id in (provider_cache.get(item['tmdb_id']) or [])
            ]
            for i, content_item in enumerate(platform_titles):
                if confirmed >= Config.WATCH_NOW_MAX_VIDEOS_PER_PLATFORM:
                    break
                print(f"   [{i+1}/{len(platform_titles)}] {content_item['title'][:40]}")
                self.process_trending_content(content_item, platform, platform_id)
                confirmed += 1
            return confirmed

        for platform, platform_id in active_platforms:
            count = process_platform((platform, platform_id))
            processed += count
            print(f"   ✅ {platform} done — {count} items processed")
        
        # Restore original
        self.tmdb.get_watch_providers = original_get_providers
        
        print("\n" + "="*70)
        print(f"✅ WATCH NOW FLOW COMPLETE - {processed} items across {total_platforms} platforms")
        print("="*70)

# ============================================================================
# SCORE COMPUTER
# ============================================================================

class ScoreComputer:
    def __init__(self):
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.scoring = ScoringEngine()
        self.tmdb = TMDbResolver()  
    
    def compute_all(self):
        print("\n" + "="*70)
        print("📊 COMPUTING SCORES")
        print("="*70)

        # Load everything in TWO queries instead of N+1
        spinner = Spinner("Loading content + reviews").start()
        content_result = self.db.table('content').select('*').execute()
        reviews_result = self.db.table('reviews').select(
            'content_id,source,sentiment,confidence,weighted_sentiment'
        ).execute()
        spinner.stop()

        if not content_result.data:
            print("⚡️ No content found")
            return

        # Index reviews by content_id — O(1) lookup per title
        from collections import defaultdict
        reviews_by_id = defaultdict(list)
        for r in (reviews_result.data or []):
            reviews_by_id[r['content_id']].append(r)

        total = len(content_result.data)
        print(f"   {total} titles, {len(reviews_result.data or [])} reviews loaded")

        score_batch = []
        for content in content_result.data:
            content_id = content['id']
            reviews    = reviews_by_id.get(content_id, [])
            if not reviews:
                continue

            yt_reviews  = [r for r in reviews if r['source'] == 'youtube']
            red_reviews = [r for r in reviews if r['source'] == 'reddit']

            yt_score  = (np.mean([r['weighted_sentiment'] for r in yt_reviews]) + 1) * 50 if yt_reviews else 50
            red_score = (np.mean([r['weighted_sentiment'] for r in red_reviews]) + 1) * 50 if red_reviews else 50
            imdb_score = self.scoring.normalize_imdb(content.get('imdb_rating'))
            weights    = self.scoring.get_dynamic_weights(content.get('release_year'))

            if red_reviews:
                final_score = (weights['youtube'] * 0.5 * yt_score +
                               weights['youtube'] * 0.5 * red_score +
                               weights['imdb'] * imdb_score)
            else:
                final_score = weights['youtube'] * yt_score + weights['imdb'] * imdb_score

            sentiments   = [r['sentiment'] for r in reviews]
            imdb_val     = content.get('imdb_rating') or 0
            is_polarizing = (len(sentiments) >= 3 and np.std(sentiments) > 0.7 and imdb_val < 8.0)
            positive_ratio = sum(1 for r in reviews if r['sentiment'] == 1) / len(reviews)
            label = self.scoring.get_label(final_score)
            category = self.scoring.get_category(content.get('release_year'))

            print(f"   🏆 {content['title'][:35]:35} {final_score:.1f} {label}")

            score_batch.append({
                'content_id':    content_id,
                'youtube_score': round(yt_score, 1),
                'reddit_score':  round(red_score, 1),
                'imdb_score':    round(imdb_score, 1),
                'engagement_score': 0.0,
                'final_score':   round(final_score, 1),
                'label':         label,
                'category':      category,
                'review_count':  len(reviews),
                'positive_ratio': round(positive_ratio, 2),
                'is_polarizing': bool(is_polarizing),
                'sentiment_std': round(np.std(sentiments), 2) if len(sentiments) > 1 else 0.0
            })

        # Bulk upsert all scores in one call
        if score_batch:
            try:
                self.db.table('scores').upsert(score_batch, on_conflict='content_id').execute()
                print(f"\n   ✅ {len(score_batch)} scores saved in one shot")
            except Exception as e:
                print(f"   ❌ Bulk score save failed: {e}")

        print("\n" + "="*70)
        print("✅ SCORING COMPLETE")
        print("="*70)
        self.show_top_ranked()
    
    def show_top_ranked(self):
        print(f"\n🏆 TOP RANKED CONTENT")
        print("="*70)
        
        result = self.db.table('scores').select('*, content(title, platform, content_type)').order('final_score', desc=True).limit(10).execute()
        
        if not result.data:
            print("No scores found")
            return
        
        for idx, row in enumerate(result.data, 1):
            content = row['content']
            title = content['title'] if content else 'Unknown'
            platform = content['platform'] if content else 'Unknown'
            ctype = '📺' if content.get('content_type') == 'tv' else '🎬'
            category = row.get('category', 'catalog').upper()
            
            print(f"{idx:2}. {title[:35]:35} | {platform:15} {ctype} | {row['final_score']:5.1f} | {row['label']} | {category}")



# ============================================================================
# ASYNC WATCH NOW PIPELINE  (v2 — maximum speed)
# ============================================================================

class AsyncWatchNowPipeline:
    """
    Maximum-speed Watch Now pipeline.

    Key optimisations vs v1:
    - Semaphore raised to 8 concurrent titles
    - Transcripts disabled (title+description is fast and nearly as good)
    - Sentiment for ALL reviews of a title runs in parallel via thread pool
    - All reviews for a title bulk-upserted in one DB call per source
    - YouTube stats for both videos fetched simultaneously
    - Reddit + Critics + YouTube + TMDb all fire at the same time per title
    - ssl=False on aiohttp kills Mac ConnectionReset(54) entirely
    """

    SEMAPHORE = 12  # titles in flight simultaneously

    # YouTube Data API v3 quota costs (units per call):
    #   search.list   = 100 units  <- the expensive one
    #   videos.list   =   1 unit
    #   channels.list =   1 unit
    # Default daily quota = 10,000 units -> ~98 search calls/day.
    # With 20 titles/run = ~5 full runs before exhaustion.
    YT_SEARCH_COST   = 100
    YT_DAILY_QUOTA   = 9_500   # leave 500 units headroom
    YT_CACHE_TTL_HRS = 23      # re-use results within same calendar day

    def __init__(self):
        self.yt_key    = Config.YOUTUBE_API_KEY
        self.tmdb_key  = Config.TMDB_API_KEY
        self.db        = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.sentiment = SentimentAnalyzer()
        self.scoring   = ScoringEngine()
        self.reddit    = RedditIngester() if Config.USE_REDDIT else None
        self.critic    = CriticReviewScraper()
        self.tmdb      = TMDbResolver()
        self._session  = None
        self._executor = ThreadPoolExecutor(max_workers=32)

        # YouTube quota guard + result cache
        self._yt_quota_used  = 0
        self._yt_quota_blown = False   # True once we get a 403/quota error
        self._yt_cache: dict = {}      # query -> List[video_dict]
        self._load_yt_cache()

    # ── aiohttp helpers ───────────────────────────────────────────────────

    async def _aget(self, url: str, params: dict, retries: int = 3,
                    label: str = "") -> Optional[dict]:
        for attempt in range(retries):
            try:
                async with self._session.get(url, params=params, timeout=12) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status == 429:
                        wait = 2 ** attempt
                        print(f"   ⚡️ Rate-limited {label or url.split('/')[-1]} — waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if r.status in (400, 403, 404):
                        return None
                    if attempt == retries - 1:
                        print(f"   ⚡️ HTTP {r.status} for {label or url.split('/')[-1]}")
                    return None
            except Exception as e:
                if attempt == retries - 1:
                    print(f"   ⚡️ {label or 'request'} failed: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(0.75 * (attempt + 1))
        return None

    # ── Provider prefetch — all titles simultaneously ─────────────────────

    async def _fetch_providers_one(self, item: dict) -> tuple:
        data = await self._aget(
            f"https://api.themoviedb.org/3/{item['content_type']}/{item['tmdb_id']}/watch/providers",
            {'api_key': self.tmdb_key}
        )
        ids = []
        if data:
            india = data.get('results', {}).get('IN', {})
            ids = [p['provider_id'] for p in india.get('flatrate', [])]
        return item['tmdb_id'], ids

    async def _build_provider_cache(self, titles: List[dict]) -> dict:
        print(f"   🚀 Provider check: {len(titles)} titles simultaneously...")
        results = await asyncio.gather(
            *[self._fetch_providers_one(t) for t in titles],
            return_exceptions=True
        )
        return {r[0]: r[1] for r in results if isinstance(r, tuple)}

    # ── YouTube — search + both video stats simultaneously ────────────────

    def _load_yt_cache(self) -> None:
        """Pre-populate cache from reviews saved in last YT_CACHE_TTL_HRS hours.
        Each saved YouTube review row has source_id=video_id and reviewer=channel.
        We reconstruct enough to skip the search call for titles already in DB.
        """
        try:
            cutoff = (datetime.now() - timedelta(hours=self.YT_CACHE_TTL_HRS)).isoformat()
            rows = (self.db.table('reviews')
                       .select('source_id,reviewer,review_text,content_id')
                       .eq('source', 'youtube')
                       .gte('created_at', cutoff)
                       .execute().data or [])
            # Group by content_id -> list of video stubs
            from collections import defaultdict
            by_content: dict = defaultdict(list)
            for r in rows:
                by_content[r['content_id']].append({
                    'video_id': r['source_id'],
                    'title': r['review_text'][:80],
                    'description': '',
                    'channel': r['reviewer'],
                    'channel_id': '',
                    '_cached': True
                })
            # Store by content_id key so _process_youtube can check it
            self._yt_cache_by_content = dict(by_content)
            if rows:
                print(f"   📦 YouTube cache: {len(rows)} reviews loaded ({len(by_content)} titles) — saves quota")
        except Exception as e:
            self._yt_cache_by_content = {}
            print(f"   ⚡️ YouTube cache load failed: {e}")

    async def _yt_search(self, query: str) -> List[dict]:
        """Search YouTube — returns [] immediately if daily quota is blown."""
        if self._yt_quota_blown:
            return []

        # Check if we'd exceed today's budget
        if self._yt_quota_used + self.YT_SEARCH_COST > self.YT_DAILY_QUOTA:
            if not self._yt_quota_blown:
                print(f"   ⚠️  YouTube quota budget reached ({self._yt_quota_used} units used) — skipping remaining YT calls")
                self._yt_quota_blown = True
            return []

        data = await self._aget(
            "https://www.googleapis.com/youtube/v3/search",
            {'part': 'snippet', 'q': query, 'type': 'video',
             'maxResults': 2, 'key': self.yt_key, 'order': 'relevance'},
            label="YouTube search"
        )
        if not data:
            # Check if this looks like a quota 403 (already logged by _aget)
            # Mark quota as blown so we stop hammering the API
            self._yt_quota_blown = True
            print(f"   ⚠️  YouTube search failed — quota likely exhausted. Skipping YT for remaining titles.")
            return []

        self._yt_quota_used += self.YT_SEARCH_COST
        return [{'video_id': i['id']['videoId'],
                 'title': i['snippet']['title'],
                 'description': i['snippet']['description'],
                 'channel': i['snippet']['channelTitle'],
                 'channel_id': i['snippet']['channelId']}
                for i in data.get('items', [])]

    async def _yt_stats_one(self, video_id: str, channel_id: str) -> dict:
        vdata, cdata = await asyncio.gather(
            self._aget("https://www.googleapis.com/youtube/v3/videos",
                       {'part': 'statistics', 'id': video_id, 'key': self.yt_key}),
            self._aget("https://www.googleapis.com/youtube/v3/channels",
                       {'part': 'statistics', 'id': channel_id, 'key': self.yt_key})
        )
        s = {}
        if vdata and vdata.get('items'):
            st = vdata['items'][0].get('statistics', {})
            s.update(views=int(st.get('viewCount',0)),
                     likes=int(st.get('likeCount',0)),
                     comments=int(st.get('commentCount',0)))
        if cdata and cdata.get('items'):
            st = cdata['items'][0].get('statistics', {})
            s['subscribers'] = int(st.get('subscriberCount', 0))
        return s

    async def _process_youtube(self, content_id: int, title: str,
                                platform: str, is_hindi: bool, loop) -> List[dict]:
        """Returns list of review rows — caller does the bulk upsert.
        Checks DB cache first to avoid burning quota on already-seen titles.
        Sentiment uses VADER (instant, no network) for title+description texts.
        """
        # Cache hit — content already has YouTube reviews from today
        if content_id in getattr(self, '_yt_cache_by_content', {}):
            cached = self._yt_cache_by_content[content_id]
            print(f"   📦 YouTube cached ({len(cached)} reviews) — 0 quota used")
            return []   # rows already in DB; no need to re-save

        if self._yt_quota_blown:
            print(f"   ⚡️ YouTube quota exhausted — skipping, saving TMDb/Reddit only")
            return []

        query = f"{title} Hindi review {platform}" if is_hindi else f"{title} {platform} review"
        print(f"   🔍 Searching: {query}")
        videos = await self._yt_search(query)
        if not videos:
            print(f"   ⚡️ YouTube: no results for '{query}'")
            return []

        # Fetch stats for all videos simultaneously
        all_stats = await asyncio.gather(
            *[self._yt_stats_one(v['video_id'], v['channel_id']) for v in videos]
        )

        rows = []
        for video, stats in zip(videos, all_stats):
            text = f"{video['title']} {video['description']}"
            # VADER is pure Python — zero network, runs in microseconds, no executor needed
            sent = self.sentiment._vader_analyze(text)
            yw = self.scoring.youtube_weight(
                stats.get('views', 0), stats.get('subscribers', 0), stats.get('comments', 0))
            print(f"     📺 {video['title'][:50]}...")
            print(f"     💭 Sentiment: {sent['sentiment']} (conf: {sent['confidence']:.2f})")
            rows.append({
                'content_id': content_id, 'source': 'youtube',
                'source_url': f"https://youtube.com/watch?v={video['video_id']}",
                'source_id': video['video_id'], 'reviewer': video['channel'],
                'reviewer_subscribers': stats.get('subscribers', 0),
                'review_text': text[:1000],
                'sentiment': sent['sentiment'], 'confidence': sent['confidence'],
                'views': stats.get('views', 0), 'likes': stats.get('likes', 0),
                'comments_count': stats.get('comments', 0),
                'youtube_weight': yw,
                'weighted_sentiment': sent['sentiment'] * sent['confidence'] * yw
            })
        return rows

    # ── TMDb reviews — parallel sentiment ────────────────────────────────

    async def _process_tmdb_reviews(self, content_id: int, tmdb_id: int,
                                     media_type: str, loop) -> List[dict]:
        data = await self._aget(
            f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/reviews",
            {'api_key': self.tmdb_key, 'language': 'en-US', 'page': 1},
            label="TMDb reviews"
        )
        if not data:
            return []

        reviews = [r for r in data.get('results', [])[:6] if len(r.get('content','')) >= 50]
        if not reviews:
            return []

        rows = []
        for review in reviews:
            rating = review.get('author_details', {}).get('rating')
            if rating is not None:
                # Rating is available — derive sentiment instantly, no network needed
                if rating >= 7:   sent = {'sentiment': 1,  'confidence': min(1.0,(rating-5)/5)}
                elif rating <= 4: sent = {'sentiment': -1, 'confidence': min(1.0,(5-rating)/5)}
                else:             sent = {'sentiment': 0,  'confidence': 0.3}
            else:
                # No rating — VADER on review text, still instant
                sent = self.sentiment._vader_analyze(review['content'][:1000])
            rows.append({
                'content_id': content_id, 'source': 'tmdb',
                'source_url': review.get('url', ''),
                'source_id': f"tmdb_{review.get('id','')}",
                'reviewer': review.get('author', 'TMDb User'),
                'review_text': review['content'][:800],
                'sentiment': sent['sentiment'], 'confidence': sent['confidence'],
                'weighted_sentiment': sent['sentiment'] * sent['confidence']
            })
        return rows

    # ── Reddit + Critics — sync, run in thread pool ───────────────────────

    def _reddit_sync(self, content_id: int, title: str,
                     media_type: str, is_hindi: bool) -> List[dict]:
        if not self.reddit:
            return []
        threads = self.reddit.get_reddit_discussions(title, media_type, is_hindi=is_hindi)
        if not threads:
            return []
        rows = []
        for thread in threads:
            for comment in thread['comments']:
                rows.append({
                    'content_id': content_id, 'source': 'reddit',
                    'source_url': thread['url'],
                    'source_id': f"{thread['url']}_{hash(comment['text'][:10])}",
                    'reviewer': f"r/{thread.get('subreddit','reddit')}",
                    'review_text': comment['text'],
                    'sentiment': comment['sentiment'],
                    'confidence': comment['confidence'],
                    'weighted_sentiment': comment['sentiment'] * comment['confidence']
                })
        return rows

    def _critics_sync(self, content_id: int, title: str,
                      media_type: str, year: Optional[int], is_hindi: bool) -> List[dict]:
        """RT scraper — only runs when --critics flag passed, skipped by default.
        RT scraping costs 8-16s per title (2 retries × 8s timeout) and often
        returns nothing. Enable with: python3 streaming_tracker_v3.py --critics
        """
        if not Config.USE_CRITICS:
            return []
        scrapers = [('rotten_tomatoes', lambda: self.critic.scrape_rotten_tomatoes(title, media_type, year))]
        rows = []
        for source, fn in scrapers:
            try:
                result = fn()
                if not result:
                    continue
                rows.append({
                    'content_id': content_id, 'source': result['source'],
                    'source_url': result['url'],
                    'source_id': f"{result['source']}_{content_id}",
                    'reviewer': result['reviewer'],
                    'review_text': result.get('text','')[:800],
                    'sentiment': result['sentiment'], 'confidence': result['confidence'],
                    'weighted_sentiment': result['sentiment'] * result['confidence']
                })
            except Exception:
                pass
        return rows

    # ── Bulk DB upsert ────────────────────────────────────────────────────

    def _bulk_save_reviews(self, rows: List[dict]) -> None:
        """Upsert all rows in one call per source type."""
        if not rows:
            return
        from collections import defaultdict
        by_source = defaultdict(list)
        for r in rows:
            by_source[r['source']].append(r)
        for source, batch in by_source.items():
            try:
                self.db.table('reviews').upsert(batch, on_conflict='source,source_id').execute()
            except Exception as e:
                # Fall back to one-by-one if bulk fails
                for r in batch:
                    try:
                        self.db.table('reviews').upsert(r, on_conflict='source,source_id').execute()
                    except Exception:
                        pass

    # ── Per-title orchestration ───────────────────────────────────────────

    async def _process_title(self, title_data: dict, platform: str,
                              semaphore, loop) -> None:
        async with semaphore:
            title      = title_data['title']
            tmdb_id    = title_data['tmdb_id']
            media_type = title_data['content_type']
            is_hindi   = title_data.get('is_indian', False)
            year       = title_data.get('release_year')

            print(f"   ⚡ {title[:40]} [{platform}]")

            content_data = {
                'tmdb_id': tmdb_id, 'title': title,
                'original_title': title_data.get('original_title'),
                'platform': platform, 'content_type': media_type,
                'release_year': year, 'imdb_rating': title_data.get('imdb_rating'),
                'poster_path': title_data.get('poster_path'),
                'overview': title_data.get('overview'),
                'discovery_source': title_data.get('category', 'trending')
            }
            try:
                result = self.db.table('content').upsert(
                    content_data, on_conflict='tmdb_id').execute()
                content_id = result.data[0]['id']
            except Exception as e:
                print(f"   ❌ DB error {title}: {e}")
                return

            # All four sources fire simultaneously
            yt_rows, tmdb_rows, reddit_rows, critic_rows = await asyncio.gather(
                self._process_youtube(content_id, title, platform, is_hindi, loop),
                self._process_tmdb_reviews(content_id, tmdb_id, media_type, loop),
                loop.run_in_executor(self._executor, self._reddit_sync,
                                     content_id, title, media_type, is_hindi),
                loop.run_in_executor(self._executor, self._critics_sync,
                                     content_id, title, media_type, year, is_hindi),
                return_exceptions=True
            )

            # Collect and bulk-save all reviews in one shot
            all_rows = []
            for batch in [yt_rows, tmdb_rows, reddit_rows, critic_rows]:
                if isinstance(batch, list):
                    all_rows.extend(batch)

            await loop.run_in_executor(self._executor, self._bulk_save_reviews, all_rows)
            print(f"   ✅ {title[:40]} — {len(all_rows)} reviews saved")

    # ── Main async entry point ────────────────────────────────────────────

    async def _run_async(self) -> None:
        loop = asyncio.get_event_loop()
        import aiohttp
        connector = aiohttp.TCPConnector(ssl=False, limit=50)
        async with aiohttp.ClientSession(connector=connector) as session:
            self._session = session

            # Trending fetch — staggered slightly to avoid TMDb connection resets
            print("   🌐 Fetching trending titles...")
            def _fetch_global():
                return self.tmdb.get_trending('all', 'week', Config.WATCH_NOW_TRENDING_LIMIT)
            def _fetch_hindi_movie():
                time.sleep(0.3)   # stagger to avoid simultaneous connection flood
                return self.tmdb.get_trending_indian('movie', limit=10)
            def _fetch_hindi_tv():
                time.sleep(0.6)
                return self.tmdb.get_trending_indian('tv', limit=10)

            results = await asyncio.gather(
                loop.run_in_executor(self._executor, _fetch_global),
                loop.run_in_executor(self._executor, _fetch_hindi_movie),
                loop.run_in_executor(self._executor, _fetch_hindi_tv),
                return_exceptions=True
            )
            trending_all = results[0] if isinstance(results[0], list) else []
            indian = []
            for r in results[1:]:
                if isinstance(r, list):
                    indian.extend(r)

            seen = {t['tmdb_id'] for t in trending_all}
            unique_hindi = [i for i in indian if i['tmdb_id'] not in seen]
            interleaved, hindi_iter = [], iter(unique_hindi)
            for i, item in enumerate(trending_all):
                interleaved.append(item)
                if (i + 1) % 3 == 0:
                    try: interleaved.append(next(hindi_iter))
                    except StopIteration: pass
            interleaved.extend(hindi_iter)
            trending_all = interleaved
            print(f"   📦 {len(trending_all)} titles")

            # Provider cache — all simultaneously
            provider_cache = await self._build_provider_cache(trending_all)

            # Active platforms
            active_platforms = []
            for platform, pid in Config.PLATFORMS.items():
                matches = sum(1 for t in trending_all if pid in provider_cache.get(t['tmdb_id'], []))
                if matches:
                    active_platforms.append((platform, pid))
                    print(f"   ✅ {platform}: {matches} matches")
                else:
                    print(f"   ⏭️  {platform}: 0 — skipping")

            # Build all tasks across all platforms simultaneously
            semaphore = asyncio.Semaphore(self.SEMAPHORE)
            tasks = []
            seen_title_platform = set()
            for platform, pid in active_platforms:
                platform_titles = [
                    t for t in trending_all
                    if pid in provider_cache.get(t['tmdb_id'], [])
                ][:Config.WATCH_NOW_MAX_VIDEOS_PER_PLATFORM]
                for t in platform_titles:
                    key = (t['tmdb_id'], platform)
                    if key not in seen_title_platform:
                        seen_title_platform.add(key)
                        tasks.append(self._process_title(t, platform, semaphore, loop))

            print(f"\n   🚀 {len(tasks)} jobs, {self.SEMAPHORE} concurrent — go!")
            await asyncio.gather(*tasks, return_exceptions=True)

    def run(self) -> None:
        print("\n" + "="*70)
        print("🚀 WATCH NOW FLOW — Async Pipeline v2")
        print("="*70)
        try:
            asyncio.run(self._run_async())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_async())
        finally:
            self._executor.shutdown(wait=False)
        print("\n" + "="*70)
        print("✅ WATCH NOW FLOW COMPLETE")
        print("="*70)

def cleanup_old_data(days_old=7):
    """Remove content and associated data older than X days"""
    print(f"\n🧹 Cleaning up data older than {days_old} days...")
    db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    
    cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
    
    spinner = Spinner("Scanning for old data").start()
    try:
        # Get old content IDs
        old_content = db.table('content').select('id').lt('created_at', cutoff_date).execute()
        spinner.stop()
        
        if old_content.data:
            old_ids = [item['id'] for item in old_content.data]
            
            # Batch delete by ID list instead of one-by-one
            BATCH = 50
            for i in range(0, len(old_ids), BATCH):
                chunk = old_ids[i:i + BATCH]
                db.table('reviews').delete().in_('content_id', chunk).execute()
                db.table('scores').delete().in_('content_id', chunk).execute()
            
            # Delete content
            db.table('content').delete().lt('created_at', cutoff_date).execute()
            
            print(f"   ✅ Removed {len(old_ids)} old entries")
        else:
            print(f"   ℹ️ No old data to clean")
    except Exception as e:
        print(f"   ⚡️ Cleanup failed: {e}")

# ============================================================================
# MAIN - RUN BOTH FLOWS
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-reddit', action='store_true',
                        help='Disable Reddit')
    parser.add_argument('--no-critics', action='store_true',
                        help='Disable RT critic scraper')
    parser.add_argument('--llm-sentiment', action='store_true',
                        help='Use Groq/Gemini sentiment (slower, marginal gain)')
    args, _ = parser.parse_known_args()

    if args.no_reddit:
        Config.USE_REDDIT = False
        print("📡 Reddit disabled")
    else:
        Config.USE_REDDIT = True
        print("📡 Reddit enabled")
    if args.no_critics:
        Config.USE_CRITICS = False
        print("🍅 RT critics disabled")
    else:
        Config.USE_CRITICS = True
        print("🍅 RT critics enabled")
    if args.llm_sentiment:
        print("🧠 LLM sentiment enabled (will be slower)")

    print("\n" + "="*70)
    print("🎬 STREAMING TRACKER V3.0 - TWO-FLOW SYSTEM")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not Config.TMDB_API_KEY:
        print("❌ Missing TMDB_API_KEY in .env")
        return
    
    if not Config.YOUTUBE_API_KEY:
        print("❌ Missing YOUTUBE_API_KEY in .env")
        return
    
    if os.getenv("OMDB_API_KEY"):
        print("✅ OMDB API key found — RT scores via API (reliable)")
    else:
        print("⚠️  No OMDB_API_KEY — RT scraping will likely be blocked. Get a free key at omdbapi.com")
    
    print("✅ API keys loaded")
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        print("✅ Transcript API available")
    except ImportError:
        print("⚡️ Transcript API not installed")
        Config.USE_TRANSCRIPTS = False
    
    try:
        import feedparser
        print("✅ Feedparser available (Reddit enabled)")
    except ImportError:
        print("⚡️ Feedparser not installed - Reddit disabled")
        Config.USE_REDDIT = False
    
    # Clean up old data
    cleanup_old_data(days_old=7)
    
    # FLOW 1: DISCOVER (No reviews, just availability)
    print("\n🔍 STARTING DISCOVER FLOW...")
    discover = DiscoverFlow()
    discover.save_discover_content()
    
    # FLOW 2: WATCH NOW (With reviews & scoring — async pipeline)
    print("\n📺 STARTING WATCH NOW FLOW...")
    ingester = AsyncWatchNowPipeline()
    ingester.run()
    
    computer = ScoreComputer()
    computer.compute_all()
    
    print("\n" + "="*70)
    print("🎉 BOTH FLOWS COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print("   ✅ Discover content saved (no reviews)")
    print("   ✅ Watch Now content scored (with reviews)")
    print("\nNext: streamlit run dashboard_v3.py")

if __name__ == "__main__":
    main()
