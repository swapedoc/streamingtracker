# streaming_tracker.py - Two-Flow System: Watch Now + Discover
# Run with: python3 streaming_tracker.py

import os
import re
import time
import math
import asyncio
import json
import requests
import numpy as np
import urllib.parse
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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
    }
    
    # Genre IDs for TMDb
    GENRES = {
        # Movie genre IDs
        'Action': 28,
        'Horror': 27,
        'Comedy': 35,
        'Drama': 18,
        'Thriller': 53,
        'Sci-Fi': 878,
        'Romance': 10749,
        # TV-specific genre IDs (TMDb uses different IDs for TV)
        # These map back to the same label as their movie equivalent
        # FIX #18: _tv entries removed — TV genres use _TV_GENRE_MAP directly
    }
    
    # WATCH NOW FLOW (with reviews & scoring)
    WATCH_NOW_TRENDING_LIMIT = 120
    WATCH_NOW_MAX_VIDEOS_PER_PLATFORM = 25
    
    # DISCOVER FLOW (no reviews, just availability)
    DISCOVER_CLASSICS_LIMIT = 120
    DISCOVER_GENRE_LIMIT = 80
    DISCOVER_UNDERDOG_LIMIT = 60
    DISCOVER_ENABLED_GENRES = ['Action', 'Horror', 'Thriller', 'Comedy', 'Drama' ,'Romance', 'Sci-Fi']
    DISCOVER_INDIAN_LIMIT = 120
    
    # All Indian languages — used by DiscoverFlow to categorise content as 'indian'.
    # Broad set so Tamil, Telugu, Malayalam etc. blockbusters appear in Discover
    # (many have Hindi dubs and are perfectly watchable via the Hindi dub filter).
    INDIAN_LANGUAGES = {'hi', 'ta', 'te', 'ml', 'kn', 'bn', 'mr', 'pa'}

    # Watch Now trending fetch is explicitly Hindi-only — non-Hindi content
    # without a dub isn't useful in the scored/reviewed Watch Now feed.
    WATCH_NOW_LANGUAGES = ['hi']
    INDIAN_LANGUAGE_NAMES = {
        'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam',
        'kn': 'Kannada', 'bn': 'Bengali', 'mr': 'Marathi', 'pa': 'Punjabi',
    }
    
    # FIX #17: USE_TRANSCRIPTS removed — always False, never read anywhere
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

        # Build genre_id → label lookup once at construction — avoids lazy-init
        # TOCTOU race when multiple threads call _resolve_genre simultaneously.
        self._id_to_genre: Dict[int, str] = {}
        for label, gid in Config.GENRES.items():
            if '_tv' not in label:
                self._id_to_genre[gid] = label
        for gid, label in self._TV_GENRE_MAP.items():
            self._id_to_genre[gid] = label
    
    def get_trending(self, media_type='all', time_window='week', limit=20) -> List[Dict]:
        """Get trending content from TMDb — supports multi-page fetching for limit > 20"""
        url = f"{self.base_url}/trending/{media_type}/{time_window}"
        pages_needed = math.ceil(limit / 20)
        all_results = []

        for page in range(1, pages_needed + 1):
            params = {'api_key': self.api_key, 'page': page}
            for attempt in range(4):
                try:
                    response = self._session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    page_data = response.json()
                    all_results.extend(page_data.get('results', []))
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"TMDb trending error (page {page}): {e}")
                    else:
                        time.sleep(attempt + 1)

        data = {'results': all_results[:limit]}

        try:
            trending = []
            current_year = datetime.now().year
            WATCH_NOW_MAX_AGE = 3   # exclude titles older than 3 years from Watch Now

            for item in data.get('results', [])[:limit]:
                if item.get('media_type') not in ['movie', 'tv']:
                    continue

                item_media_type = item['media_type']  # FIX #12: was 'media_type' — shadowed method param
                title = item.get('title') if item_media_type == 'movie' else item.get('name')
                release_year = self._extract_year(
                    item.get('release_date') if item_media_type == 'movie' else item.get('first_air_date')
                )

                # Skip old MOVIES — they belong in Discover, not Watch Now
                # TV series are EXEMPT — a show trending now likely has a new season
                if item_media_type == 'movie' and release_year and (current_year - release_year) > WATCH_NOW_MAX_AGE:
                    continue

                trending.append({
                    'tmdb_id': item['id'],
                    'title': title,
                    'original_title': item.get('original_title') or item.get('original_name'),
                    'content_type': item_media_type,  # FIX #12
                    'release_year': release_year,
                    'poster_path': item.get('poster_path'),
                    'overview': item.get('overview'),
                    'popularity': item.get('popularity', 0),
                    'imdb_rating': item.get('vote_average'),
                    'category': 'trending',
                    'genre': self._resolve_genre(item.get('genre_ids', [])),
                    'tv_genre': self._resolve_tv_genre(item.get('genre_ids', [])) if item_media_type == 'tv' else None,
                })

            print(f"✅ Found {len(trending)} trending titles (within {WATCH_NOW_MAX_AGE} years)")
            return trending

        except Exception as e:
            print(f"❌ Error parsing TMDb data: {e}")
            return []
    
    def get_trending_indian(self, media_type='movie', limit=20) -> List[Dict]:
        """
        Get genuinely trending Hindi content for the Watch Now feed.
        Intentionally Hindi-only (original_language=hi) — Watch Now is a scored,
        reviewed feed and non-Hindi content without a dub isn't useful there.
        Use DiscoverFlow for the full Indian-language catalog (all 8 languages).
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
                    title = item.get("title") if media_type == "movie" else item.get("name")
                    release_date = item.get("release_date") if media_type == "movie" else item.get("first_air_date")
                    release_year = self._extract_year(release_date)
                    # Movies older than 3 years go to Discover, not Watch Now
                    # TV is exempt — a show with a new season is always valid
                    if media_type == "movie" and release_year and (current_year - release_year) > 3:
                        continue
                    seen_ids.add(item["id"])
                    all_results.append({
                        "tmdb_id": item["id"],
                        "title": title,
                        "original_title": item.get("original_title") or item.get("original_name"),
                        "content_type": media_type,
                        "release_year": release_year,
                        "poster_path": item.get("poster_path"),
                        "overview": item.get("overview"),
                        "popularity": item.get("popularity", 0),
                        "imdb_rating": item.get("vote_average"),
                        "category": "trending",
                        "genre": self._resolve_genre(item.get("genre_ids", []), fallback="Hindi"),
                        "tv_genre": self._resolve_tv_genre(item.get("genre_ids", [])) if media_type == "tv" else None,
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
            min_date = f"{current_year - 3}-01-01"
            for mt in ([media_type] if media_type in ("movie", "tv") else ["movie", "tv"]):
                if len(all_results) >= limit:
                    break
                discover_params = {
                    "api_key": self.api_key,
                    "sort_by": "popularity.desc",
                    "with_original_language": "hi",
                    "vote_count.gte": 20,
                    "with_watch_providers": "8|119|350|2336",
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
                            "genre": self._resolve_genre(item.get("genre_ids", []), fallback="Hindi"),
                            "tv_genre": self._resolve_tv_genre(item.get("genre_ids", [])) if mt == "tv" else None,
                            "language": "hi",
                            "language_name": "Hindi",
                            "is_indian": True
                        })
                except Exception as e:
                    print(f"   ⚡️ Discover fallback failed ({mt}): {e}")

        print(f"✅ Found {len(all_results)} trending Hindi {media_type} titles")
        return all_results[:limit]

    def _extract_year(self, date_str):
        if date_str:
            try:
                return int(date_str.split('-')[0])
            except (ValueError, AttributeError):
                pass
        return None

    # Priority order: more specific / scarcer genres first so they're not
    # drowned out by Drama or Action which TMDb attaches to almost everything.
    _GENRE_PRIORITY = [
        'Horror', 'Sci-Fi', 'Thriller', 'Romance', 'Comedy', 'Action', 'Drama'
    ]

    # TMDb TV genre IDs → our normalised movie-style label
    # Used by _resolve_genre so films and TV share the same label vocabulary
    _TV_GENRE_MAP = {
        10759: 'Action',    # Action & Adventure → Action
        10765: 'Sci-Fi',    # Sci-Fi & Fantasy   → Sci-Fi
        10766: 'Drama',     # Soap
        10768: 'Drama',     # War & Politics
        10762: 'Comedy',    # Kids
        10763: 'Drama',     # News
        10764: 'Drama',     # Reality
        80:    'Thriller',  # Crime → closest match for Thriller TV
        9648:  'Thriller',  # Mystery → also Thriller
    }

    # TMDb TV genre IDs → human-readable TV label (used for tv_genre field)
    _TV_GENRE_LABELS = {
        10759: 'Action & Adventure',
        10765: 'Sci-Fi & Fantasy',
        35:    'Comedy',
        18:    'Drama',
        80:    'Crime',
        9648:  'Mystery',
        10751: 'Family',
        16:    'Animation',
        99:    'Documentary',
        10766: 'Soap',
        10768: 'War & Politics',
        37:    'Western',
    }

    # Priority for TV genre label selection
    _TV_GENRE_PRIORITY = [
        'Action & Adventure', 'Sci-Fi & Fantasy', 'Crime', 'Mystery',
        'Comedy', 'Drama', 'Animation', 'Family', 'Documentary',
        'Soap', 'War & Politics', 'Western',
    ]

    def _resolve_genre(self, genre_ids: list, fallback: Optional[str] = None) -> Optional[str]:
        """
        Returns a movie-style genre label (Horror, Action, Sci-Fi etc.)
        Works for both movies and TV — TV IDs are mapped via _TV_GENRE_MAP.
        """
        matched = [self._id_to_genre[gid] for gid in (genre_ids or [])
                   if gid in self._id_to_genre]
        if not matched:
            return fallback

        matched = list(dict.fromkeys(matched))  # deduplicate, preserve order

        for preferred in self._GENRE_PRIORITY:
            if preferred in matched:
                return preferred
        return matched[0]

    def _resolve_tv_genre(self, genre_ids: list) -> Optional[str]:
        """
        Returns the best human-readable TMDb TV genre label for a TV show.
        e.g. 'Action & Adventure', 'Crime', 'Sci-Fi & Fantasy', 'Mystery' etc.
        Returns None for movies (caller should only call this for tv content).
        """
        matched = [self._TV_GENRE_LABELS[gid] for gid in (genre_ids or [])
                   if gid in self._TV_GENRE_LABELS]
        if not matched:
            return None

        matched = list(dict.fromkeys(matched))  # deduplicate, preserve order

        for preferred in self._TV_GENRE_PRIORITY:
            if preferred in matched:
                return preferred
        return matched[0]

    def get_runtime_and_trailer(self, tmdb_id: int, media_type: str,
                                title: str = None, year: int = None) -> dict:
        """
        Fetch runtime details + official trailer ID.
        title/year are optional — used only for the YouTube fallback when
        TMDb /videos returns nothing.
        """
        result = {
            'runtime': None, 'seasons': None,
            'episode_count': None, 'episode_runtime': None,
            'trailer_id': None,
        }
        mt = 'movie' if media_type != 'tv' else 'tv'

        # ── Details (runtime / seasons) ───────────────────────────────────
        try:
            r = self._session.get(
                f"{self.base_url}/{mt}/{tmdb_id}",
                params={'api_key': self.api_key},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if mt == 'movie':
                    result['runtime'] = data.get('runtime') or None
                else:
                    result['seasons']       = data.get('number_of_seasons') or None
                    result['episode_count'] = data.get('number_of_episodes') or None
                    ep_rt = None
                    er = [x for x in (data.get('episode_run_time') or []) if x and x > 0]
                    if er:
                        ep_rt = er[0]
                    if not ep_rt:
                        last = data.get('last_episode_to_air') or {}
                        ep_rt = last.get('runtime') or None
                    if not ep_rt:
                        nxt = data.get('next_episode_to_air') or {}
                        ep_rt = nxt.get('runtime') or None
                    result['episode_runtime'] = ep_rt
        except Exception as e:
            print(f"   ⚡️ TMDb runtime fetch error: {e}")

        # ── Official trailer via TMDb /videos (free, no YouTube quota) ────
        try:
            r = self._session.get(
                f"{self.base_url}/{mt}/{tmdb_id}/videos",
                params={'api_key': self.api_key, 'language': 'en-US'},
                timeout=10,
            )
            if r.status_code == 200:
                videos = r.json().get('results', [])
                for vtype in ('Trailer', 'Teaser'):
                    for v in videos:
                        if (v.get('site') == 'YouTube'
                                and v.get('type') == vtype
                                and v.get('official', True)):
                            result['trailer_id'] = v['key']
                            break
                    if result['trailer_id']:
                        break

            # If nothing found in English, try the Hindi-language endpoint
            # (many Bollywood/Indian titles only have trailers filed under 'hi')
            if not result['trailer_id']:
                r2 = self._session.get(
                    f"{self.base_url}/{mt}/{tmdb_id}/videos",
                    params={'api_key': self.api_key, 'language': 'hi'},
                    timeout=10,
                )
                if r2.status_code == 200:
                    videos2 = r2.json().get('results', [])
                    for vtype in ('Trailer', 'Teaser'):
                        for v in videos2:
                            if v.get('site') == 'YouTube' and v.get('type') == vtype:
                                result['trailer_id'] = v['key']
                                break
                        if result['trailer_id']:
                            break
        except Exception as e:
            print(f"   ⚡️ TMDb trailer fetch error: {e}")

        # ── YouTube fallback — only fires when TMDb has no trailer ────────
        if not result['trailer_id'] and title:
            yt_id = self._youtube_trailer_fallback(title, year)
            if yt_id:
                result['trailer_id']   = yt_id
                result['_yt_fallback'] = True  # stripped before DB write

        return result

    _TRUSTED_CHANNEL_NAMES = {
        # Global streamers / studios
        'netflix', 'prime video', 'amazon prime', 'apple tv', 'disney',
        'hotstar', 'sony pictures', 'warner bros', 'universal pictures',
        'paramount', 'lionsgate', 'a24', 'marvel', 'dc', 'mgm', 'mubi',
        'ign', 'filmspot trailer', 'hulu', 'hbo',
        # Indian studios & distributors
        'zee studios', 'zee music company',
        'yash raj films',
        'dharma productions',
        't-series',
        'pen movies', 'pen marudhar',
        'eros now', 'eros movies',
        'jio studios',
        'excel entertainment',
        'maddock films',
        'tips films', 'tips official',
        'red chillies entertainment',
        'balaji motion pictures',
        'viacom18 studios', 'viacom18',
        'saregama music',
        'sony music india',
        'reliance entertainment',
        'gulshan kumar', 'bhushan kumar',
    }
    _TRUSTED_CHANNEL_IDS = {
        'UCWX3yGbOBE3mMSBSCVDzK4g', 'UCTOxLBzMBCEFTEMF7DqhAsg',
        'UC_IRUbMCnBQh5X5hTLjWLpA', 'UCmEDwvvN9LiRh0dMjB8RWLA',
        'UCzWQYUVCpZqtN93H8RR44Qw', 'UCi8e0iOVk1fEOogdfu4YgfA',
        'UCF9imwbTCaZCOcuZnWTFBkA', 'UCvC4D8onUfXzvjTOM-dBfEA',
        'UCjmJDM5pRKbUlVIzDYwz-1A', 'UCaw03IoN618hT5-bXu6LoAA',
        'UCYLbpjXO5BwstZXhBpqECRg', 'UCgMPP6RejnQEoEX-bwOFZLA',
        'UCR_Gp53bEfFTL2W6VLMJ15g', 'UCFFbwnve3yF62-tVXkTyHqg',
        'UC9zY_E8mcAo_Oq772LEZq8Q', 'UCiEEF51uRAeZeCo8CJFhGWw',
    }

    _yt_fallback_calls = 0          # class-level counter shared across all instances
    _yt_fallback_limit = 30         # max trailer fallback searches per process run
    _yt_fallback_lock  = threading.Lock()

    def _youtube_trailer_fallback(self, title: str, year: int = None):
        """Search YouTube for an official trailer from a trusted channel only.
        Returns None rather than risk returning a fan-made or unofficial video.
        Capped at _yt_fallback_limit calls per process run to protect daily quota —
        trailer searches for 879 discover rows would cost 87,900 units otherwise.

        Two-pass strategy, both require a trusted channel:
          Pass 1: query with year — precise, avoids wrong-film hits
          Pass 2: query without year — catches titles with wrong/missing year
        No fallback to untrusted channels. No trailer is better than a fan trailer.
        """
        yt_key = Config.YOUTUBE_API_KEY
        if not yt_key:
            return None

        # Hard cap — protect daily quota from being consumed by trailer searches
        with TMDbResolver._yt_fallback_lock:
            if TMDbResolver._yt_fallback_calls >= TMDbResolver._yt_fallback_limit:
                return None   # silent — no print spam for every skipped title
            TMDbResolver._yt_fallback_calls += 1
            current = TMDbResolver._yt_fallback_calls
        if current == TMDbResolver._yt_fallback_limit:
            print(f"   ⚠️  YouTube trailer fallback cap reached ({TMDbResolver._yt_fallback_limit} searches) — "
                  f"skipping remaining. Increase TMDbResolver._yt_fallback_limit if needed.")

        REJECT = {
            'reaction', 'review', 'fan made', 'fan-made', 'fan trailer',
            'fan concept', 'concept trailer', 'breakdown', 'explained',
            'analysis', 'ranked', 'every scene', 'deleted',
            'behind the scenes', 'making of', 'interview',
            'featurette', 'clip', 'scene', 'spoiler', 'pitch meeting',
        }

        # Build query list — with year first for precision, then without
        queries = []
        if year:
            queries.append(f"{title} {year} official trailer")
        queries.append(f"{title} official trailer")

        try:
            for query in queries:
                r = self._session.get(
                    'https://www.googleapis.com/youtube/v3/search',
                    params={
                        'part': 'snippet', 'q': query, 'type': 'video',
                        'maxResults': 8, 'order': 'relevance', 'key': yt_key,
                    },
                    timeout=10,
                )
                if r.status_code == 403:
                    return None   # quota exhausted — stop immediately
                if not r.ok:
                    continue

                for item in r.json().get('items', []):
                    vid_id     = item.get('id', {}).get('videoId', '')
                    snippet    = item.get('snippet', {})
                    # Use original-case title for the reject/trailer check after lowercasing
                    vid_title  = snippet.get('title', '').lower()
                    channel    = snippet.get('channelTitle', '').lower()
                    channel_id = snippet.get('channelId', '')

                    if not vid_id:
                        continue
                    # Must look like a trailer or teaser
                    if 'trailer' not in vid_title and 'teaser' not in vid_title:
                        continue
                    # Must not be noise/fan content
                    if any(w in vid_title for w in REJECT):
                        continue
                    # Must be from a trusted channel — no exceptions
                    if not (any(t in channel for t in self._TRUSTED_CHANNEL_NAMES)
                            or channel_id in self._TRUSTED_CHANNEL_IDS):
                        continue
                    return vid_id

            return None   # nothing trusted found — return None, not garbage

        except Exception as e:
            print(f"   ⚡️ YouTube trailer fallback error: {e}")  # FIX #10
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
        jobs = []
        PAGES_PER_TYPE = 4   # 4 pages × 20 results = 80 per media_type per category

        # ── Classics ──────────────────────────────────────────────────────
        for mt in ['movie', 'tv']:
            base = {
                'api_key': self.api_key,
                'sort_by': 'vote_average.desc',
                'vote_average.gte': 8.0,
                'vote_count.gte': 500,
                'with_watch_providers': '8|119|350|2336',
                'watch_region': 'IN',
            }
            for page in range(1, PAGES_PER_TYPE + 1):
                jobs.append({'mt': mt, 'params': {**base, 'page': page},
                             'category': 'classics', 'genre': None})

        # ── Genres ────────────────────────────────────────────────────────
        # TMDb uses different genre IDs for TV vs movies.
        # e.g. Action movie=28, Action TV=10759 (Action & Adventure)
        #      Sci-Fi movie=878, Sci-Fi TV=10765 (Sci-Fi & Fantasy)
        # All others (Horror=27, Thriller=53, Comedy=35, Drama=18, Romance=10749)
        # are either the same ID or have no direct TV equivalent so we use movie ID.
        TV_GENRE_ID_OVERRIDE = {
            'Action': 10759,   # Action & Adventure
            'Sci-Fi': 10765,   # Sci-Fi & Fantasy
        }
        for genre_name in Config.DISCOVER_ENABLED_GENRES:
            genre_id = Config.GENRES.get(genre_name)
            if not genre_id:
                continue
            for mt in ['movie', 'tv']:
                # Use TV-specific genre ID if available, otherwise fall back to movie ID
                effective_genre_id = TV_GENRE_ID_OVERRIDE.get(genre_name, genre_id) if mt == 'tv' else genre_id
                base = {
                    'api_key': self.api_key,
                    'with_genres': effective_genre_id,
                    'sort_by': 'popularity.desc',
                    'vote_average.gte': 6.0 if mt == 'tv' else 6.5,
                    'vote_count.gte':   50  if mt == 'tv' else 100,
                    'with_watch_providers': '8|119|350|2336',
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
                'with_watch_providers': '8|119|350|2336',
                'watch_region': 'IN',
            }
            for page in range(1, 3):   # 2 pages × 2 types = 60 gems max
                jobs.append({'mt': mt, 'params': {**base, 'page': page},
                             'category': 'underdog', 'genre': None})

        # ── Indian / Hindi ─────────────────────────────────────────────────
        # Fetch all Indian-language content for Discover. Users can then filter
        # by Hindi dub to narrow to what's actually watchable without subtitles.
        min_date = f"{datetime.now().year - 3}-01-01"
        for lang in Config.INDIAN_LANGUAGES:
            for mt in ['movie', 'tv']:
                base = {
                    'api_key': self.api_key,
                    'with_original_language': lang,
                    'sort_by': 'popularity.desc',
                    'vote_count.gte': 20,
                    'with_watch_providers': '8|119|350|2336',
                    'watch_region': 'IN',
                }
                if mt == 'movie':
                    base['primary_release_date.gte'] = min_date
                else:
                    base['air_date.gte'] = min_date
                for page in range(1, PAGES_PER_TYPE + 1):
                    jobs.append({'mt': mt, 'params': {**base, 'page': page},
                                 'category': 'indian', 'genre': 'Hindi',
                                 'language': lang})

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
                            # Resolve genre from the item's actual genre_ids, not
                            # from which query bucket fetched it. job['genre'] is
                            # only used as a fallback (e.g. 'Hindi' for Indian bucket).
                            resolved_genre = self.tmdb._resolve_genre(
                                item.get('genre_ids', []),
                                fallback=job['genre']
                            )
                            # For TV shows, also store the native TMDb TV genre label
                            resolved_tv_genre = (
                                self.tmdb._resolve_tv_genre(item.get('genre_ids', []))
                                if mt == 'tv' else None
                            )
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
                                'genre': resolved_genre,
                                'tv_genre': resolved_tv_genre,
                            }
                            if job['category'] == 'indian':
                                parsed['language'] = item.get('original_language') or job.get('language', 'hi')
                            items.append(parsed)
                        return items
                except Exception as e:
                    print(f"   ⚡️ TMDb discover page error (attempt {attempt+1}): {e}")  # FIX #10
                    if attempt < 2:
                        await asyncio.sleep(0.75 * (attempt + 1))
            return []

    async def _fetch_all_async(self, jobs: List[Dict]) -> List[Dict]:
        """
        Fire all discover-page fetches AND provider checks concurrently in one
        aiohttp session — eliminates the slow per-item sync thread-pool entirely.
        """
        import aiohttp
        # ── Phase 1: fetch all /discover pages ────────────────────────────
        page_sem = asyncio.Semaphore(8)
        connector = aiohttp.TCPConnector(ssl=False, limit=20)

        done_count = 0
        total_jobs = len(jobs)

        async def _fetch_with_progress(session, job, semaphore):
            nonlocal done_count
            try:
                result = await asyncio.wait_for(
                    self._fetch_page(session, job, semaphore),
                    timeout=20
                )
            except asyncio.TimeoutError:
                print(f"   ⏱️  Timeout: {job.get('category','?')} page {job.get('page','?')}")
                result = []
            except Exception as e:
                result = []
            done_count += 1
            if done_count % 10 == 0 or done_count == total_jobs:
                print(f"   📄 Pages: {done_count}/{total_jobs} done...")
            return result

        async with aiohttp.ClientSession(connector=connector) as session:
            page_tasks = [_fetch_with_progress(session, job, page_sem) for job in jobs]
            page_results = await asyncio.gather(*page_tasks, return_exceptions=True)

            raw_items: List[Dict] = []
            for r in page_results:
                if isinstance(r, list):
                    raw_items.extend(r)

            print(f"   ✅ Pages done — {len(raw_items)} raw items collected")

            # Deduplicate per (tmdb_id, category) before provider checks
            seen_keys: set = set()
            unique_items: List[Dict] = []
            for item in raw_items:
                key = (item['tmdb_id'], item['category'])
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_items.append(item)

            # ── Phase 2: provider checks — all async, same session ─────────
            tmdb_id_to_items: Dict[int, List[Dict]] = {}
            for item in unique_items:
                tmdb_id_to_items.setdefault(item['tmdb_id'], []).append(item)

            provider_sem = asyncio.Semaphore(40)
            prov_done = 0
            total_prov = len(tmdb_id_to_items)

            async def fetch_providers(tmdb_id: int, content_type: str) -> tuple:
                nonlocal prov_done
                url = (f"https://api.themoviedb.org/3/{content_type}"
                       f"/{tmdb_id}/watch/providers")
                params = {'api_key': self.api_key}
                for attempt in range(3):
                    try:
                        async with provider_sem:
                            async with session.get(url, params=params, timeout=10) as resp:
                                if resp.status == 429:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                if resp.status != 200:
                                    prov_done += 1
                                    return tmdb_id, []
                                data = await resp.json()
                                india = data.get('results', {}).get('IN', {})
                                ids = [p['provider_id']
                                       for p in india.get('flatrate', [])]
                                prov_done += 1
                                if prov_done % 50 == 0 or prov_done == total_prov:
                                    print(f"   🔍 Providers: {prov_done}/{total_prov}...")
                                return tmdb_id, ids
                    except Exception as e:
                        print(f"   ⚡️ Provider fetch error for tmdb_id {tmdb_id} (attempt {attempt+1}): {e}")
                        if attempt < 2:
                            await asyncio.sleep(0.5 * (attempt + 1))
                prov_done += 1
                return tmdb_id, []

            unique_tmdb = list(tmdb_id_to_items.items())
            print(f"\n   🔍 Checking providers for {total_prov} unique titles...")

            provider_tasks = [
                fetch_providers(tid, items[0]['content_type'])
                for tid, items in unique_tmdb
            ]
            provider_results = await asyncio.gather(*provider_tasks, return_exceptions=True)
            print(f"   ✅ Providers done")

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
        jobs = self._build_jobs()
        print(f"   🚀 Firing {len(jobs)} TMDb page requests concurrently...")

        expanded = asyncio.run(self._fetch_all_async(jobs))

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

        # Deduplicate by (tmdb_id, platform).
        # Priority: classics=0, underdog=1, indian=2, genre_*=3
        # BUT for genre buckets: if the resolved genre matches the bucket, that's
        # a better match than a genre bucket that fetched it incidentally.
        # e.g. House MD in genre_comedy bucket but resolved genre=Drama → deprioritise.
        CATEGORY_RANK = {'classics': 0, 'underdog': 1, 'indian': 2}

        def _item_rank(item):
            cat = item['category']
            base = CATEGORY_RANK.get(cat, 3)
            if cat.startswith('genre_'):
                # Extract bucket genre label e.g. 'genre_horror' -> 'Horror'
                bucket_genre = cat.replace('genre_', '').capitalize()
                # Sci-Fi special case
                if bucket_genre == 'Sci-fi':
                    bucket_genre = 'Sci-Fi'
                resolved = item.get('genre') or ''
                # Penalise if the resolved genre doesn't match the bucket
                if resolved.lower() != bucket_genre.lower():
                    base = 10  # effectively last priority
            return base

        seen_pairs: dict = {}
        for item, platform in expanded:
            key = (item['tmdb_id'], platform)
            rank = _item_rank(item)
            if key not in seen_pairs or rank < _item_rank(seen_pairs[key][0]):
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
                'tv_genre':        item.get('tv_genre'),
                'popularity':      item['popularity'],
                'source':          'tracker',
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

        # ── Stale row cleanup ──────────────────────────────────────────────
        # Delete any discover_content rows NOT in this run's fresh fetch.
        # These are titles no longer available on that platform in India
        # (expired licences, removed content, etc.).
        # Safety guard: skip if fetch returned suspiciously few titles —
        # avoids wiping the table on a partial API failure.
        print("\n   🧹 Cleaning up stale Discover rows...")
        try:
            fresh_keys = {(r['tmdb_id'], r['platform']) for r in rows}

            if len(fresh_keys) < 50:
                print("   ⚠️  Fetch returned very few titles — skipping stale cleanup as safety measure")
            else:
                existing = []
                for _start in range(0, 100_000, 1000):
                    _page = self.db.table('discover_content') \
                        .select('id, tmdb_id, platform, source') \
                        .range(_start, _start + 999).execute()
                    if not _page.data:
                        break
                    existing.extend(_page.data)
                    if len(_page.data) < 1000:
                        break

                stale_ids = [
                    row['id'] for row in existing
                    if (row['tmdb_id'], row['platform']) not in fresh_keys
                    and row.get('source', 'tracker') == 'tracker'  # never delete bulk catalog rows
                ]

                if not stale_ids:
                    print("   ✅ No stale rows found")
                else:
                    for i in range(0, len(stale_ids), BATCH):
                        chunk = stale_ids[i:i + BATCH]
                        self.db.table('discover_content').delete() \
                            .in_('id', chunk).execute()
                    print(f"   🗑️  Removed {len(stale_ids)} stale rows "
                          f"(titles no longer available on their platform)")
        except Exception as e:
            print(f"   ⚠️  Stale cleanup failed (non-fatal): {e}")

        print("="*70)
# ============================================================================
# SENTIMENT ANALYSIS - 3-TIER CASCADE SYSTEM
# ============================================================================

class SentimentAnalyzer:
    # Class-level lock: serialize ALL Groq calls across threads to avoid rate-limit storms.
    # With 12 concurrent titles each calling Groq, all 12 hit the API simultaneously — instant 429.
    # Serialising them with a lock means Groq gets one call at a time.
    _groq_lock = threading.Lock()
    _groq_rate_limit_until: float = 0.0   # shared across all instances
    _gemini_rate_limit_until: float = 0.0  # shared across all instances

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
            if SentimentAnalyzer._groq_rate_limit_until > time.time():
                pass  # still in cooldown, skip to Gemini silently
            else:
                result = self._groq_analyze(text)
                if result:
                    return result
                # silent fallback to Gemini/VADER
        # Try Tier 2: Gemini
        if self.use_gemini:
            result = self._gemini_analyze(text)
            if result:
                return result
        
        # Tier 3: VADER (Always works)
        return self._vader_analyze(text)
    
    def _groq_analyze(self, text: str) -> Optional[Dict]:
        try:
            with self._groq_lock:   # serialize across threads — avoid simultaneous 429s
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
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                SentimentAnalyzer._groq_rate_limit_until = time.time() + 60
                print(f"  ⚡️ Groq rate limited — cooling down 60s, trying Gemini...")
            elif 'Expecting value' in err or 'JSONDecodeError' in err or len(err.strip()) == 0:
                pass  # empty response from Groq — silent fallback to Gemini, not an error
            else:
                print(f"  ⚡️ Groq error: {err[:100]}")
            return None
    
    def _gemini_analyze(self, text: str) -> Optional[Dict]:
        if SentimentAnalyzer._gemini_rate_limit_until > time.time():
            return None  # still in cooldown
        try:
            prompt = f"""Analyze the sentiment of this movie/TV review.

Review: {text[:4000]}

Return ONLY a JSON object with this exact format:
{{"sentiment": -1 or 0 or 1, "confidence": 0.0 to 1.0}}

Where:
- sentiment: -1 (negative), 0 (neutral), 1 (positive)
- confidence: how certain you are (0.0 = unsure, 1.0 = very certain)"""

            from concurrent.futures import ThreadPoolExecutor as _GTE, TimeoutError as _GTO
            with _GTE(max_workers=1) as _gex:
                _gfut = _gex.submit(self.gemini_client.models.generate_content,
                                    model='gemini-2.0-flash', contents=prompt)
                try:
                    response = _gfut.result(timeout=15)
                except _GTO:
                    print(f"  ⚡️ Gemini timed out (15s) — falling back to VADER")
                    return None

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
                SentimentAnalyzer._gemini_rate_limit_until = time.time() + 60
                print(f"  ⚡️ Gemini rate limited — cooling down 60s...")
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

    # ── Vibe label map — genre → (metric name, emoji) ────────────────────
    VIBE_LABELS = {
        'Horror':   ('Scare Factor',  '🔪'),
        'Thriller': ('Tension Meter', '⚡'),
        'Action':   ('Adrenaline',    '💥'),
        'Comedy':   ('Laugh Meter',   '😂'),
        'Sci-Fi':   ('Mind-Bend',     '🌀'),
        'Romance':  ('Heart Score',   '💕'),
        'Drama':    ('Emotional Hit', '🎭'),
    }
    VIBE_DEFAULT = ('Vibe Score', '✨')

    def vibe_label_for(self, genre: str) -> tuple:
        """Return (metric_name, emoji) for a given genre."""
        return self.VIBE_LABELS.get(genre or '', self.VIBE_DEFAULT)

    def vibe_analyze(self, review_texts: list, genre: str) -> Optional[Dict]:
        """
        Analyse a list of review snippets and return a genre-specific vibe score 1-10.
        Tries Groq first (fast, timeout=10s), falls back to Gemini.
        Returns None on any failure — never blocks the main score save.

        Returns: {'vibe_score': float, 'vibe_label': str} or None
        """
        if not review_texts:
            return None

        metric, _ = self.vibe_label_for(genre)
        # Combine up to 8 reviews. review_text is now stored full-length (no DB truncation),
        # but we still cap the LLM prompt at 3000 chars to stay within token limits.
        combined = '\n\n'.join(t[:400] for t in review_texts[:8])
        combined = combined[:3000]

        prompt = f"""You are a film critic analysing audience reviews.

Genre: {genre or 'General'}
Metric: {metric}

Reviews:
{combined}

Based ONLY on what the reviews say, rate the "{metric}" of this title on a scale of 1 to 10.
- 1 = Almost none / very weak
- 5 = Moderate / average
- 10 = Extreme / outstanding

Return ONLY a JSON object, no extra text:
{{"vibe_score": <number 1-10>, "reasoning": "<one sentence>"}}"""

        # Try Groq first
        if self.use_groq and SentimentAnalyzer._groq_rate_limit_until <= time.time():
            try:
                with self._groq_lock:  # serialize vibe calls same as _groq_analyze
                    response = self.groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=80,
                        timeout=10,
                    )
                raw = response.choices[0].message.content.strip()
                raw = raw.replace('```json', '').replace('```', '').strip()
                result = json.loads(raw)
                score = float(result.get('vibe_score', 0))
                if 1 <= score <= 10:
                    return {'vibe_score': round(score, 1), 'vibe_label': metric}
            except Exception as e:
                if '429' in str(e) or 'rate' in str(e).lower():
                    SentimentAnalyzer._groq_rate_limit_until = time.time() + 60
                # Non-fatal — fall through to Gemini

        # Try Gemini
        if self.use_gemini and SentimentAnalyzer._gemini_rate_limit_until <= time.time():
            try:
                from concurrent.futures import ThreadPoolExecutor as _VTE, TimeoutError as _VTO
                with _VTE(max_workers=1) as _vex:
                    _vfut = _vex.submit(self.gemini_client.models.generate_content,
                                        model='gemini-2.0-flash', contents=prompt)
                    try:
                        response = _vfut.result(timeout=15)
                    except _VTO:
                        print(f"  ⚡️ Gemini vibe timed out (15s)")
                        raise Exception("timeout")
                raw = response.text.strip().replace('```json', '').replace('```', '').strip()
                result = json.loads(raw)
                score = float(result.get('vibe_score', 0))
                if 1 <= score <= 10:
                    return {'vibe_score': round(score, 1), 'vibe_label': metric}
            except Exception as e:
                if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                    SentimentAnalyzer._gemini_rate_limit_until = time.time() + 60
                # Non-fatal — return None

        return None  # Both tiers failed — score row still saves fine without vibe

# ============================================================================
# REDDIT INGESTER
# ============================================================================

class RedditIngester:
    # OAuth2 authenticated — 998 req/window vs 10 unauthenticated.
    # Token is fetched once per process and reused (expires in 24h).
    _rate_lock = threading.Lock()
    _token_lock = threading.Lock()          # guards _oauth_token + _token_expires_at
    _last_call_time: float = 0.0
    _oauth_token: Optional[str] = None          # shared across all instances
    _token_expires_at: float = 0.0

    # OAuth base — must use oauth.reddit.com when sending Bearer token
    _OAUTH_BASE = "https://oauth.reddit.com"
    _TOKEN_URL  = "https://www.reddit.com/api/v1/access_token"

    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self._session = requests.Session()
        self._TIMEOUT = (4, 8)
        self.noise_phrases = [
            'edit:', 'source:', 'just here', 'this is the way',
            'lol', 'lmao', 'haha', '^', 'deleted', 'removed'
        ]
        # Fetch OAuth token (or reuse cached one)
        token = self._get_oauth_token()
        if token:
            self._session.headers.update({
                'Authorization': f'Bearer {token}',
                'User-Agent': 'python:streamiq.scraper:v1.0 (by /u/stream_tracker_bot)',
            })
            print("   🔑 Reddit OAuth active (998 req/window)")
        else:
            self._session.headers.update({
                'User-Agent': 'python:streamiq.scraper:v1.0 (by /u/stream_tracker_bot)',
            })
            print("   ⚠️  Reddit OAuth unavailable — falling back to unauthenticated")
        # Pre-load which content_ids already have fresh Reddit reviews
        self._load_reddit_cache()

    REDDIT_CACHE_TTL_HRS = 23   # skip re-scraping if reviews saved within last 23h

    def _load_reddit_cache(self) -> None:
        """
        Load content_ids that already have Reddit reviews saved in the last 23h.
        Mirrors the YT cache pattern — avoids hammering r/bollywood on every run.
        """
        try:
            cutoff = (datetime.now() - timedelta(hours=self.REDDIT_CACHE_TTL_HRS)).isoformat()
            rows = (self.db.table('reviews')
                       .select('content_id')
                       .eq('source', 'reddit')
                       .gte('created_at', cutoff)
                       .execute().data or [])
            self._reddit_cached_ids = {r['content_id'] for r in rows}
            if self._reddit_cached_ids:
                print(f"   📦 Reddit cache: {len(self._reddit_cached_ids)} titles already have fresh reviews — will skip")
        except Exception as e:
            self._reddit_cached_ids = set()
            print(f"   ⚡️ Reddit cache load failed: {e}")

    @classmethod
    def _get_oauth_token(cls) -> Optional[str]:
        """
        Fetch a Reddit OAuth2 token using client_credentials grant.
        Result is cached at the class level so it's fetched once per run,
        not once per title. Token is valid for 24h (86400s).
        Thread-safe via double-checked locking on _token_lock.
        """
        # Fast path — token already valid, no lock needed
        if cls._oauth_token and time.time() < cls._token_expires_at - 60:
            return cls._oauth_token

        # Slow path — acquire lock then re-check in case another thread
        # already refreshed the token while we were waiting
        with cls._token_lock:
            if cls._oauth_token and time.time() < cls._token_expires_at - 60:
                return cls._oauth_token

            client_id     = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            username      = os.getenv('REDDIT_USERNAME', '')
            password      = os.getenv('REDDIT_PASSWORD', '')

            if not client_id or not client_secret:
                return None

            ua = 'python:streamiq.scraper:v1.0 (by /u/stream_tracker_bot)'
            payload = (
                {'grant_type': 'password', 'username': username, 'password': password}
                if username and password
                else {'grant_type': 'client_credentials'}
            )
            try:
                r = requests.post(
                    cls._TOKEN_URL,
                    data=payload,
                    auth=(client_id, client_secret),
                    headers={'User-Agent': ua},
                    timeout=10,
                )
                if r.status_code == 200 and 'access_token' in r.json():
                    data = r.json()
                    cls._oauth_token      = data['access_token']
                    cls._token_expires_at = time.time() + data.get('expires_in', 86400)
                    return cls._oauth_token
                else:
                    print(f"   ⚠️  Reddit token fetch failed: HTTP {r.status_code} — {r.text[:100]}")
            except Exception as e:
                print(f"   ⚠️  Reddit token fetch error: {e}")
            return None

    def _api_url(self, path: str) -> str:
        """
        Return the correct base URL.
        oauth.reddit.com MUST be used when an Authorization header is present —
        sending a Bearer token to www.reddit.com returns 403.
        Falls back to www.reddit.com if not authenticated.
        """
        if RedditIngester._oauth_token:
            return f"{self._OAUTH_BASE}{path}"
        return f"https://www.reddit.com{path}"

    def _throttle(self):
        """
        Authenticated apps get ~1 req/sec sustained (600 req/10min).
        0.15s gap is safe; we use 0.2s to be polite.
        Unauthenticated fallback keeps the original 1.5s gap.
        """
        gap_needed = 0.2 if RedditIngester._oauth_token else 1.5  # FIX NEW-C: 0.1 exceeds Reddit rate limit
        with RedditIngester._rate_lock:
            gap = time.time() - RedditIngester._last_call_time
            if gap < gap_needed:
                time.sleep(gap_needed - gap)
            RedditIngester._last_call_time = time.time()

    def _get_show_subreddit(self, title: str) -> Optional[str]:
        """
        Try to find a show-specific subreddit by checking r/<slug>/about.json.
        Uses public www.reddit.com (not oauth) — about.json is publicly readable
        and oauth.reddit.com can return 0 subscribers for valid subs under
        client_credentials grant.
        Results cached so each title is only looked up once per run.
        """
        if not hasattr(self, '_sub_cache'):
            self._sub_cache = {}
        if title in self._sub_cache:
            return self._sub_cache[title]

        clean        = re.sub(r"['\-:,\.!?&]", '', title)
        slug_nospace = re.sub(r'\s+', '', clean)       # StrangerThings
        slug_under   = re.sub(r'\s+', '_', clean)      # Stranger_Things

        # Try both slug variants — always use public URL, never oauth for this check
        for slug in dict.fromkeys([slug_nospace, slug_under]):
            try:
                # Must use plain requests (not self._session) — self._session has
                # Authorization: Bearer which www.reddit.com rejects with 403.
                # about.json is publicly readable without OAuth.
                r = requests.get(
                    f"https://www.reddit.com/r/{slug}/about.json",
                    headers={"User-Agent": "python:streamiq.scraper:v1.0"},
                    timeout=4,
                )
                if r.status_code == 200:
                    subs = r.json().get('data', {}).get('subscribers', 0)
                    if subs > 500:   # lowered from 1000 — valid small show subs exist
                        self._sub_cache[title] = slug
                        return slug
            except Exception as e:
                print(f"   ⚡️ Subreddit slug lookup error for '{title}': {e}")

        self._sub_cache[title] = None
        return None

    def _get_subreddits(self, title: str, media_type: str, is_hindi: bool,
                        check_show_sub: bool = True) -> List[str]:
        """Return ordered list of subreddits to search, best signal first."""
        title_lower = title.lower()
        # Anime detection — check for common anime title patterns
        anime_keywords = ['kaisen', 'jujutsu', 'demon slayer', 'attack on titan',
                          'one piece', 'naruto', 'dragon ball', 'my hero', 'chainsaw',
                          'frieren', 'spy x', 'vinland', 'bleach', 'hunter x']
        is_anime = any(k in title_lower for k in anime_keywords)

        # Try to find a show-specific subreddit first (best signal)
        # Skipped when check_show_sub=False (quota-blown fast path) — saves 2 API calls
        if media_type == 'tv' and check_show_sub:
            print(f"     🔎 Checking for r/{re.sub(r'[^a-zA-Z0-9]', '', title)} subreddit...", end=' ', flush=True)
            show_sub = self._get_show_subreddit(title)
            print("found ✅" if show_sub else "not found")
        else:
            show_sub = None

        if is_anime:
            base = ['anime', 'Animesuggest', 'television', 'movies']
        elif is_hindi:
            base = ['bollywood', 'india', 'HindiMovies',
                    'television' if media_type == 'tv' else 'movies']
        elif media_type == 'tv':
            base = ['television', 'NetflixBestOf', 'PrimeVideo', 'Jiohotstar',
                    'TrueFilm', 'binge']
        else:
            base = ['movies', 'TrueFilm', 'worldcinema', 'MovieSuggestions',
                    'NetflixBestOf', 'PrimeVideo']

        # Prepend show-specific sub if found — it has the richest discussion
        if show_sub and show_sub not in base:
            return [show_sub] + base
        return base

    def _get_queries(self, title: str) -> List[str]:
        """Multiple queries — fallback ensures we find something even for short/generic titles.
        Normalise ALLCAPS titles (e.g. JUJUTSU KAISEN → Jujutsu Kaisen) so Reddit search
        actually matches threads, since Reddit threads use normal capitalisation.
        """
        # Normalise: ALLCAPS → Title Case, leave mixed-case as-is
        title_norm = title.title() if title.isupper() else title
        queries = [f"{title_norm} review", f"{title_norm} discussion"]
        # Add exact-match only if title is short enough to be unambiguous
        if len(title_norm) > 4:
            queries.append(f'"{title_norm}"')
        # Also try original title as final fallback if normalised is different
        if title_norm != title:
            queries.append(title)
        return queries

    def _search_subreddit(self, subreddit: str, query: str, limit: int = 5) -> List[Dict]:
        """
        Search a subreddit. Tries JSON API first (richer, more reliable),
        falls back to RSS if JSON search is blocked or returns nothing.
        """
        self._throttle()

        # ── Primary: JSON search ──────────────────────────────────────────
        url = self._api_url(f"/r/{subreddit}/search.json")
        params = {'q': query, 'restrict_sr': 'on', 'sort': 'relevance', 't': 'all', 'limit': limit}
        for attempt in range(2):
            try:
                resp = self._session.get(url, params=params, timeout=self._TIMEOUT)
                if resp.status_code == 429:
                    time.sleep(4)
                    continue
                if resp.status_code != 200:
                    # Visible error — tells us if we're getting 403 (blocked) vs empty
                    print(f" [HTTP {resp.status_code}]", end="")
                    return []
                posts = []
                for child in resp.json().get('data', {}).get('children', [])[:limit]:
                    p = child.get('data', {})
                    if p.get('id') and p.get('permalink'):
                        posts.append({'data': {
                            'id':        p['id'],
                            'title':     p.get('title', ''),
                            'permalink': p.get('permalink', ''),
                            'score':     p.get('score', 0),
                        }})
                return posts
            except Exception as e:
                print(f"   ⚡️ Reddit search error: {e}")
                time.sleep(1)
        return []

    def _extract_comments(self, thread_id: str) -> List[Dict]:
        """Fetch top comments for a thread via JSON API."""
        self._throttle()
        url = self._api_url(f"/comments/{thread_id}.json")
        params = {'limit': 10, 'depth': 1, 'sort': 'top'}
        for attempt in range(2):
            try:
                resp = self._session.get(url, params=params, timeout=self._TIMEOUT)
                if resp.status_code == 429:
                    time.sleep(4)
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    if len(data) >= 2:
                        return self._parse_comment_json(data[1])
                break
            except Exception as e:
                print(f"   ⚡️ Reddit comment fetch error: {e}")
                time.sleep(1)
        return []

    def _parse_comment_json(self, comments_data: dict) -> List[Dict]:
        """Parse comments from Reddit JSON response."""
        extracted = []
        for comment in comments_data.get('data', {}).get('children', [])[:8]:
            c_data = comment.get('data', {})
            body = c_data.get('body', '')
            if (body and body not in ('[deleted]', '[removed]')
                    and len(body) > 30
                    and not any(p in body.lower() for p in self.noise_phrases)):
                sent = self.sentiment.analyze(body)
                extracted.append({
                    'text': body,
                    'sentiment': sent['sentiment'],
                    'confidence': sent['confidence'],
                    'score': c_data.get('score', 0)
                })
        return extracted

    def get_reddit_discussions(self, title: str, media_type: str,
                               is_hindi: bool = False,
                               max_subs: int = 3,
                               content_id: Optional[int] = None) -> List[Dict]:
        """
        Search up to max_subs subreddits with up to 3 query fallbacks.
        Stops as soon as 4 threads with comments are found.
        Falls back to an all-of-reddit search if subreddit searches yield nothing.
        """
        # Cache hit — skip entirely if we already saved fresh reviews for this title
        if content_id and content_id in getattr(self, '_reddit_cached_ids', set()):
            print(f"     📦 Reddit cached — skipping search")
            return []
        subreddits = self._get_subreddits(title, media_type, is_hindi,
                                          check_show_sub=(max_subs >= 3))[:max_subs]
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
                posts = self._search_subreddit(subreddit, query, limit=5)
                if not posts:
                    continue  # try next query fallback only if this one returned nothing
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
                if sub_found:
                    break  # got threads from this subreddit — no need to try more queries

            print(f"{sub_found} threads" if sub_found else "none")

        # ── All-of-Reddit fallback when subreddit searches return nothing ──
        if not all_threads:
            title_norm = title.title() if title.isupper() else title
            fallback_query = f"{title_norm} review"
            print(f"     🔍 Fallback: all-of-Reddit search for '{fallback_query}'...", end=' ', flush=True)
            self._throttle()
            try:
                url = self._api_url("/search.json")
                resp = self._session.get(url, params={
                    'q': fallback_query, 'sort': 'relevance', 't': 'year', 'limit': 5
                }, timeout=self._TIMEOUT)
                if resp.status_code == 200:
                    results = resp.json().get('data', {}).get('children', [])
                    fallback_found = 0
                    for item in results:
                        if len(all_threads) >= 4:
                            break
                        d = item.get('data', {})
                        permalink = d.get('permalink', '')
                        if '/comments/' not in permalink:
                            continue
                        thread_id = permalink.split('/comments/')[1].split('/')[0]
                        if thread_id in seen_ids:
                            continue
                        thread_title = d.get('title', '')
                        skip_keywords = ['trailer', 'ama', 'casting', 'renewed', 'cancelled', 'official']
                        if any(k in thread_title.lower() for k in skip_keywords):
                            continue
                        seen_ids.add(thread_id)
                        comments = self._extract_comments(thread_id)
                        if comments:
                            all_threads.append({
                                'title': thread_title,
                                'url': f"https://www.reddit.com{permalink}",
                                'subreddit': d.get('subreddit', 'reddit'),
                                'comments': comments,
                                'upvotes': d.get('score', 0)
                            })
                            fallback_found += 1
                    print(f"{fallback_found} threads" if fallback_found else "none")
            except Exception as e:
                print(f"error: {e}")

        total_comments = sum(len(t['comments']) for t in all_threads)
        if all_threads:
            print(f"     ✅ {len(all_threads)} threads, {total_comments} comments across subreddits")
        else:
            print(f"     ⚡️ No Reddit discussions found")
        return all_threads

    # FIX #20: compute_reddit_score() removed — dead code
    # FIX #20: save_reddit_reviews() removed — dead code
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
    # FIX #19: _ddg_find_url() removed — dead code, zero callers
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
                data = json.loads(script.string or '')
                rating = data.get('aggregateRating', {})
                if rating:
                    val = self._safe_float(str(rating.get('ratingValue', '')))
                    if val:
                        tomatometer = int(val * 10) if val <= 10 else int(val)
                        break
            except Exception as e:
                print(f"   ⚡️ RT score parse error: {e}")

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
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    # FIX #21: fetch_all() removed — dead code, zero callers. _critics_sync() used instead.

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
        if rating is None or rating <= 0:
            return 50
        return max(0, min(100, (rating - 5) * 20))
    
    @staticmethod
    def get_dynamic_weights(release_year: int) -> Dict:
        """Dynamic weights based on content age (Recency Decay).
        youtube + reddit + imdb always sum to 1.0.
        When reddit data is absent the caller uses youtube + imdb only,
        scaling them back to 1.0 proportionally.
        """
        current_year = datetime.now().year
        age = current_year - release_year if release_year else 10

        # RT weight is carved from imdb (both are professional/critic signals).
        # All four weights sum to 1.0.
        # When RT is absent the caller rescales the remaining three proportionally.
        if age <= 1:
            return {'youtube': 0.45, 'reddit': 0.20, 'rt': 0.10, 'imdb': 0.25}
        elif age <= 3:
            return {'youtube': 0.35, 'reddit': 0.15, 'rt': 0.10, 'imdb': 0.40}
        elif age <= 5:
            return {'youtube': 0.28, 'reddit': 0.12, 'rt': 0.10, 'imdb': 0.50}
        else:
            return {'youtube': 0.21, 'reddit': 0.09, 'rt': 0.10, 'imdb': 0.60}
    
    @staticmethod
    def get_category(release_year: int) -> str:
        """Categorize content as Trending or Catalog"""
        current_year = datetime.now().year
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
# SCORE COMPUTER
# ============================================================================

class ScoreComputer:
    def __init__(self):
        self.db        = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.scoring   = ScoringEngine()
        self.tmdb      = TMDbResolver()
        self.sentiment = SentimentAnalyzer()   # for vibe_analyze
    
    def compute_all(self):
        print("\n" + "="*70)
        print("📊 COMPUTING SCORES")
        print("="*70)

        # Load everything in TWO queries instead of N+1
        spinner = Spinner("Loading content + reviews").start()
        # Paginate content too — same Supabase 1000 row cap
        all_content = []
        page = 0
        while True:
            batch = self.db.table('content').select('*').range(page * 1000, (page + 1) * 1000 - 1).execute()
            if not batch.data:
                break
            all_content.extend(batch.data)
            if len(batch.data) < 1000:
                break
            page += 1
        class _ContentResult:
            def __init__(self, data): self.data = data
        content_result = _ContentResult(all_content)

        # Paginate reviews — Supabase silently caps at 1000 rows without this
        all_reviews = []
        PAGE_SIZE = 1000
        page = 0
        while True:
            batch = self.db.table('reviews').select(
                'content_id,source,sentiment,confidence,weighted_sentiment,review_text'
            ).range(page * PAGE_SIZE, (page + 1) * PAGE_SIZE - 1).execute()
            if not batch.data:
                break
            all_reviews.extend(batch.data)
            if len(batch.data) < PAGE_SIZE:
                break
            page += 1

        class _ReviewsResult:
            def __init__(self, data): self.data = data
        reviews_result = _ReviewsResult(all_reviews)
        spinner.stop()
        print(f"   📦 {len(all_reviews)} reviews loaded across {page + 1} page(s)")

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
        current_year = datetime.now().year

        # FIX NEW-F: pre-compute vibe scores in parallel (was sequential blocking Groq calls)
        from concurrent.futures import ThreadPoolExecutor as _VTPE
        def _compute_vibe_one(content):
            cid   = content['id']
            genre = content.get('genre') or ''
            texts = [r['review_text'] for r in reviews_by_id.get(cid, []) if r.get('review_text')]
            return cid, self.sentiment.vibe_analyze(texts, genre)

        # Pre-filter: only score content that will actually pass the loop below.
        # Avoids burning Groq/Gemini calls on old movies that get skipped anyway.
        scorable = [
            c for c in content_result.data
            if not (
                c.get('content_type') == 'movie'
                and (c.get('release_year') or 0)
                and (current_year - c['release_year']) > 3
            )
        ]

        vibe_map: dict = {}
        if scorable:
            with _VTPE(max_workers=4) as _vp:
                for cid, vibe in _vp.map(_compute_vibe_one, scorable):
                    vibe_map[cid] = vibe

        for content in content_result.data:
            content_id = content['id']
            # Safety net — skip old movies even if they made it into DB
            release_year = content.get('release_year') or 0
            if content.get('content_type') == 'movie' and release_year and (current_year - release_year) > 3:
                continue
            reviews    = reviews_by_id.get(content_id, [])
            if not reviews:
                continue

            yt_reviews  = [r for r in reviews if r['source'] == 'youtube']
            red_reviews = [r for r in reviews if r['source'] == 'reddit']
            rt_reviews  = [r for r in reviews if r['source'] == 'rotten_tomatoes']

            yt_score   = (np.mean([r['weighted_sentiment'] for r in yt_reviews]) + 1) * 50 if yt_reviews else 50
            red_score  = (np.mean([r['weighted_sentiment'] for r in red_reviews]) + 1) * 50 if red_reviews else 50
            rt_score   = (np.mean([r['weighted_sentiment'] for r in rt_reviews])  + 1) * 50 if rt_reviews else 50
            imdb_score = self.scoring.normalize_imdb(content.get('imdb_rating'))
            weights    = self.scoring.get_dynamic_weights(content.get('release_year'))

            has_reddit = bool(red_reviews)
            has_rt     = bool(rt_reviews)

            if has_reddit and has_rt:
                # All four sources present — full formula
                final_score = (weights['youtube'] * yt_score +
                               weights['reddit']  * red_score +
                               weights['rt']      * rt_score  +
                               weights['imdb']    * imdb_score)
            elif has_reddit and not has_rt:
                # No RT — rescale youtube + reddit + imdb proportionally
                drop   = weights['rt']
                scale  = 1.0 / (1.0 - drop)
                final_score = (weights['youtube'] * scale * yt_score +
                               weights['reddit']  * scale * red_score +
                               weights['imdb']    * scale * imdb_score)
            elif has_rt and not has_reddit:
                # No Reddit — rescale youtube + rt + imdb proportionally
                drop   = weights['reddit']
                scale  = 1.0 / (1.0 - drop)
                final_score = (weights['youtube'] * scale * yt_score +
                               weights['rt']      * scale * rt_score  +
                               weights['imdb']    * scale * imdb_score)
            else:
                # Neither Reddit nor RT — rescale youtube + imdb proportionally
                yt_w   = weights['youtube'] / (weights['youtube'] + weights['imdb'])
                imdb_w = weights['imdb']    / (weights['youtube'] + weights['imdb'])
                final_score = yt_w * yt_score + imdb_w * imdb_score

            sentiments   = [r['sentiment'] for r in reviews]
            imdb_val     = content.get('imdb_rating') or 0
            is_polarizing = (len(sentiments) >= 3 and np.std(sentiments) > 0.7 and imdb_val < 8.0)
            positive_ratio = sum(1 for r in reviews if r['sentiment'] == 1) / len(reviews)
            label    = self.scoring.get_label(final_score)
            category = self.scoring.get_category(content.get('release_year'))

            # ── Vibe score — looked up from pre-computed parallel batch (FIX NEW-F) ──
            vibe       = vibe_map.get(content_id)
            vibe_score = vibe['vibe_score'] if vibe else None
            vibe_label = vibe['vibe_label'] if vibe else None

            vibe_note = f" | {vibe_label}: {vibe_score}/10" if vibe else ''
            print(f"   🏆 {content['title'][:35]:35} {final_score:.1f} {label}{vibe_note}")

            score_row = {
                'content_id':      content_id,
                'youtube_score':   round(yt_score, 1),
                'reddit_score':    round(red_score, 1),
                'imdb_score':      round(imdb_score, 1),
                'engagement_score': round(rt_score, 1),  # repurposed: stores RT score (was always 0)
                'final_score':     round(final_score, 1),
                'label':           label,
                'category':        category,
                'review_count':    len(reviews),
                'positive_ratio':  round(positive_ratio, 2),
                'is_polarizing':   bool(is_polarizing),
                'sentiment_std':   round(np.std(sentiments), 2) if len(sentiments) > 1 else 0.0,
            }
            # Only write vibe columns when LLM succeeded — never overwrite a good
            # score with None just because this run's API timed out
            if vibe_score is not None:
                score_row['vibe_score'] = vibe_score
                score_row['vibe_label'] = vibe_label

            score_batch.append(score_row)

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
        # FIX NEW-B: per-thread TMDbResolver — executor threads must not share a Session
        self._tmdb_local = threading.local()

        # YouTube quota guard + result cache
        self._yt_quota_used  = 0
        self._yt_quota_blown = False   # True once we get a 403/quota error
        self._yt_cache: dict = {}      # query -> List[video_dict]
        self._load_yt_cache()

    # ── aiohttp helpers ───────────────────────────────────────────────────

    async def _aget(self, url: str, params: dict, retries: int = 3,
                    label: str = "", return_status: bool = False) -> Optional[dict]:
        """
        Async GET helper.
        If return_status=True, returns (data_or_None, http_status_int) tuple.
        status=0 means a network/exception failure (not an HTTP error).
        """
        last_status = 0
        for attempt in range(retries):
            try:
                async with self._session.get(url, params=params, timeout=12) as r:
                    last_status = r.status
                    if r.status == 200:
                        data = await r.json()
                        return (data, 200) if return_status else data
                    if r.status == 429:
                        wait = 2 ** attempt
                        print(f"   ⚡️ Rate-limited {label or url.split('/')[-1]} — waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if r.status == 403:
                        # Only log if quota not already blown — avoids spam when
                        # multiple concurrent titles all hit 403 simultaneously
                        if label and not getattr(self, '_yt_quota_blown', False):
                            print(f"   🚫 HTTP 403 for {label} — API key invalid or quota exhausted")
                        return (None, 403) if return_status else None
                    if r.status in (400, 404):
                        return (None, r.status) if return_status else None
                    # Other non-200: log on last attempt
                    if attempt == retries - 1:
                        print(f"   ⚡️ HTTP {r.status} for {label or url.split('/')[-1]}")
                    return (None, r.status) if return_status else None
            except Exception as e:
                last_status = 0
                if attempt == retries - 1:
                    print(f"   ⚡️ {label or 'request'} failed: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(0.75 * (attempt + 1))
        return (None, last_status) if return_status else None

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
                       .gte('created_at', cutoff)  # reviews table only has created_at (confirmed live schema)
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
        """
        Search YouTube — returns [] immediately if quota is blown or key is invalid.
        Uses return_status=True on _aget to distinguish 403 (key/quota dead) from
        transient network errors. Caps to 3 concurrent searches via semaphore.
        """
        if self._yt_quota_blown:
            return []

        if self._yt_quota_used + self.YT_SEARCH_COST > self.YT_DAILY_QUOTA:
            if not self._yt_quota_blown:
                print(f"   ⚠️  YouTube quota budget reached ({self._yt_quota_used} units used) — disabling YouTube for this run")
                self._yt_quota_blown = True
            return []

        # FIX NEW-5: semaphore initialized in _run_async before tasks start
        async with self._yt_search_sem:
            data, status = await self._aget(
                "https://www.googleapis.com/youtube/v3/search",
                {'part': 'snippet', 'q': query, 'type': 'video',
                 'maxResults': 3, 'key': self.yt_key, 'order': 'relevance'},
                label="YouTube search",
                return_status=True
            )

        if data is None:
            if status == 403:
                # Key invalid or quota exhausted — print ONCE then silence all future messages
                if not self._yt_quota_blown:
                    self._yt_quota_blown = True
                    print(
                        f"   🚫 YouTube API returned 403 — key invalid or daily quota exhausted.\n"
                        f"      Check quota at: console.cloud.google.com/apis/api/youtube.googleapis.com/quotas\n"
                        f"      Disabling YouTube for this run — TMDb + Reddit reviews still collected."
                    )
                # else: already blown and already printed — stay silent
            elif status == 0:
                # Pure network failure (exception) — track consecutive failures
                self._yt_fail_count = getattr(self, '_yt_fail_count', 0) + 1
                if self._yt_fail_count >= 3:
                    self._yt_quota_blown = True
                    print(f"   🚫 YouTube: {self._yt_fail_count} consecutive network failures — disabling for this run.")
                else:
                    print(f"   ⚡️ YouTube network error ({self._yt_fail_count}/3) — query: '{query}'")
            else:
                print(f"   ⚡️ YouTube HTTP {status} — query: '{query}'")
            return []

        # Success — reset failure counter and track quota spend
        self._yt_fail_count = 0
        self._yt_quota_used += self.YT_SEARCH_COST
        return [{'video_id': i['id']['videoId'],
                 'title': i['snippet']['title'],
                 'description': i['snippet']['description'],
                 'channel': i['snippet']['channelTitle'],
                 'channel_id': i['snippet']['channelId']}
                for i in data.get('items', [])
                if i.get('id', {}).get('videoId')]

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

    async def _process_youtube(self, content_id: int, tmdb_id: int, title: str,
                                platform: str, is_hindi: bool) -> List[dict]:
        """Returns list of review rows — caller does the bulk upsert.
        Checks DB cache first to avoid burning quota on already-seen titles.
        Sentiment uses VADER (instant, no network) for title+description texts.
        """
        # ── In-run YouTube dedup — keyed by tmdb_id, not (tmdb_id, platform) ──
        # The same title can appear on multiple platforms (e.g. JUJUTSU KAISEN on
        # Netflix AND Jiohotstar). All platforms share the same content_id (upsert
        # on tmdb_id), so we only need ONE YouTube search per unique title per run.
        if not hasattr(self, '_yt_searched_tmdb'):
            self._yt_searched_tmdb = set()
        if tmdb_id in self._yt_searched_tmdb:
            print(f"   📦 YouTube already searched this run — skipping duplicate (saves 100 quota units)")
            return []

        # Claim this tmdb_id slot NOW — before any await — so any sibling coroutine
        # (same title on a second platform) that resumes while we're awaiting below
        # sees it in the set and skips immediately. Without this, both coroutines pass
        # the check above before either adds to the set, firing two searches.
        self._yt_searched_tmdb.add(tmdb_id)

        # Cache hit — content already has YouTube reviews from a previous run today
        if content_id in getattr(self, '_yt_cache_by_content', {}):
            cached = self._yt_cache_by_content[content_id]
            print(f"   📦 YouTube cached ({len(cached)} reviews) — 0 quota used")
            return []

        if self._yt_quota_blown:
            return []  # already announced once — no need to repeat per title

        # Normalise title to Title Case for better YouTube matching
        # e.g. "JUJUTSU KAISEN" → "Jujutsu Kaisen" matches more video titles
        title_normalised = title.title() if title.isupper() else title

        # Progressive query fallbacks — broad to specific so we always find something
        # Avoids: "Dhurandhar Hindi review Netflix" finding nothing because the video
        # is titled "Dhurandhar Review | Netflix" without the exact word order.
        if is_hindi:
            queries = [
                f"{title_normalised} Hindi review {platform}",
                f"{title_normalised} review Hindi",
                f"{title_normalised} movie review",
                f"{title_normalised} review",
            ]
        else:
            queries = [
                f"{title_normalised} {platform} review",
                f"{title_normalised} review",
                f"{title_normalised} series review" if "season" in title_normalised.lower() else f"{title_normalised} film review",
            ]

        videos = []
        used_query = None
        for q in queries:
            if self._yt_quota_blown:
                break
            print(f"   🔍 Searching: {q}")
            videos = await self._yt_search(q)
            if videos:
                used_query = q
                break  # found results — stop trying fallbacks
            # Small delay between fallback attempts to avoid burst
            await asyncio.sleep(0.3)

        if not videos:
            # _yt_search already printed the failure reason — no need for another message
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
                'review_text': text,
                'sentiment': sent['sentiment'], 'confidence': sent['confidence'],
                'views': stats.get('views', 0), 'likes': stats.get('likes', 0),
                'comments_count': stats.get('comments', 0),
                'youtube_weight': yw,
                'weighted_sentiment': sent['sentiment'] * sent['confidence'] * yw
            })
        # tmdb_id was already registered at the top of this function
        return rows

    # ── TMDb reviews — parallel sentiment ────────────────────────────────

    async def _process_tmdb_reviews(self, content_id: int, tmdb_id: int,
                                     media_type: str) -> List[dict]:
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
                'review_text': review['content'],
                'sentiment': sent['sentiment'], 'confidence': sent['confidence'],
                'weighted_sentiment': sent['sentiment'] * sent['confidence']
            })
        return rows

    # ── Reddit + Critics — sync, run in thread pool ───────────────────────

    def _reddit_sync(self, content_id: int, title: str,
                     media_type: str, is_hindi: bool) -> List[dict]:
        if not self.reddit:
            return []
        # If YouTube quota is already blown this run, Reddit is the primary source
        # so we still run it — but cap at 2 subreddits max to keep it snappy.
        max_subs = 2 if getattr(self, '_yt_quota_blown', False) else 3
        threads = self.reddit.get_reddit_discussions(
            title, media_type, is_hindi=is_hindi, max_subs=max_subs,
            content_id=content_id
        )
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
        returns nothing. Enable with: python3 streaming_tracker.py --critics
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
                    'review_text': result.get('text', ''),
                    'sentiment': result['sentiment'], 'confidence': result['confidence'],
                    'weighted_sentiment': result['sentiment'] * result['confidence']
                })
            except Exception as e:
                print(f"   ⚡️ Critics sync row build error: {e}")  # FIX #10
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
            for attempt in range(3):
                try:
                    self.db.table('reviews').upsert(batch, on_conflict='source,source_id').execute()
                    break
                except Exception as e:
                    err = str(e).lower()
                    is_conn = any(x in err for x in ('disconnect','connection','reset','timeout'))
                    if attempt < 2 and is_conn:
                        time.sleep(2 * (attempt + 1))
                        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
                    else:
                        for r in batch:
                            for r_att in range(3):
                                try:
                                    self.db.table('reviews').upsert(r, on_conflict='source,source_id').execute()
                                    break
                                except Exception as re:
                                    if r_att < 2 and any(x in str(re).lower() for x in ('disconnect','connection','reset')):
                                        time.sleep(1.5 * (r_att + 1))
                                        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
                                    else:
                                        print(f"   ⚡️ Review DB upsert error: {re}")
                        break  # FIX #10

    # ── Per-title orchestration ───────────────────────────────────────────

    def _get_thread_tmdb(self) -> 'TMDbResolver':
        """Return a per-thread TMDbResolver so executor threads never share a Session."""
        if not hasattr(self._tmdb_local, 'instance'):
            self._tmdb_local.instance = TMDbResolver()
        return self._tmdb_local.instance

    async def _process_title(self, title_data: dict, platform: str,
                              semaphore) -> None:
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
                'discovery_source': title_data.get('category', 'trending'),
                'genre': title_data.get('genre'),
                'tv_genre': title_data.get('tv_genre'),
                'updated_at': datetime.now(timezone.utc).isoformat(),  # FIX: utcnow() deprecated
            }
            try:
                result = self.db.table('content').upsert(
                    content_data, on_conflict='tmdb_id,platform').execute()  # FIX NEW-4
                content_id = result.data[0]['id']
            except Exception as e:
                print(f"   ❌ DB error {title}: {e}")
                return

            # All sources + runtime/trailer details fire simultaneously
            yt_rows, tmdb_rows, reddit_rows, critic_rows, details = await asyncio.gather(
                self._process_youtube(content_id, tmdb_id, title, platform, is_hindi),
                self._process_tmdb_reviews(content_id, tmdb_id, media_type),
                asyncio.get_running_loop().run_in_executor(self._executor, self._reddit_sync,
                                     content_id, title, media_type, is_hindi),
                asyncio.get_running_loop().run_in_executor(self._executor, self._critics_sync,
                                     content_id, title, media_type, year, is_hindi),
                asyncio.get_running_loop().run_in_executor(self._executor,
                                     # FIX NEW-B: per-thread resolver
                                     lambda: self._get_thread_tmdb().get_runtime_and_trailer(
                                         tmdb_id, media_type, title, year)),
                return_exceptions=True
            )

            # Patch content row with runtime + trailer if we got them
            if isinstance(details, dict) and any(v is not None for v in details.values()):
                try:
                    yt_fallback = details.pop('_yt_fallback', False)
                    self.db.table('content').update(details).eq('id', content_id).execute()
                    trailer_note = ''
                    if details.get('trailer_id'):
                        src = '🔍YT' if yt_fallback else '🎬TMDb'
                        trailer_note = f" {src}={details['trailer_id'][:8]}.."
                    runtime_note = ''
                    if details.get('runtime'):
                        h, m = divmod(details['runtime'], 60)
                        runtime_note = f" ⏱ {h}h{m:02d}m"
                    elif details.get('seasons'):
                        runtime_note = f" ⏱ {details['seasons']}s"
                    if runtime_note or trailer_note:
                        print(f"   📐 {title[:35]}{runtime_note}{trailer_note}")
                except Exception as e:
                    print(f"   ⚠️  Details save failed for {title[:30]}: {e}")

            # Collect and bulk-save all reviews in one shot
            all_rows = []
            for batch in [yt_rows, tmdb_rows, reddit_rows, critic_rows]:
                if isinstance(batch, list):
                    all_rows.extend(batch)

            await asyncio.get_running_loop().run_in_executor(self._executor, self._bulk_save_reviews, all_rows)
            print(f"   ✅ {title[:40]} — {len(all_rows)} reviews saved")

    # ── Main async entry point ────────────────────────────────────────────

    async def _run_async(self) -> None:
        loop = asyncio.get_running_loop()
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
                asyncio.get_running_loop().run_in_executor(self._executor, _fetch_global),
                asyncio.get_running_loop().run_in_executor(self._executor, _fetch_hindi_movie),
                asyncio.get_running_loop().run_in_executor(self._executor, _fetch_hindi_tv),
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
            self._yt_search_sem = asyncio.Semaphore(3)  # FIX NEW-5: shared
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
                        tasks.append(self._process_title(t, platform, semaphore))

            print(f"\n   🚀 {len(tasks)} jobs, {self.SEMAPHORE} concurrent — go!")
            await asyncio.gather(*tasks, return_exceptions=True)

    def run(self) -> None:
        print("\n" + "="*70)
        print("🚀 WATCH NOW FLOW — Async Pipeline v2")
        print("="*70)
        try:
            asyncio.run(self._run_async())
        finally:
            self._executor.shutdown(wait=False)
        print("\n" + "="*70)
        print("✅ WATCH NOW FLOW COMPLETE")
        print("="*70)

def cleanup_old_data(days_old=7):
    """
    Remove Watch Now content (movies + TV) not seen in the last `days_old` days.
    Both types are re-fetched fresh on every run, so deleting by age is safe —
    anything still trending will be re-inserted immediately on the next run.
    Deletes reviews and scores via ID list (precise) not by date (imprecise).
    """
    print(f"\n🧹 Cleaning up Watch Now data older than {days_old} days...")
    db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

    cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

    spinner = Spinner("Scanning for old data").start()
    try:
        # Collect ALL old content (movies + TV) — both are refreshed every run
        all_old = []
        _page = 0
        while True:
            batch = db.table('content').select('id') \
                .lt('created_at', cutoff_date) \
                .range(_page * 1000, (_page + 1) * 1000 - 1).execute()
            if not batch.data:
                break
            all_old.extend(batch.data)
            if len(batch.data) < 1000:
                break
            _page += 1
        spinner.stop()

        if all_old:
            old_ids = [item['id'] for item in all_old]

            # Delete reviews + scores by ID list — precise, no date ambiguity
            BATCH = 50
            for i in range(0, len(old_ids), BATCH):
                chunk = old_ids[i:i + BATCH]
                db.table('reviews').delete().in_('content_id', chunk).execute()
                db.table('scores').delete().in_('content_id', chunk).execute()

            # Delete content by ID list — same set, no accidental over-delete
            for i in range(0, len(old_ids), BATCH):
                chunk = old_ids[i:i + BATCH]
                db.table('content').delete().in_('id', chunk).execute()

            print(f"   ✅ Removed {len(old_ids)} old entries (movies + TV)")
        else:
            print(f"   ℹ️ No old data to clean")
    except Exception as e:
        print(f"   ⚡️ Cleanup failed: {e}")

# FIX #11: cleanup_old_movies() removed — get_trending() filters >3yr movies before upsert

# ============================================================================
# JUSTWATCH FETCHER — Direct platform streaming URLs via JustWatch GraphQL API
# Confirmed working 2026-02-24. Queries apis.justwatch.com/graphql (no key).
# Returns real netflix.com/title/... and app.primevideo.com/detail?gti=... URLs.
# ============================================================================

class JustWatchFetcher:

    GQL = "https://apis.justwatch.com/graphql"

    # Confirmed shortNames from live API probe
    PLATFORM_MAP = {
        'nfx': 'Netflix',
        'prv': 'Prime Video',
        'pva': 'Prime Video',  # ad-supported tier
        'jhs': 'Jiohotstar',   # confirmed live
        'hst': 'Jiohotstar',   # legacy shortName — still returned by JustWatch for some titles
        'dnp': 'Jiohotstar',   # legacy
        'hot': 'Jiohotstar',   # legacy
        'jio': 'Jiohotstar',   # JioCinema merged into Jiohotstar
        'atp': 'Apple TV+',
    }

    SEARCH_Q = """
    query GetSearchTitles($searchQuery: String!, $country: Country!, $first: Int!) {
      popularTitles(country: $country, filter: { searchQuery: $searchQuery }, first: $first) {
        edges { node { id __typename } }
      }
    }
    """

    OFFERS_Q = """
    query GetTitleOffers($nodeId: ID!, $country: Country!) {
      node(id: $nodeId) {
        ... on Movie { offers(country: $country, platform: WEB) { standardWebURL package { shortName } } }
        ... on Show  { offers(country: $country, platform: WEB) { standardWebURL package { shortName } } }
      }
    }
    """

    # Thread-local storage — one requests.Session per thread so concurrent
    # workers in _fetch_and_update don't share a non-thread-safe session.
    _local = threading.local()

    # JustWatch request headers defined once — used by both _gql and _gql_threadsafe.
    _HEADERS = {
        'User-Agent':   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Content-Type': 'application/json',
        'Origin':       'https://www.justwatch.com',
        'Referer':      'https://www.justwatch.com/',
    }

    def __init__(self):
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        # Single session for serial callers (non-threaded paths)
        self.session = requests.Session()
        self.session.headers.update(self._HEADERS)

    def _gql(self, query, variables):
        """GQL call using the shared session — for serial (non-threaded) use."""
        try:
            r = self.session.post(self.GQL, json={'query': query, 'variables': variables}, timeout=12)
            if r.status_code != 200:
                return None
            data = r.json()
            if 'errors' in data:
                return None
            return data
        except Exception as e:
            print(f"   ⚡️ JustWatch GQL fetch error: {e}")
            return None

    def _gql_threadsafe(self, query, variables):
        """GQL call using a per-thread session — safe to call from a ThreadPoolExecutor."""
        if not hasattr(self._local, 'session'):
            s = requests.Session()
            s.headers.update(self._HEADERS)
            self._local.session = s
        try:
            r = self._local.session.post(self.GQL, json={'query': query, 'variables': variables}, timeout=12)
            if r.status_code == 429:
                print(f"   ⚡️ JustWatch rate limited (429) — backing off 10s")
                time.sleep(10)
                return None
            if r.status_code != 200:
                print(f"   ⚡️ JustWatch HTTP {r.status_code}")
                return None
            data = r.json()
            if 'errors' in data:
                return None
            return data
        except Exception as e:
            print(f"   ⚡️ JustWatch GQL fetch error: {e}")
            return None

    def _fetch_and_update(self, rows, table_name):
        """Fetch JustWatch URLs concurrently and update the table."""
        updated = 0
        lock = threading.Lock()

        def find_url(item):
            title      = item['title']
            media_type = item['content_type']
            platform   = item['platform']

            data = self._gql_threadsafe(self.SEARCH_Q, {'searchQuery': title, 'country': 'IN', 'first': 8})
            if not data:
                return item, None, None

            edges = data.get('data', {}).get('popularTitles', {}).get('edges', [])
            want  = 'Movie' if media_type != 'tv' else 'Show'
            candidates = [e['node']['id'] for e in edges if e['node'].get('__typename') == want]
            fallback   = [e['node']['id'] for e in edges if e['node'].get('__typename') != want]

            for node_id in candidates + fallback:
                time.sleep(0.1)  # gentle gap — avoids burst 429s across 5 workers
                od = self._gql_threadsafe(self.OFFERS_Q, {'nodeId': node_id, 'country': 'IN'})
                if not od:
                    continue
                offers = (od.get('data', {}).get('node') or {}).get('offers', [])\
                
                for offer in offers:
                    short = offer.get('package', {}).get('shortName', '')
                    plat  = self.PLATFORM_MAP.get(short)
                    url   = offer.get('standardWebURL', '')
                    if plat == platform and url:
                        return item, url, None  # validUntil removed from JustWatch schema

            return item, None, None

        MAX_JW_WORKERS = 5  # keep low — JustWatch rate-limits on burst
        with ThreadPoolExecutor(max_workers=MAX_JW_WORKERS) as executor:
            futures = {executor.submit(find_url, item): item for item in rows}
            done = 0
            for future in as_completed(futures):
                done += 1
                item, url, leaving_date = future.result()
                title = item['title']
                if done % 50 == 0:
                    print(f"   ⏳ {done}/{len(rows)} processed...")
                if not url:
                    continue
                try:
                    update_payload = {'stream_url': url}
                    # Always write leaving_date — None clears a stale date if the
                    # licence was renewed and JustWatch no longer reports one
                    update_payload['leaving_date'] = leaving_date
                    self.db.table(table_name).update(update_payload).eq('id', item['id']).execute()
                    with lock:
                        updated += 1
                    leaving_note = f"  ⏳ leaves {leaving_date}" if leaving_date else ''
                    print(f"   ✅ {title[:38]:<38} → {url[:46]}{leaving_note}")
                except Exception as e:
                    print(f"   ❌ DB: {str(e)[:60]}")

        return updated

    def run(self):
        print("\n" + "="*70)
        print("🔗 FETCHING DIRECT STREAMING LINKS — Watch Now (JustWatch)")
        print("="*70)

        # FIX NEW-10: paginate — Supabase silently caps at 1000 rows
        all_rows = []
        for start in range(0, 50000, 1000):
            page = self.db.table('content').select(
                'id, title, original_title, content_type, platform, release_year, stream_url, leaving_date'
            ).range(start, start + 999).execute()
            if not page.data:
                break
            all_rows.extend(page.data)
            if len(page.data) < 1000:
                break
        if not all_rows:
            print("⚡️ No Watch Now content found")
            return
        updated = self._fetch_and_update(all_rows, 'content')
        print(f"\n✅ {updated}/{len(all_rows)} Watch Now titles got direct stream links")

    def run_discover(self):
        print("\n" + "="*70)
        print("🔗 FETCHING DIRECT STREAMING LINKS — Discover (JustWatch)")
        print("="*70)

        # Paginate to bypass Supabase 1000 row default limit
        all_data = []
        for start in range(0, 10000, 1000):
            page = self.db.table('discover_content').select(
                'id, title, original_title, content_type, platform, release_year, stream_url, leaving_date'
            ).range(start, start + 999).execute()
            if not page.data:
                break
            all_data.extend(page.data)
            if len(page.data) < 1000:
                break

        if not all_data:
            print("⚡️ No Discover content found")
            return

        total = len(all_data)
        today = datetime.now().date()
        recheck_window_days = 14   # re-check titles leaving within 14 days

        # ── Smart skip logic ─────────────────────────────────────────────────
        # Only re-fetch a title if:
        #   (a) it has no stream URL yet                → needs first-time fetch
        #   (b) it has a leaving_date within 14 days   → expiry may have changed
        #   (c) it has a leaving_date in the past      → may have been renewed
        # Skip everything else — JustWatch data doesn't change hourly.
        needs_check = []
        skipped     = 0
        for row in all_data:
            if not row.get('stream_url'):
                needs_check.append(row)   # (a) no URL yet
                continue
            ld = row.get('leaving_date')
            if ld:
                try:
                    leave = datetime.strptime(ld, '%Y-%m-%d').date()
                    days_left = (leave - today).days
                    if days_left <= recheck_window_days:
                        needs_check.append(row)   # (b)/(c) expiring soon or already gone
                    else:
                        skipped += 1
                except ValueError:
                    needs_check.append(row)   # malformed date — recheck to be safe
            else:
                skipped += 1   # has URL, no leaving_date → stable, skip

        already_linked = sum(1 for r in all_data if r.get('stream_url'))
        print(f"   📊 {total} total titles  |  {already_linked} already linked")
        print(f"   ⏭️  {skipped} stable titles skipped (have URL, no expiry pressure)")
        print(f"   🔄 {len(needs_check)} titles to check "
              f"({total - already_linked} new  +  "
              f"{len(needs_check) - (total - already_linked)} expiring soon)")

        if not needs_check:
            print("   ✅ Nothing to update — all titles stable")
            return

        updated = self._fetch_and_update(needs_check, 'discover_content')
        print(f"\n✅ {updated}/{len(needs_check)} Discover titles updated")



# ============================================================================
# MAIN - RUN BOTH FLOWS
def embed_new_discover():
    """
    Embed any discover_content rows missing a vector for Vibe Search.
    Delegates entirely to generate_embeddings.run_incremental() — single
    source of truth for batch size, rate limiting and retry logic.
    """
    try:
        import generate_embeddings
    except ImportError:
        print('\n⚠️  generate_embeddings.py not found — skipping embedding step')
        return

    db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    generate_embeddings.run_incremental(db)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-reddit', action='store_true',
                        help='Disable Reddit')
    parser.add_argument('--no-critics', action='store_true',
                        help='Disable RT critic scraper')
    parser.add_argument('--discover-only', action='store_true',
                        help='Skip Watch Now flow — only run Discover + stream URL fetch')
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
    print("🧠 LLM sentiment enabled (Groq → Gemini → VADER cascade)")

    print("\n" + "="*70)
    print("🎬 STREAMING TRACKER - TWO-FLOW SYSTEM")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not Config.TMDB_API_KEY:
        print("❌ Missing TMDB_API_KEY in .env")
        return
    
    if not Config.YOUTUBE_API_KEY:
        print("⚠️  No YOUTUBE_API_KEY — YouTube reviews disabled. TMDb + Reddit still run.")
    
    if os.getenv("OMDB_API_KEY"):
        print("✅ OMDB API key found — RT scores via API (reliable)")
    else:
        print("⚠️  No OMDB_API_KEY — RT scraping will likely be blocked. Get a free key at omdbapi.com")
    
    print("✅ API keys loaded")
    
    # FIX #17: YouTubeTranscriptApi import removed — no transcript code paths exist
    
    # FIX NEW-8: feedparser check removed — feedparser has zero callers, was silently disabling Reddit
    
    # Clean up old data
    cleanup_old_data(days_old=7)
    # FIX #11: cleanup_old_movies() removed — get_trending() already filters >3yr movies before upsert
    
    # FLOW 1: DISCOVER (No reviews, just availability)
    print("\n🔍 STARTING DISCOVER FLOW...")
    discover = DiscoverFlow()
    discover.save_discover_content()

    # FLOW 2: WATCH NOW (With reviews & scoring — async pipeline)
    if args.discover_only:
        print("\n⏭️  Skipping Watch Now flow (--discover-only)")
        jw = JustWatchFetcher()
        jw.run_discover()  # Discover — fetches links for 'discover_content' table
    else:
        print("\n📺 STARTING WATCH NOW FLOW...")
        ingester = AsyncWatchNowPipeline()
        ingester.run()

        computer = ScoreComputer()
        computer.compute_all()

        jw = JustWatchFetcher()
        jw.run()           # Watch Now — fetches links for 'content' table
        jw.run_discover()  # Discover — fetches links for 'discover_content' table

    # Embed any new Discover titles for Vibe Search
    embed_new_discover()

    print("\n" + "="*70)
    print("🎉 DONE!")
    print("="*70)
    print("\n📊 Summary:")
    print("   ✅ Discover content saved (with stream URLs)")
    if not args.discover_only:
        print("   ✅ Watch Now content scored (with reviews + stream URLs)")

if __name__ == "__main__":
    main()
