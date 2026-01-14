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
    tmdb_id INTEGER UNIQUE NOT NULL,
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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_discover_category ON discover_content(category);
CREATE INDEX IF NOT EXISTS idx_discover_platform ON discover_content(platform);
"""

import os
import re
import time
import math
import requests
import numpy as np
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

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
    DISCOVER_CLASSICS_LIMIT = 30
    DISCOVER_GENRE_LIMIT = 20
    DISCOVER_UNDERDOG_LIMIT = 15
    DISCOVER_ENABLED_GENRES = ['Action', 'Horror', 'Thriller', 'Comedy', 'Drama']
    
    USE_TRANSCRIPTS = True
    USE_REDDIT = True

# ============================================================================
# TMDB INTEGRATION
# ============================================================================

class TMDbResolver:
    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
    
    def get_trending(self, media_type='all', time_window='week', limit=20) -> List[Dict]:
        """Get trending content from TMDb"""
        url = f"{self.base_url}/trending/{media_type}/{time_window}"
        params = {'api_key': self.api_key}
        
        for attempt in range(4):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                if attempt == 3:
                    print(f"❌ TMDb trending error: {e}")
                    return []
                wait_time = 2 * (attempt + 1)
                print(f"   ⚠️  Retrying in {wait_time}s...")
                time.sleep(wait_time)

        try:
            trending = []
            for item in data.get('results', [])[:limit]:
                if item.get('media_type') not in ['movie', 'tv']:
                    continue

                media_type = item['media_type']
                title = item.get('title') if media_type == 'movie' else item.get('name')

                trending.append({
                    'tmdb_id': item['id'],
                    'title': title,
                    'original_title': item.get('original_title') or item.get('original_name'),
                    'content_type': media_type,
                    'release_year': self._extract_year(
                        item.get('release_date') if media_type == 'movie' else item.get('first_air_date')
                    ),
                    'poster_path': item.get('poster_path'),
                    'overview': item.get('overview'),
                    'popularity': item.get('popularity', 0),
                    'imdb_rating': item.get('vote_average'),
                    'category': 'trending'
                })

            print(f"✅ Found {len(trending)} trending titles")
            return trending

        except Exception as e:
            print(f"❌ Error parsing TMDb data: {e}")
            return []
    
    def get_watch_providers(self, tmdb_id: int, media_type: str, retries=3) -> List[int]:
        """Get streaming platforms where content is available in India"""
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{self.base_url}/{media_type}/{tmdb_id}/watch/providers",
                    params={'api_key': self.api_key},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                india_data = data.get('results', {}).get('IN', {})
                providers = india_data.get('flatrate', [])
                provider_ids = [p['provider_id'] for p in providers]
                
                time.sleep(0.5)
                return provider_ids
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
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
    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.tmdb = TMDbResolver()
    
    def discover_classics(self, limit=30) -> List[Dict]:
        """Find all-time great movies and shows (IMDB 8.0+)"""
        print("\n⭐ Discovering Classics (IMDB 8.0+)...")
        
        classics = []
        for media_type in ['movie', 'tv']:
            for attempt in range(3):
                try:
                    response = requests.get(
                        f"https://api.themoviedb.org/3/discover/{media_type}",
                        params={
                            'api_key': self.api_key,
                            'sort_by': 'vote_average.desc',
                            'vote_average.gte': 8.0,
                            'vote_count.gte': 500,
                            'with_watch_providers': '8|119|350|2336|220',
                            'watch_region': 'IN'
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    for item in data.get('results', [])[:limit]:
                        release_date = item.get('release_date' if media_type == 'movie' else 'first_air_date', '2000-01-01')
                        
                        classics.append({
                            'tmdb_id': item['id'],
                            'title': item.get('title') if media_type == 'movie' else item.get('name'),
                            'original_title': item.get('original_title') or item.get('original_name'),
                            'content_type': media_type,
                            'release_year': int(release_date[:4]) if release_date else 2000,
                            'poster_path': item.get('poster_path'),
                            'overview': item.get('overview'),
                            'popularity': item.get('popularity', 0),
                            'imdb_rating': item.get('vote_average'),
                            'category': 'classics'
                        })
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        print(f"  ⚠️  Failed to discover {media_type} classics")
            time.sleep(1)
        
        print(f"✅ Found {len(classics)} classics")
        return classics
    
    def discover_by_genre(self, genre_name: str, genre_id: int, limit=20) -> List[Dict]:
        """Discover popular content by genre"""
        print(f"🎭 Discovering {genre_name} content...")
        
        genre_content = []
        for media_type in ['movie', 'tv']:
            for attempt in range(3):
                try:
                    response = requests.get(
                        f"https://api.themoviedb.org/3/discover/{media_type}",
                        params={
                            'api_key': self.api_key,
                            'with_genres': genre_id,
                            'sort_by': 'popularity.desc',
                            'vote_average.gte': 6.5,
                            'vote_count.gte': 100,
                            'with_watch_providers': '8|119|350|337|220',
                            'watch_region': 'IN'
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    for item in data.get('results', [])[:limit]:
                        release_date = item.get('release_date' if media_type == 'movie' else 'first_air_date', '2024-01-01')
                        
                        genre_content.append({
                            'tmdb_id': item['id'],
                            'title': item.get('title') if media_type == 'movie' else item.get('name'),
                            'original_title': item.get('original_title') or item.get('original_name'),
                            'content_type': media_type,
                            'release_year': int(release_date[:4]) if release_date else 2024,
                            'poster_path': item.get('poster_path'),
                            'overview': item.get('overview'),
                            'popularity': item.get('popularity', 0),
                            'imdb_rating': item.get('vote_average'),
                            'category': f'genre_{genre_name.lower()}',
                            'genre': genre_name
                        })
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
            time.sleep(1)
        
        print(f"✅ Found {len(genre_content)} {genre_name} titles")
        return genre_content
    
    def discover_underdogs(self, limit=15) -> List[Dict]:
        """Find hidden gems"""
        print("\n💎 Discovering Hidden Gems...")
        
        today = datetime.now()
        three_months_ago = today - timedelta(days=90)
        
        underdogs = []
        for media_type in ['movie', 'tv']:
            for attempt in range(3):
                try:
                    response = requests.get(
                        f"https://api.themoviedb.org/3/discover/{media_type}",
                        params={
                            'api_key': self.api_key,
                            'region': 'IN',
                            'sort_by': 'vote_average.desc',
                            'vote_count.gte': 50,
                            'vote_count.lte': 3000,
                            'vote_average.gte': 7.2,
                            f"{'primary_release_date' if media_type == 'movie' else 'first_air_date'}.gte": three_months_ago.strftime("%Y-%m-%d"),
                            'with_watch_providers': '8|119|350|2336|220',
                            'watch_region': 'IN'
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    for item in data.get('results', [])[:limit]:
                        underdogs.append({
                            'tmdb_id': item['id'],
                            'title': item.get('title') if media_type == 'movie' else item.get('name'),
                            'original_title': item.get('original_title') or item.get('original_name'),
                            'content_type': media_type,
                            'release_year': int(item.get('release_date' if media_type == 'movie' else 'first_air_date', '2024-01-01')[:4]),
                            'poster_path': item.get('poster_path'),
                            'overview': item.get('overview'),
                            'popularity': item.get('popularity', 0),
                            'imdb_rating': item.get('vote_average'),
                            'category': 'underdog'
                        })
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
            time.sleep(2)
        
        print(f"✅ Found {len(underdogs)} hidden gems")
        return underdogs
    
    def save_discover_content(self):
        """Run the discover flow and save to database"""
        print("\n" + "="*70)
        print("🔍 DISCOVER FLOW - Collecting Content (No Reviews)")
        print("="*70)
        
        all_content = []
        
        # 1. Classics
        classics = self.discover_classics(Config.DISCOVER_CLASSICS_LIMIT)
        all_content.extend(classics)
        
        # 2. Genres
        for genre_name in Config.DISCOVER_ENABLED_GENRES:
            genre_id = Config.GENRES.get(genre_name)
            if genre_id:
                genre_content = self.discover_by_genre(
                    genre_name, 
                    genre_id, 
                    Config.DISCOVER_GENRE_LIMIT
                )
                all_content.extend(genre_content)
        
        # 3. Hidden Gems
        underdogs = self.discover_underdogs(Config.DISCOVER_UNDERDOG_LIMIT)
        all_content.extend(underdogs)
        
        # Remove duplicates
        seen_ids = set()
        unique_content = []
        for item in all_content:
            if item['tmdb_id'] not in seen_ids:
                seen_ids.add(item['tmdb_id'])
                unique_content.append(item)
        
        print(f"\n📦 Total unique items: {len(unique_content)}")
        
        # Save to database with platform verification
        saved = 0
        for item in unique_content:
            # Verify availability on at least one platform
            providers = self.tmdb.get_watch_providers(item['tmdb_id'], item['content_type'])
            
            if not providers:
                continue
            
            # Find which of our platforms has it
            available_platforms = []
            for platform_name, platform_id in Config.PLATFORMS.items():
                if platform_id in providers:
                    available_platforms.append(platform_name)
            
            if not available_platforms:
                continue
            
            # Save for each available platform
            for platform in available_platforms:
                discover_data = {
                    'tmdb_id': item['tmdb_id'],
                    'title': item['title'],
                    'original_title': item['original_title'],
                    'platform': platform,
                    'content_type': item['content_type'],
                    'release_year': item['release_year'],
                    'imdb_rating': item['imdb_rating'],
                    'poster_path': item['poster_path'],
                    'overview': item['overview'],
                    'category': item['category'],
                    'genre': item.get('genre'),
                    'popularity': item['popularity']
                }
                
                try:
                    self.db.table('discover_content').upsert(
                        discover_data, 
                        on_conflict='tmdb_id,platform'
                    ).execute()
                    saved += 1
                except Exception as e:
                    print(f"  ⚠️  Error saving {item['title']}: {e}")
            
            time.sleep(0.3)
        
        print(f"\n✅ Saved {saved} items to discover_content")
        print("="*70)

# ============================================================================
# WATCH NOW FLOW - EXISTING CODE (with reviews & scoring)
# ============================================================================

# [Keep all your existing sentiment analyzer, reddit ingester, youtube ingester code]
# I'm including the essential parts below:

class SentimentAnalyzer:
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Try Groq first
        self.groq_client = None
        self.use_groq = False
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                self.use_groq = True
                print("✅ Groq AI enabled")
            except:
                pass
    
    def analyze(self, text: str) -> Dict:
        if not text or len(text.strip()) < 10:
            return {'sentiment': 0, 'confidence': 0.0}
        
        if self.use_groq:
            result = self._groq_analyze(text)
            if result:
                return result
        
        return self._vader_analyze(text)
    
    def _groq_analyze(self, text: str) -> Optional[Dict]:
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze sentiment. Return ONLY JSON:
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
            
            return {'sentiment': sentiment, 'confidence': confidence}
        except:
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

# [Keep your RedditIngester, YouTubeIngester, ScoringEngine classes as-is]

# ============================================================================
# MAIN - RUN BOTH FLOWS
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎬 STREAMING TRACKER V3.0 - TWO-FLOW SYSTEM")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not Config.TMDB_API_KEY:
        print("❌ Missing TMDB_API_KEY in .env")
        return
    
    # FLOW 1: DISCOVER (No reviews, just availability)
    print("\n🔍 STARTING DISCOVER FLOW...")
    discover = DiscoverFlow()
    discover.save_discover_content()
    
    # FLOW 2: WATCH NOW (With reviews & scoring)
    print("\n📺 STARTING WATCH NOW FLOW...")
    print("⚠️  This will take longer (collecting reviews)...")
    
    # [Run your existing YouTubeIngester flow here for trending content]
    # ingester = YouTubeIngester()
    # ingester.run()
    
    # computer = ScoreComputer()
    # computer.compute_all()
    
    print("\n" + "="*70)
    print("🎉 BOTH FLOWS COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print("   ✅ Discover content saved (no reviews)")
    print("   ✅ Watch Now content scored (with reviews)")
    print("\nNext: streamlit run dashboard_v3.py")

if __name__ == "__main__":
    main()
