# streaming_tracker_v25.py - Production Version with Reddit Integration
# Run with: python3 streaming_tracker_v25.py

"""
STREAMING TRACKER V2.5 - FINAL WITH REDDIT

FEATURES:
1. Real platform filtering via TMDb Watch Providers
2. IMDB ratings from TMDb vote_average
3. Fixed engagement double-counting
4. Transcript analysis (optional, with fallback)
5. Gemini Flash sentiment (with VADER fallback)
6. Reddit discussion ingestion via RSS (no API key needed!)
7. Better polarization detection (min 3 reviews)
8. Recency-based dynamic weights
9. Trending vs Catalog categorization

DATABASE MIGRATION NEEDED:
Run this SQL in Supabase:

ALTER TABLE scores ADD COLUMN IF NOT EXISTS category TEXT DEFAULT 'catalog';
CREATE INDEX IF NOT EXISTS idx_scores_category ON scores(category);
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
    
    SEARCH_KEYWORDS = ['review 2025', 'honest review']
    
    # WATCH NOW FLOW (Gets reviews)
    TRENDING_LIMIT = 40
    UNDERDOG_LIMIT = 20
    GENRE_WATCH_NOW_LIMIT = 15  # Popular picks per genre
    GENRE_LIMIT = 20
    MAX_VIDEOS_PER_PLATFORM = 10
    
    # DISCOVER FLOW (No reviews, just browsing)
    CLASSICS_LIMIT = 50
    GENRE_DISCOVER_LIMIT = 30  # Full catalog per genre
    
    ENABLED_GENRES = ['Action', 'Horror', 'Thriller', 'Comedy', 'Drama', 'Romance']
    DISCOVER_CLASSICS_LIMIT = 30
    DISCOVER_GENRE_LIMIT = 20
    DISCOVER_UNDERDOG_LIMIT = 15
    DISCOVER_ENABLED_GENRES = ['Action', 'Horror', 'Thriller', 'Comedy', 'Drama', 'Romance'] 
    USE_TRANSCRIPTS = True
    USE_REDDIT = True
    STRICT_PLATFORM_FILTERING = True    
# ============================================================================
# TMDB INTEGRATION
# ============================================================================

class TMDbResolver:
    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
    
    def get_trending(self, media_type='all', time_window='week', limit=20) -> List[Dict]:
        """Get trending content from TMDb with Retry Logic"""
        
        # SURGICAL EDIT: Added retry loop for connection stability
        url = f"{self.base_url}/trending/{media_type}/{time_window}"
        params = {'api_key': self.api_key}
        data = {}
        
        # Try up to 4 times
        max_retries = 4
        for attempt in range(max_retries):
            try:
                # Lowered timeout to 10s so we don't wait forever on a dead link
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                break # Success! Exit the loop
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌  TMDb trending error: {e}")
                    return [] # Give up after 4 tries
                
                wait_time = 2 * (attempt + 1)
                print(f"   ⚡️  Connection unstable. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
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
                    'imdb_rating': item.get('vote_average')
                })

            print(f"\n✅ Found {len(trending)} trending titles")
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
                    print(f"  ⚡️ Provider check failed (attempt {attempt+1}/{retries}), retrying...")
                    time.sleep(2)
                else:
                    print(f"  ⚡️ Provider check failed after {retries} attempts: {e}")
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
# UNDERDOG DISCOVERY
# ============================================================================

def discover_underdogs(api_key: str, limit=10) -> List[Dict]:
    """Find hidden gems - high quality but under-the-radar content"""
    print("\n🔍  Discovering Hidden Gems (Underdogs)...")
    
    today = datetime.now()
    three_months_ago = today - timedelta(days=90)
    
    underdogs = []
    
    for media_type in ['movie', 'tv']:
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"https://api.themoviedb.org/3/discover/{media_type}",
                    params={
                        'api_key': api_key,
                        'region': 'IN',
                        'sort_by': 'vote_average.desc',
                        'vote_count.gte': 50,
                        'vote_count.lte': 3000,
                        'vote_average.gte': 7.2,
                        f"{'primary_release_date' if media_type == 'movie' else 'first_air_date'}.gte": three_months_ago.strftime("%Y-%m-%d"),
                        'with_watch_providers': '8|119|350|2336|220',
                        'watch_region': 'IN'
                    },
                    timeout=30,
                    headers={'Connection': 'close'}  # Prevent keep-alive issues
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
                
                # Success - break retry loop
                break
                
            except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                    print(f"  ⚡️  Connection error for {media_type}, retrying in {wait_time}s... (attempt {attempt+1}/{retries})")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Failed to discover {media_type} underdogs after {retries} attempts")
            
            except Exception as e:
                print(f"  ⚡️ Underdog discovery error for {media_type}: {e}")
                break
        
        time.sleep(2)  # Delay between movie and TV
    
    print(f"✅ Found {len(underdogs)} hidden gems")
    return underdogs
# ============================================================================
# GENRE-BASED DISCOVERY
# ============================================================================

def discover_by_genre(api_key: str, genre_name: str, genre_id: int, limit=15) -> List[Dict]:
    """
    Discover popular content by genre
    Returns highly-rated, available content in specific genres
    """
    print(f"\n🎭  Discovering {genre_name} content...")
    
    genre_content = []
    
    for media_type in ['movie', 'tv']:
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"https://api.themoviedb.org/3/discover/{media_type}",
                    params={
                        'api_key': api_key,
                        'with_genres': genre_id,
                        'sort_by': 'popularity.desc',
                        'vote_average.gte': 6.5,  # Quality filter
                        'vote_count.gte': 100,     # Has enough ratings
                        'with_watch_providers': '8|119|350|2336|220',  # Our platforms
                        'watch_region': 'IN',
                        'page': 1
                    },
                    timeout=30,
                    headers={'Connection': 'close'}
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
                
                break  # Success
                
            except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  ⚡️  Connection error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌  Failed to discover {genre_name} {media_type}")
            
            except Exception as e:
                print(f"  ⚡️  Error discovering {genre_name} {media_type}: {e}")
                break
        
        time.sleep(1)
    
    print(f"✅  Found {len(genre_content)} {genre_name} titles")
    return genre_content

# ============================================================================
# TOP RATED CLASSICS DISCOVERY
# ============================================================================

def discover_classics(api_key: str, limit=20) -> List[Dict]:
    """
    Discover all-time great movies and shows (IMDB 8.0+)
    These are proven classics that stand the test of time
    """
    print("\n⭐  Discovering Top Rated Classics (IMDB 8.0+)...")
    
    classics = []
    
    for media_type in ['movie', 'tv']:
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"https://api.themoviedb.org/3/discover/{media_type}",
                    params={
                        'api_key': api_key,
                        'sort_by': 'vote_average.desc',
                        'vote_average.gte': 8.0,   # IMDB 8.0+
                        'vote_count.gte': 500,      # Must have substantial ratings
                        'with_watch_providers': '8|119|350|2336|220',
                        'watch_region': 'IN',
                        'page': 1
                    },
                    timeout=30,
                    headers={'Connection': 'close'}
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
                        'category': 'classics',
                        'is_classic': True
                    })
                
                break
                
            except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  ⚡️ Connection error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Failed to discover {media_type} classics")
            
            except Exception as e:
                print(f"  ⚡️  Error discovering {media_type} classics: {e}")
                break
        
        time.sleep(1)
    
    print(f"✅  Found {len(classics)} all-time classics")
    return classics
# ============================================================================
# SENTIMENT ANALYSIS - 3-TIER CASCADE SYSTEM
# ============================================================================

class SentimentAnalyzer:
    def __init__(self):
        # Always initialize VADER as final fallback
        self._init_vader()
        
        # Tier 1: Groq (Fastest & Most generous free tier)
        self.groq_client = None
        self.use_groq = False
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                self.use_groq = True
                print("✅  Tier 1: Groq AI enabled (Primary)")
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
    
    def _init_vader(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict:
        if not text or len(text.strip()) < 10:
            return {'sentiment': 0, 'confidence': 0.0}
        
        # Try Tier 1: Groq
        if self.use_groq:
            result = self._groq_analyze(text)
            if result:
                return result
            print("    ⤵️  Groq failed, falling back to Gemini...")        
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
                print(f"  ⚡️  Groq rate limited, trying Gemini...")
                self.use_groq = False  # Disable for this session
            else:
                print(f"  ⚡️  Groq error: {str(e)[:100]}")
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
# Replace the entire RedditIngester class with this:
class RedditIngester:
    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        # Use a "Real Browser" User-Agent to avoid being treated like a bot
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_reddit_discussions(self, title: str, media_type: str) -> List[Dict]:
        """Search Reddit using the JSON API (More reliable than RSS)"""
        subreddit = "television" if media_type == 'tv' else "movies"
        
        # 1. Search Query
        search_url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            'q': f"{title} discussion",
            'restrict_sr': 'on', # Restrict to the specific subreddit
            'sort': 'relevance',
            't': 'all',          # Search all time, not just today
            'limit': 3
        }

        print(f"     📡 Searching r/{subreddit} for '{title}'...")
        
        try:
            resp = requests.get(search_url, headers=self.headers, params=params, timeout=10)
            
            # Debugging: Print exactly what happened if it fails
            if resp.status_code != 200:
                print(f"     ⚡️  Reddit Error {resp.status_code}")
                return []
                
            data = resp.json()
            posts = data.get('data', {}).get('children', [])
            
            if not posts:
                print("     ⚡️  No threads found (JSON returned empty list)")
                return []

            print(f"     ✅  Found {len(posts)} threads. Fetching comments...")
            
            results = []
            for post in posts:
                post_data = post['data']
                thread_title = post_data.get('title', '')
                thread_id = post_data.get('id')
                permalink = post_data.get('permalink')
                
                # Filter out irrelevant threads (Trailers, etc.)
                if "trailer" in thread_title.lower():
                    continue

                # 2. Fetch Comments for this thread
                # URL format: https://www.reddit.com/r/{sub}/comments/{id}.json
                comments_url = f"https://www.reddit.com/comments/{thread_id}.json"
                
                c_resp = requests.get(comments_url, headers=self.headers, timeout=10)
                if c_resp.status_code != 200: continue
                
                c_data = c_resp.json()
                # Reddit returns a list: [thread_info, comments_info]
                if len(c_data) < 2: continue
                
                comments_list = c_data[1].get('data', {}).get('children', [])
                
                extracted_comments = []
                for comment in comments_list[:15]: # Top 15 comments
                    c_body = comment.get('data', {}).get('body', '')
                    
                    if c_body and c_body != '[deleted]' and len(c_body) > 20:
                        # Sentiment Analysis
                        sent = self.sentiment.analyze(c_body)
                        extracted_comments.append({
                            'text': c_body[:500], # Store first 500 chars
                            'sentiment': sent['sentiment'],
                            'confidence': sent['confidence']
                        })
                
                if extracted_comments:
                    results.append({
                        'title': thread_title,
                        'url': f"https://www.reddit.com{permalink}",
                        'comments': extracted_comments
                    })
                    
                time.sleep(1) # Polite delay
                
            return results

        except Exception as e:
            print(f"     ⚡️  Reddit Network Error: {e}")
            return []

    # Keep your existing helper methods
    def compute_reddit_score(self, threads: List[Dict]) -> float:
        if not threads: return 50.0
        all_sentiments = []
        for thread in threads:
            for comment in thread['comments']:
                all_sentiments.append(comment['sentiment'])
        
        if not all_sentiments: return 50.0
        # Map -1..1 to 0..100
        avg = sum(all_sentiments) / len(all_sentiments)
        return (avg + 1) * 50

    def save_reddit_reviews(self, content_id: int, threads: List[Dict]):
        # Reuse the exact logic you had before, just ensure it loops correctly
        count = 0
        for thread in threads:
            for comment in thread['comments']:
                review_data = {
                    'content_id': content_id,
                    'source': 'reddit',
                    'source_url': thread['url'],
                    'source_id': f"{thread['url']}_{hash(comment['text'][:10])}",
                    'reviewer': 'Reddit User',
                    'review_text': comment['text'],
                    'sentiment_score': (comment['sentiment'] + 1) * 50, # Save as 0-100
                    'confidence': comment['confidence']
                }
                try:
                    self.db.table('reviews').upsert(review_data, on_conflict='source,source_id').execute()
                    count += 1
                except: pass
        if count > 0:
            print(f"     💾 Saved {count} Reddit comments")
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
            return "🔥  Must Watch"
        elif score >= 65:
            return "👍  Worth Your Time"
        elif score >= 50:
            return "🤷  Genre Fans Only"
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
        
        for video in videos:
            print(f"     📺 {video['title'][:50]}...")
            
            stats = self.get_video_stats(video['video_id'], video['channel_id'])
            
            if not stats:
                continue
            
            transcript = self.get_transcript(video['video_id'])
            
            if transcript:
                review_text = transcript
                print(f"     📝 Using transcript ({len(transcript)} chars)")
            else:
                review_text = f"{video['title']} {video['description']}"
                print(f"     📝 Using title + description")
            
            sentiment_result = self.sentiment.analyze(review_text)
            print(f"     💭 Sentiment: {sentiment_result['sentiment']} (conf: {sentiment_result['confidence']:.2f})")
            
            youtube_weight = self.scoring.youtube_weight(
                stats.get('views', 0),
                stats.get('subscribers', 0),
                stats.get('comments', 0)
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
            
            time.sleep(2)
        
        # Get Reddit discussions if enabled
        if self.reddit:
            print(f"   📡 Searching Reddit discussions...")
            threads = self.reddit.get_reddit_discussions(title, media_type)
            
            if threads:
                reddit_score = self.reddit.compute_reddit_score(threads)
                print(f"   📊 Reddit Score: {reddit_score:.1f} from {len(threads)} thread(s)")
                self.reddit.save_reddit_reviews(content_id, threads)
            else:
                print(f"   ⚡️ No Reddit discussions found")
    
    def run(self):
        print("\n" + "="*70)
        print("🚀 CONTENT DISCOVERY V2.5 - MULTI-SOURCE")
        print("="*70)
        
        all_content = []
        
        # Source 1: TRENDING
        print("\n📊 Discovering Trending Content from TMDb...")
        trending_all = self.tmdb.get_trending('all', 'week', limit=Config.TRENDING_LIMIT)
        all_content.extend(trending_all)
        
        # Source 2: UNDERDOGS (Hidden Gems)
        print("\n🔍 Discovering Hidden Gems...")
        underdogs = discover_underdogs(Config.TMDB_API_KEY, limit=Config.UNDERDOG_LIMIT)
        all_content.extend(underdogs)
        
        # Source 3: TOP RATED CLASSICS (All-time greats)
        print("\n⭐ Discovering Classics...")
        classics = discover_classics(Config.TMDB_API_KEY, limit=Config.CLASSICS_LIMIT)
        all_content.extend(classics)
        
        # Source 4: GENRE-BASED DISCOVERY
        print("\n🎭 Discovering by Genre...")
        for genre_name in Config.ENABLED_GENRES:
            genre_id = Config.GENRES.get(genre_name)
            if genre_id:
                genre_content = discover_by_genre(
                    Config.TMDB_API_KEY, 
                    genre_name, 
                    genre_id, 
                    limit=Config.GENRE_LIMIT
                )
                all_content.extend(genre_content)
        
        # Remove duplicates (same tmdb_id)
        seen_ids = set()
        unique_content = []
        for item in all_content:
            if item['tmdb_id'] not in seen_ids:
                seen_ids.add(item['tmdb_id'])
                unique_content.append(item)
        
        all_content = unique_content
        
        if not all_content:
            print("❌ No content discovered")
            return
        
        print(f"\n✅ DISCOVERY SUMMARY:")
        print(f"   🔥 Trending: {len(trending_all)}")
        print(f"   💎 Underdogs: {len(underdogs)}")
        print(f"   ⭐ Classics: {len([x for x in all_content if x.get('category') == 'classics'])}")
       
        genre_counts = {}
        for item in all_content:
            if 'genre' in item:
                genre = item['genre']
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        for genre, count in genre_counts.items():
            print(f"   🎭 {genre}: {count}")
        
        print(f"   📦 Total Unique: {len(all_content)}")
        
        # Show sample with better categorization
        print(f"\n📋 Sample of content to process:")
        for i, item in enumerate(all_content[:20], 1):
            category = item.get('category', '')
            
            if category == 'trending':
                emoji = "🔥"
                tag = ""
            elif category == 'underdog':
                emoji = "💎"
                tag = ""
            elif category == 'classics':
                emoji = "⭐"
                tag = " [CLASSIC]"
            elif category.startswith('genre_'):
                emoji = "🎭"
                tag = f" [{item.get('genre')}]"
            else:
                emoji = "📺"
                tag = ""
            
            print(f"   {i:2}. {emoji} {item['title'][:40]:40}{tag} | IMDB: {item.get('imdb_rating', 'N/A')}")
        
        if len(all_content) > 20:
            print(f"   ... and {len(all_content) - 20} more") 
        processed = 0
        
        for platform, platform_id in Config.PLATFORMS.items():
            print(f"\n{'='*70}")
            print(f"🎬 {platform.upper()} (Provider ID: {platform_id})")
            print(f"{'='*70}")
            
            platform_processed = 0
            
            for content_item in all_content:
                if platform_processed >= Config.MAX_VIDEOS_PER_PLATFORM:
                    break
                
                self.process_trending_content(content_item, platform, platform_id)
                platform_processed += 1
                time.sleep(2)
            
            processed += platform_processed
        
        print("\n" + "="*70)
        print(f"✅ INGESTION COMPLETE - Processed {processed} items")
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
        print("📊 COMPUTING SCORES WITH RECENCY DECAY")
        print("="*70)
        
        content_result = self.db.table('content').select('*').execute()
        
        if not content_result.data:
            print("⚡️ No content found")
            return
        
        for content in content_result.data:
            content_id = content['id']
            title = content['title']
             # ✅ VERIFY AVAILABILITY BEFORE SCORING
            providers = self.tmdb.get_watch_providers(content['tmdb_id'], content['content_type'])
            platform_id = Config.PLATFORMS.get(content['platform'])
            
            if not providers or platform_id not in providers:
                print(f"\n🗑️ {title} - No longer on {content['platform']}, removing...")
                self.db.table('reviews').delete().eq('content_id', content_id).execute()
                self.db.table('scores').delete().eq('content_id', content_id).execute()
                self.db.table('content').delete().eq('id', content_id).execute()
                continue  # Skip scoring
            
            print(f"\n🎬 {title}")
            
            reviews_result = self.db.table('reviews').select('*').eq('content_id', content_id).execute()
            
            if not reviews_result.data:
                print(f"   ⚡️ No reviews")
                continue
            
            reviews = reviews_result.data
            print(f"   📊 {len(reviews)} reviews")
            
            youtube_reviews = [r for r in reviews if r['source'] == 'youtube']
            reddit_reviews = [r for r in reviews if r['source'] == 'reddit']
            
            if youtube_reviews:
                weighted_sents = [r['weighted_sentiment'] for r in youtube_reviews]
                avg = np.mean(weighted_sents)
                youtube_score = (avg + 1) * 50
            else:
                youtube_score = 50
            
            if reddit_reviews:
                weighted_sents = [r['weighted_sentiment'] for r in reddit_reviews]
                avg = np.mean(weighted_sents)
                reddit_score = (avg + 1) * 50
            else:
                reddit_score = 50
            
            imdb_score = self.scoring.normalize_imdb(content.get('imdb_rating'))
            
            # Dynamic weights based on age
            weights = self.scoring.get_dynamic_weights(content.get('release_year'))
            
            # If we have Reddit data, adjust weights
            has_reddit = reddit_score != 50 and len(reddit_reviews) > 0
            
            if has_reddit:
                # Distribute weight: YouTube + Reddit share social weight, IMDB gets its weight
                final_score = (
                    weights['youtube'] * 0.5 * youtube_score +
                    weights['youtube'] * 0.5 * reddit_score +
                    weights['imdb'] * imdb_score
                )
            else:
                # Original weights: YouTube, IMDB
                final_score = (
                    weights['youtube'] * youtube_score +
                    weights['imdb'] * imdb_score
                )
            
            label = self.scoring.get_label(final_score)
            
            # Determine category
            category = self.scoring.get_category(content.get('release_year'))
            
            sentiments = [r['sentiment'] for r in reviews]
            is_polarizing = len(sentiments) >= 3 and np.std(sentiments) > 0.7
            
            positive_ratio = sum(1 for r in reviews if r['sentiment'] == 1) / len(reviews)
            
            print(f"   🏆 {final_score:.1f} - {label}")
            print(f"      YouTube: {youtube_score:.1f} | Reddit: {reddit_score:.1f} | IMDB: {imdb_score:.1f}")
            print(f"      Weights: YT {weights['youtube']:.0%} | IMDB {weights['imdb']:.0%} | Category: {category.upper()}")
            
            if is_polarizing:
                print(f"   🧨 POLARIZING")
            
            score_data = {
                'content_id': content_id,
                'youtube_score': round(youtube_score, 1),
                'reddit_score': round(reddit_score, 1),
                'imdb_score': round(imdb_score, 1),
                'engagement_score': 0.0,
                'final_score': round(final_score, 1),
                'label': label,
                'category': category,
                'review_count': len(reviews),
                'positive_ratio': round(positive_ratio, 2),
                'is_polarizing': bool(is_polarizing),
                'sentiment_std': round(np.std(sentiments), 2) if len(sentiments) > 1 else 0.0
            }
            
            try:
                self.db.table('scores').upsert(score_data, on_conflict='content_id').execute()
                print(f"   ✅ Saved")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
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

# Add this before the main() function

def cleanup_old_data(days_old=7):
    """Remove content and associated data older than X days"""
    print(f"\n🧹 Cleaning up data older than {days_old} days...")
    db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    
    cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
    
    try:
        # Get old content IDs
        old_content = db.table('content').select('id').lt('created_at', cutoff_date).execute()
        
        if old_content.data:
            old_ids = [item['id'] for item in old_content.data]
            
            # Delete reviews first (foreign key constraint)
            for content_id in old_ids:
                db.table('reviews').delete().eq('content_id', content_id).execute()
                db.table('scores').delete().eq('content_id', content_id).execute()
            
            # Delete content
            db.table('content').delete().lt('created_at', cutoff_date).execute()
            
            print(f"   ✅ Removed {len(old_ids)} old entries")
        else:
            print(f"   ℹ️ No old data to clean")
    except Exception as e:
        print(f"   ⚡️ Cleanup failed: {e}")
# ===========================================================================
#  DISCOVER FLOW
# ===========================================================================
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
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎬 STREAMING TRACKER V2.5 - FINAL WITH REDDIT")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # ... existing checks ...
    
    # FLOW 1: DISCOVER (No reviews, just availability)
    print("\n🔍 STARTING DISCOVER FLOW...")
    discover = DiscoverFlow()
    discover.save_discover_content()
    #Main logic 
    if not Config.YOUTUBE_API_KEY:
        print("❌ Missing YOUTUBE_API_KEY in .env")
        return
    
    if not Config.TMDB_API_KEY:
        print("❌ Missing TMDB_API_KEY in .env")
        return
    
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
    cleanup_old_data(days_old=7)  # Remove content older than 7 days
    
    ingester = YouTubeIngester()
    ingester.run()
    
    computer = ScoreComputer()
    computer.compute_all()
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print("="*70)
    print("\nNext: streamlit run dashboard_v2.py")

if __name__ == "__main__":
    main()
