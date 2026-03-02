#!/usr/bin/env python3
"""
enrich_trailers_cron.py — Weekly trailer + runtime backfill for discover_content
=================================================================================
Run this SEPARATELY from streaming_tracker_v3.py, on a weekly cron schedule.
It is intentionally isolated so it cannot touch the main tracker's YouTube quota.

WHAT IT DOES:
  1. Loads all discover_content rows missing trailer_id or runtime
  2. Tries TMDb /videos first (free, no YouTube quota)
  3. Falls back to YouTube search ONLY for rows that TMDb couldn't resolve
  4. Applies any manual overrides from the trailer_overrides table
  5. Saves progress to DB as it goes — safe to interrupt and re-run

QUOTA COST:
  - TMDb calls: free (no quota)
  - YouTube searches: capped at YOUTUBE_SEARCH_CAP (default 80/run = 8,000 units)
  - With 10,000 units/day quota, this script uses at most 8,000 leaving
    2,000 for the main tracker on the same day

CRON SETUP (runs every Sunday at 2am):
  0 2 * * 0 cd /path/to/project && python3 enrich_trailers_cron.py >> logs/trailers.log 2>&1

CLI FLAGS:
  --force         Re-fetch trailers for ALL rows (ignores existing trailer_id)
  --limit N       Only process N rows this run (default: all missing)
  --yt-cap N      Max YouTube fallback searches (default: 80)
  --dry-run       Print what would be done without saving to DB
"""

import os
import sys
import time
import threading
import argparse
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
from supabase import create_client
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

TMDB_API_KEY    = os.getenv('TMDB_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
SUPABASE_URL    = os.getenv('SUPABASE_URL')
SUPABASE_KEY    = os.getenv('SUPABASE_KEY')

WORKERS             = 6      # concurrent TMDb calls (YouTube calls are serialised)
YOUTUBE_SEARCH_CAP  = 80     # max YouTube searches per run (80 × 100 units = 8,000)
TMDB_SLEEP          = 0.15   # seconds between TMDb calls per worker to stay under rate limit

# ── Trusted YouTube channel names / IDs (same list as main tracker) ───────────

TRUSTED_CHANNEL_NAMES = {
    'netflix', 'prime video', 'amazon prime', 'apple tv', 'disney',
    'hotstar', 'sony pictures', 'warner bros', 'universal pictures',
    'paramount', 'lionsgate', 'a24', 'marvel', 'dc', 'mgm', 'mubi',
    'ign', 'filmspot trailer', 'hulu', 'hbo',
    'zee studios', 'zee music company', 'yash raj films', 'dharma productions',
    't-series', 'pen movies', 'pen marudhar', 'eros now', 'eros movies',
    'jiocinema', 'jio studios', 'excel entertainment', 'maddock films',
    'tips films', 'tips official', 'red chillies entertainment',
    'balaji motion pictures', 'viacom18 studios', 'viacom18',
    'saregama music', 'sony music india', 'reliance entertainment',
    'gulshan kumar', 'bhushan kumar',
}
TRUSTED_CHANNEL_IDS = {
    'UCWX3yGbOBE3mMSBSCVDzK4g', 'UCTOxLBzMBCEFTEMF7DqhAsg',
    'UC_IRUbMCnBQh5X5hTLjWLpA', 'UCmEDwvvN9LiRh0dMjB8RWLA',
    'UCzWQYUVCpZqtN93H8RR44Qw', 'UCi8e0iOVk1fEOogdfu4YgfA',
    'UCF9imwbTCaZCOcuZnWTFBkA', 'UCvC4D8onUfXzvjTOM-dBfEA',
    'UCjmJDM5pRKbUlVIzDYwz-1A', 'UCaw03IoN618hT5-bXu6LoAA',
    'UCYLbpjXO5BwstZXhBpqECRg', 'UCgMPP6RejnQEoEX-bwOFZLA',
    'UCR_Gp53bEfFTL2W6VLMJ15g', 'UCFFbwnve3yF62-tVXkTyHqg',
    'UC9zY_E8mcAo_Oq772LEZq8Q', 'UCiEEF51uRAeZeCo8CJFhGWw',
}
REJECT_WORDS = {
    'reaction', 'review', 'fan made', 'fan-made', 'fan trailer',
    'fan concept', 'concept trailer', 'breakdown', 'explained',
    'analysis', 'ranked', 'every scene', 'deleted',
    'behind the scenes', 'making of', 'interview',
    'featurette', 'clip', 'scene', 'spoiler', 'pitch meeting',
}

# ── HTTP session with retry ───────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['GET'])
    s.mount('https://', HTTPAdapter(max_retries=retry, pool_maxsize=10))
    s.headers.update({'User-Agent': 'streaming-trailer-cron/1.0'})
    return s

SESSION = _make_session()

# ── YouTube quota guard (shared across threads) ───────────────────────────────

_yt_lock        = Lock()
_yt_calls_made  = 0
_yt_quota_blown = False   # set True on 403 or once cap is reached

def _yt_search(query: str, cap: int) -> str | None:
    """
    Search YouTube for an official trailer from a trusted channel.
    Returns video_id or None. Thread-safe quota guard — stops on 403 or cap.
    """
    global _yt_calls_made, _yt_quota_blown

    with _yt_lock:
        if _yt_quota_blown:
            return None
        if _yt_calls_made >= cap:
            _yt_quota_blown = True
            print(f"\n   ⚠️  YouTube cap reached ({cap} searches = {cap*100:,} units) — skipping remaining")
            return None
        _yt_calls_made += 1
        call_num = _yt_calls_made

    queries = [query, query.rsplit(' ', 1)[0] if ' ' in query else query]  # with year, then without
    for q in queries:
        try:
            r = SESSION.get(
                'https://www.googleapis.com/youtube/v3/search',
                params={'part': 'snippet', 'q': q, 'type': 'video',
                        'maxResults': 8, 'order': 'relevance', 'key': YOUTUBE_API_KEY},
                timeout=10,
            )
            if r.status_code == 403:
                with _yt_lock:
                    _yt_quota_blown = True
                print(f"\n   🚫 YouTube 403 — quota exhausted or key invalid. "
                      f"Check: console.cloud.google.com/apis/api/youtube.googleapis.com/quotas")
                return None
            if not r.ok:
                continue
            for item in r.json().get('items', []):
                vid_id     = item.get('id', {}).get('videoId', '')
                snippet    = item.get('snippet', {})
                vid_title  = snippet.get('title', '').lower()
                channel    = snippet.get('channelTitle', '').lower()
                channel_id = snippet.get('channelId', '')
                if not vid_id:
                    continue
                if 'trailer' not in vid_title and 'teaser' not in vid_title:
                    continue
                if any(w in vid_title for w in REJECT_WORDS):
                    continue
                if not (any(t in channel for t in TRUSTED_CHANNEL_NAMES)
                        or channel_id in TRUSTED_CHANNEL_IDS):
                    continue
                return vid_id
        except Exception as e:
            print(f'   ⚡️ YouTube search error: {e}')
    return None

# ── TMDb helpers ──────────────────────────────────────────────────────────────

def _tmdb_get(path: str, params: dict) -> dict | None:
    try:
        r = SESSION.get(
            f"https://api.themoviedb.org/3{path}",
            params={'api_key': TMDB_API_KEY, **params},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f'   ⚡️ TMDb API error for {path}: {e}')
    return None


def _get_runtime(tmdb_id: int, media_type: str) -> dict:
    """Returns dict with runtime/seasons/episode_count/episode_runtime."""
    result = {'runtime': None, 'seasons': None, 'episode_count': None, 'episode_runtime': None}
    data = _tmdb_get(f"/{media_type}/{tmdb_id}", {})
    if not data:
        return result
    if media_type == 'movie':
        result['runtime'] = data.get('runtime') or None
    else:
        result['seasons']       = data.get('number_of_seasons') or None
        result['episode_count'] = data.get('number_of_episodes') or None
        ep_rt = None
        er = [x for x in (data.get('episode_run_time') or []) if x and x > 0]
        if er:
            ep_rt = er[0]
        if not ep_rt:
            ep_rt = (data.get('last_episode_to_air') or {}).get('runtime') or None
        if not ep_rt:
            ep_rt = (data.get('next_episode_to_air') or {}).get('runtime') or None
        result['episode_runtime'] = ep_rt
    return result


def _get_trailer_from_tmdb(tmdb_id: int, media_type: str) -> str | None:
    """Try TMDb /videos in English then Hindi — no YouTube quota used."""
    for lang in ('en-US', 'hi'):
        data = _tmdb_get(f"/{media_type}/{tmdb_id}/videos", {'language': lang})
        if not data:
            continue
        for vtype in ('Trailer', 'Teaser'):
            for v in data.get('results', []):
                if (v.get('site') == 'YouTube'
                        and v.get('type') == vtype
                        and v.get('official', True)):
                    return v['key']
    return None


def enrich_one(row: dict, yt_cap: int, dry_run: bool) -> dict:
    """
    Enrich a single discover_content row.
    Returns a dict of fields to update (may be empty if nothing changed).
    """
    tmdb_id    = row['tmdb_id']
    media_type = row['content_type']
    title      = row.get('title', '')
    year       = row.get('release_year')
    updates    = {}

    time.sleep(TMDB_SLEEP)   # gentle rate limit per worker

    # ── Runtime ───────────────────────────────────────────────────────────────
    needs_runtime = (
        (media_type == 'movie' and not row.get('runtime')) or
        (media_type == 'tv'    and not row.get('episode_runtime'))
    )
    if needs_runtime:
        rt = _get_runtime(tmdb_id, media_type)
        for k in ('runtime', 'seasons', 'episode_count', 'episode_runtime'):
            if rt.get(k) is not None:
                updates[k] = rt[k]

    # ── Trailer ───────────────────────────────────────────────────────────────
    if not row.get('trailer_id'):
        trailer_id = _get_trailer_from_tmdb(tmdb_id, media_type)

        if not trailer_id and title and YOUTUBE_API_KEY:
            # YouTube fallback — quota-protected
            q = f"{title} {year} official trailer" if year else f"{title} official trailer"
            trailer_id = _yt_search(q, yt_cap)
            if trailer_id:
                updates['_yt_fallback'] = True  # for logging only — stripped before DB save

        if trailer_id:
            updates['trailer_id'] = trailer_id

    return updates


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Weekly trailer + runtime enrichment cron')
    parser.add_argument('--force',   action='store_true',
                        help='Re-fetch trailers for ALL rows (ignores existing trailer_id)')
    parser.add_argument('--limit',   type=int, default=0,
                        help='Only process this many rows (0 = all missing)')
    parser.add_argument('--yt-cap',  type=int, default=YOUTUBE_SEARCH_CAP,
                        help=f'Max YouTube trailer searches (default: {YOUTUBE_SEARCH_CAP})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be updated without saving to DB')
    args = parser.parse_args()

    print("=" * 65)
    print("🎬 TRAILER + RUNTIME ENRICHMENT CRON")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    if not TMDB_API_KEY:
        print("❌ Missing TMDB_API_KEY in .env"); sys.exit(1)
    if not YOUTUBE_API_KEY:
        print("⚠️  No YOUTUBE_API_KEY — will use TMDb-only (no YouTube fallback)")
    if args.dry_run:
        print("🔍 DRY RUN — nothing will be saved to DB")

    db = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ── Load rows ─────────────────────────────────────────────────────────────
    print("\n📦 Loading discover_content rows...")
    all_rows = []
    for start in range(0, 50_000, 1000):
        page = db.table('discover_content').select(
            'id, tmdb_id, title, release_year, content_type, trailer_id, runtime, episode_runtime'
        ).range(start, start + 999).execute()
        if not page.data:
            break
        all_rows.extend(page.data)
        if len(page.data) < 1000:
            break
    print(f"   Total rows: {len(all_rows)}")

    # ── Load manual overrides ─────────────────────────────────────────────────
    overrides = {}
    try:
        ov = db.table('trailer_overrides').select('tmdb_id, trailer_id').execute().data or []
        overrides = {r['tmdb_id']: r['trailer_id'] for r in ov}
        if overrides:
            print(f"   🎬 {len(overrides)} manual override(s) loaded")
    except Exception as e:
        print(f'   ⚡️ Failed to load trailer overrides: {e}')

    # Apply overrides immediately
    override_hits = [r for r in all_rows if r['tmdb_id'] in overrides and not r.get('trailer_id')]
    for r in override_hits:
        if not args.dry_run:
            try:
                db.table('discover_content').update(
                    {'trailer_id': overrides[r['tmdb_id']]}
                ).eq('id', r['id']).execute()
            except Exception as e:
                print(f"   ⚡️ Override DB save error for '{r['title']}': {e}")
        else:
            print(f"   [DRY] Override: {r['title']} → {overrides[r['tmdb_id']]}")
    if override_hits:
        print(f"   ✅ {len(override_hits)} override(s) applied")

    # ── Filter to rows that actually need work ────────────────────────────────
    to_do = []
    for r in all_rows:
        if r['tmdb_id'] in overrides:
            continue   # already handled above
        needs_trailer = args.force or not r.get('trailer_id')
        needs_runtime = (
            (r.get('content_type') == 'movie' and not r.get('runtime')) or
            (r.get('content_type') == 'tv'    and not r.get('episode_runtime'))
        )
        if needs_trailer or needs_runtime:
            to_do.append(r)

    if args.limit > 0:
        to_do = to_do[:args.limit]
        print(f"   🔢 --limit {args.limit}: processing {len(to_do)} rows")

    if not to_do:
        print("\n✅ Nothing to do — all rows already enriched.")
        return

    print(f"\n🚀 {len(to_do)} rows to enrich")
    print(f"   Workers : {WORKERS}")
    print(f"   YT cap  : {args.yt_cap} searches ({args.yt_cap * 100:,} quota units)")
    print(f"   Force   : {args.force}")
    print()

    # ── Enrich concurrently ───────────────────────────────────────────────────
    lock    = Lock()
    updated = 0
    skipped = 0
    errors  = 0
    done    = 0
    yt_hits = 0

    def _do(row):
        return row, enrich_one(row, args.yt_cap, args.dry_run)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_do, r): r for r in to_do}
        for future in as_completed(futures):
            try:
                row, updates = future.result()
                yt_fb = updates.pop('_yt_fallback', False)

                with lock:
                    done += 1
                    if updates:
                        updated += 1
                        if yt_fb:
                            yt_hits += 1
                            print(f"   🔍 YT fallback: {row['title'][:45]}")
                    else:
                        skipped += 1

                    # Progress every 50 rows
                    if done % 50 == 0 or done == len(to_do):
                        pct = done / len(to_do) * 100
                        print(f"   ⏳ {done}/{len(to_do)} ({pct:.0f}%) — "
                              f"updated={updated} yt={yt_hits} skip={skipped} err={errors}")

                if updates and not args.dry_run:
                    db.table('discover_content').update(updates).eq('id', row['id']).execute()
                elif args.dry_run and updates:
                    fields = ', '.join(f"{k}={str(v)[:20]}" for k, v in updates.items())
                    print(f"   [DRY] {row['title'][:40]} → {fields}")

            except Exception as e:
                with lock:
                    errors += 1
                    done   += 1
                print(f"   ❌ Error on row {futures[future].get('title','?')}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅ ENRICHMENT COMPLETE")
    print(f"   Rows processed : {done}")
    print(f"   Updated        : {updated}")
    print(f"   No change      : {skipped}")
    print(f"   Errors         : {errors}")
    print(f"   YT fallbacks   : {yt_hits} ({yt_hits * 100:,} quota units used)")
    print(f"   Finished       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    if args.dry_run:
        print("\n(Dry run — nothing was saved)")


if __name__ == '__main__':
    main()
