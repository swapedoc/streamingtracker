#!/usr/bin/env python3
"""
fetch_full_catalog.py — Bulk imports the complete Netflix, Prime Video,
JioHotstar and Apple TV+ catalog into discover_content.

Fetches every movie and TV show available on these platforms in India
from TMDb, auto-assigns a category, and upserts into Supabase.
category is only written for new rows — existing rows that already have a
category set are never overwritten, preserving any manual curation.

SETUP:
    pip install aiohttp python-dotenv supabase

USAGE:
    python3 fetch_full_catalog.py
    python3 fetch_full_catalog.py --dry-run      # fetch but don't save
    python3 fetch_full_catalog.py --type movie   # movies only
    python3 fetch_full_catalog.py --type tv      # series only
"""

import os
import sys
import asyncio
import argparse
import time
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client
from constants import (
    PLATFORMS, MOVIE_GENRE_MAP, TV_GENRE_MAP, TV_GENRE_LABELS,
    GENRE_PRIORITY, TV_GENRE_PRIORITY, INDIAN_LANGUAGES,
)

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

TMDB_API_KEY  = os.getenv('TMDB_API_KEY')
SUPABASE_URL  = os.getenv('SUPABASE_URL')
SUPABASE_KEY  = os.getenv('SUPABASE_KEY')

# Platform + genre maps imported from constants.py

# Category assignment thresholds
CLASSICS_YEAR      = 1990
CLASSICS_RATING    = 7.0
UNDERDOG_RATING    = 7.5
UNDERDOG_POPULARITY = 8.0

# Concurrency
PAGE_SEMAPHORE      = 12   # concurrent TMDb /discover page fetches
PROVIDER_SEMAPHORE  = 40   # concurrent provider checks
PAGES_PER_TYPE      = 100  # 100 pages × 20 results = 2000 per content type
BATCH_SIZE          = 100  # Supabase upsert batch size

# ── Genre resolution ──────────────────────────────────────────────────────────

def resolve_genre(genre_ids: list, is_tv: bool = False) -> Optional[str]:
    genre_map = TV_GENRE_MAP if is_tv else MOVIE_GENRE_MAP
    matched = list(dict.fromkeys(
        genre_map[gid] for gid in (genre_ids or []) if gid in genre_map
    ))
    if not matched:
        return None
    for preferred in GENRE_PRIORITY:
        if preferred in matched:
            return preferred
    return matched[0]


def resolve_tv_genre(genre_ids: list) -> Optional[str]:
    matched = list(dict.fromkeys(
        TV_GENRE_LABELS[gid] for gid in (genre_ids or []) if gid in TV_GENRE_LABELS
    ))
    if not matched:
        return None
    for preferred in TV_GENRE_PRIORITY:
        if preferred in matched:
            return preferred
    return matched[0]


# ── Category auto-assignment ──────────────────────────────────────────────────

def assign_category(
    original_language: str,
    release_year: int,
    imdb_rating: float,
    popularity: float,
    genre: Optional[str],
    content_type: str = 'movie',
    tv_genre: Optional[str] = None,
) -> str:
    """
    Priority order:
      1. Indian language content  → 'indian'
      2. Pre-1990 + well rated    → 'classics'
      3. High rating + low pop    → 'underdog'
      4. Genre fallback           → 'genre_*'
         Films use movie-style genre label (genre field)
         Series use TMDb TV genre label (tv_genre field)
      5. Default                  → 'genre_drama'
    """
    if original_language in INDIAN_LANGUAGES:
        return 'indian'

    rating = imdb_rating or 0
    pop    = popularity  or 0
    year   = release_year or 2000

    if year <= CLASSICS_YEAR and rating >= CLASSICS_RATING:
        return 'classics'

    if rating >= UNDERDOG_RATING and pop < UNDERDOG_POPULARITY:
        return 'underdog'

    # Films — use movie-style genre label
    film_genre_map = {
        'Action':   'genre_action',
        'Horror':   'genre_horror',
        'Thriller': 'genre_thriller',
        'Comedy':   'genre_comedy',
        'Drama':    'genre_drama',
        'Sci-Fi':   'genre_sci-fi',
        'Romance':  'genre_romance',
    }

    # Series — use native TMDb TV genre label
    tv_genre_map = {
        'Action & Adventure': 'genre_action',
        'Sci-Fi & Fantasy':   'genre_sci-fi',
        'Crime':              'genre_thriller',
        'Mystery':            'genre_thriller',
        'Comedy':             'genre_comedy',
        'Drama':              'genre_drama',
        'Animation':          'genre_action',   # Arcane, AoT etc skew action not comedy
        'Documentary':        'genre_drama',
        'War & Politics':     'genre_action',
        'Western':            'genre_action',
        'Family':             'genre_drama',    # family content skews drama not comedy
        'Soap':               'genre_drama',
    }

    if content_type == 'tv' and tv_genre:
        return tv_genre_map.get(tv_genre, 'genre_drama')

    return film_genre_map.get(genre or '', 'genre_drama')


# ── Progress bar ──────────────────────────────────────────────────────────────

def progress(current: int, total: int, label: str = ''):
    filled = int(30 * current / max(total, 1))
    bar    = '█' * filled + '░' * (30 - filled)
    pct    = current / max(total, 1) * 100
    sys.stdout.write(f'\r  [{bar}] {current}/{total} ({pct:.0f}%)  {label}   ')
    sys.stdout.flush()


# ── TMDb async fetcher ────────────────────────────────────────────────────────

async def fetch_all(api_key: str, content_types: list) -> list:
    """
    Fetch all /discover pages for each content type, then resolve providers.
    Returns list of (item_dict, platform_name) pairs.
    """
    import aiohttp

    provider_ids = set(PLATFORMS.values())
    provider_id_to_name = {v: k for k, v in PLATFORMS.items()}

    # Build page jobs
    jobs = []
    for mt in content_types:
        for page in range(1, PAGES_PER_TYPE + 1):
            jobs.append({
                'mt':   mt,
                'page': page,
                'params': {
                    'api_key':              api_key,
                    'with_watch_providers': '|'.join(str(p) for p in provider_ids),
                    'watch_region':         'IN',
                    'sort_by':              'popularity.desc',
                    'vote_count.gte':       50,
                    'page':                 page,
                },
            })

    print(f'  🚀 Firing {len(jobs)} TMDb page requests...\n')

    page_sem      = asyncio.Semaphore(PAGE_SEMAPHORE)
    connector     = aiohttp.TCPConnector(ssl=False, limit=30)
    done_count    = 0
    total_jobs    = len(jobs)
    raw_items     = []

    async def fetch_page(session, job):
        nonlocal done_count
        url = f'https://api.themoviedb.org/3/discover/{job["mt"]}'
        async with page_sem:
            for attempt in range(3):
                try:
                    async with session.get(url, params=job['params'], timeout=aiohttp.ClientTimeout(total=20)) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        if resp.status != 200:
                            if attempt < 2:
                                await asyncio.sleep(0.5 * (attempt + 1))
                                continue
                            break
                        data  = await resp.json()
                        items = []
                        mt    = job['mt']
                        for r in data.get('results', []):
                            title        = r.get('title') if mt == 'movie' else r.get('name')
                            release_date = (r.get('release_date') if mt == 'movie'
                                            else r.get('first_air_date')) or '2000-01-01'
                            if not title:
                                continue
                            is_tv   = mt == 'tv'
                            g_ids   = r.get('genre_ids', [])
                            genre   = resolve_genre(g_ids, is_tv=is_tv)
                            tv_genre = resolve_tv_genre(g_ids) if is_tv else None
                            orig_lang = r.get('original_language', '')
                            year    = int(release_date[:4])
                            rating  = r.get('vote_average', 0)
                            pop     = r.get('popularity', 0)
                            items.append({
                                'tmdb_id':        r['id'],
                                'title':          title,
                                'original_title': r.get('original_title') or r.get('original_name'),
                                'content_type':   mt,
                                'release_year':   year,
                                'poster_path':    r.get('poster_path'),
                                'overview':       (r.get('overview') or '')[:500],
                                'popularity':     pop,
                                'imdb_rating':    rating,
                                'genre':          genre,
                                'tv_genre':       tv_genre,
                                'original_language': orig_lang,
                                'category':       assign_category(orig_lang, year, rating, pop, genre, mt, tv_genre),
                            })
                        done_count += 1
                        if done_count % 10 == 0 or done_count == total_jobs:
                            progress(done_count, total_jobs, 'pages fetched')
                        return items
                except Exception as e:
                    print(f'   ⚡️ TMDb page fetch error (attempt {attempt+1}): {e}')
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (attempt + 1))
        done_count += 1
        return []

    async with aiohttp.ClientSession(connector=connector) as session:
        page_tasks   = [fetch_page(session, job) for job in jobs]
        page_results = await asyncio.gather(*page_tasks, return_exceptions=True)

    for r in page_results:
        if isinstance(r, list):
            raw_items.extend(r)

    print(f'\n\n  ✅ {len(raw_items)} raw items from TMDb')

    # Deduplicate by tmdb_id — keep one entry per tmdb_id for provider check
    seen: dict = {}
    for item in raw_items:
        tid = item['tmdb_id']
        if tid not in seen:
            seen[tid] = item
    unique_items = list(seen.values())
    print(f'  📦 {len(unique_items)} unique titles — checking providers...\n')

    # Provider check — resolve which platform(s) each title is on in India
    prov_sem   = asyncio.Semaphore(PROVIDER_SEMAPHORE)
    prov_done  = 0
    total_prov = len(unique_items)
    prov_map: dict = {}  # tmdb_id -> [platform_name, ...]

    async def fetch_providers(session, item):
        nonlocal prov_done
        tid = item['tmdb_id']
        mt  = item['content_type']
        url = f'https://api.themoviedb.org/3/{mt}/{tid}/watch/providers'
        async with prov_sem:
            for attempt in range(3):
                try:
                    async with session.get(
                        url,
                        params={'api_key': api_key},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        if resp.status != 200:
                            if attempt < 2:
                                await asyncio.sleep(0.5 * (attempt + 1))
                                continue
                            break
                        data   = await resp.json()
                        india  = data.get('results', {}).get('IN', {})
                        # FIX #15: flatrate (paid subscription) only — strip free/ads tiers.
                        # Including free/ads caused titles to appear in Discover as
                        # "available on Netflix" when they were only on a free ad-supported
                        # tier. Users clicked through and hit a paywall or the wrong page.
                        # streaming_tracker.py DiscoverFlow already uses flatrate-only;
                        # aligning here keeps both scripts consistent so a title's platform
                        # label means the same thing regardless of which script last wrote it.
                        all_providers = india.get('flatrate', [])
                        pids   = [p['provider_id'] for p in all_providers]
                        platforms = [
                            provider_id_to_name[pid]
                            for pid in pids
                            if pid in provider_id_to_name
                        ]
                        prov_map[tid] = platforms
                        break
                except Exception as e:
                    print(f'   ⚡️ Provider fetch error for tmdb_id {tid} (attempt {attempt+1}): {e}')
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (attempt + 1))
        prov_done += 1
        if prov_done % 50 == 0 or prov_done == total_prov:
            progress(prov_done, total_prov, 'providers checked')

    connector2 = aiohttp.TCPConnector(ssl=False, limit=50)
    async with aiohttp.ClientSession(connector=connector2) as session:
        prov_tasks = [fetch_providers(session, item) for item in unique_items]
        await asyncio.gather(*prov_tasks, return_exceptions=True)

    print(f'\n\n  ✅ Providers done')

    # Expand: one row per (tmdb_id, platform)
    expanded = []
    for item in unique_items:
        for platform in prov_map.get(item['tmdb_id'], []):
            expanded.append((item, platform))

    return expanded


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Bulk import full OTT catalog into discover_content')
    parser.add_argument('--dry-run', action='store_true', help='Fetch but do not save to Supabase')
    parser.add_argument('--type', choices=['movie', 'tv', 'both'], default='both', help='Content type to fetch')
    args = parser.parse_args()

    if not TMDB_API_KEY:
        print('❌  Missing TMDB_API_KEY in .env'); sys.exit(1)
    if not args.dry_run and (not SUPABASE_URL or not SUPABASE_KEY):
        print('❌  Missing SUPABASE_URL or SUPABASE_KEY in .env'); sys.exit(1)

    content_types = (
        ['movie', 'tv'] if args.type == 'both'
        else [args.type]
    )

    print()
    print('=' * 65)
    print('🎬  FULL CATALOG IMPORTER — Indian OTT')
    print('=' * 65)
    print(f'  Platforms : {", ".join(PLATFORMS.keys())}')
    print(f'  Types     : {", ".join(content_types)}')
    print(f'  Pages     : {PAGES_PER_TYPE} per type (~{PAGES_PER_TYPE * 20} titles each)')
    print(f'  Mode      : {"DRY RUN — no DB writes" if args.dry_run else "LIVE"}')
    print('=' * 65)
    print()

    start = time.time()

    # ── Fetch ──────────────────────────────────────────────────────────────
    expanded = asyncio.run(fetch_all(TMDB_API_KEY, content_types))

    # ── Category summary ───────────────────────────────────────────────────
    from collections import Counter
    cat_counts = Counter(item['category'] for item, _ in expanded)
    plat_counts = Counter(platform for _, platform in expanded)

    print(f'\n  📊 {len(expanded)} total (title, platform) pairs\n')
    print('  By category:')
    for cat, n in sorted(cat_counts.items()):
        print(f'    {cat:<20} {n}')
    print('\n  By platform:')
    for plat, n in sorted(plat_counts.items()):
        print(f'    {plat:<20} {n}')

    if args.dry_run:
        print('\n  ✅ Dry run complete — nothing saved.')
        return

    # ── Deduplicate by (tmdb_id, platform) ────────────────────────────────
    # Category priority: classics > underdog > indian > genre_*
    CATEGORY_RANK = {'classics': 0, 'underdog': 1, 'indian': 2}

    def item_rank(item):
        cat  = item['category']
        base = CATEGORY_RANK.get(cat, 3)
        if cat.startswith('genre_'):
            # Penalise if resolved genre doesn't match the bucket
            bucket = cat.replace('genre_', '').capitalize()
            if bucket == 'Sci-fi': bucket = 'Sci-Fi'
            if (item.get('genre') or '').lower() != bucket.lower():
                base = 10
        return base

    seen_pairs: dict = {}
    for item, platform in expanded:
        key  = (item['tmdb_id'], platform)
        rank = item_rank(item)
        if key not in seen_pairs or rank < item_rank(seen_pairs[key][0]):
            seen_pairs[key] = (item, platform)

    rows = []
    # FIX FC1 (part 1/2): build a parallel list of computed categories keyed by
    # (tmdb_id, platform) so we can fill them in a separate pass ONLY for rows
    # where category IS NULL. 'category' is intentionally excluded from the upsert
    # payload below — Supabase upsert updates every column in the payload on conflict,
    # so including it would overwrite any manual curation done in the dashboard.
    # The docstring said "existing categories are preserved" but the old code was
    # silently wiping them every Sunday.
    computed_categories: dict = {}
    for (item, platform) in seen_pairs.values():
        rows.append({
            'tmdb_id':        item['tmdb_id'],
            'title':          item['title'],
            'original_title': item.get('original_title'),
            'platform':       platform,
            'content_type':   item['content_type'],
            'release_year':   item['release_year'],
            'imdb_rating':    item['imdb_rating'],
            'poster_path':    item['poster_path'],
            'overview':       item.get('overview'),
            # 'category' intentionally omitted — filled below for NULL rows only
            'genre':          item.get('genre'),
            'tv_genre':       item.get('tv_genre'),
            'popularity':     item['popularity'],
            'source':         'catalog',
        })
        computed_categories[(item['tmdb_id'], platform)] = item['category']

    print(f'\n  💾 Saving {len(rows)} rows to Supabase...\n')

    # ── Upsert to Supabase ─────────────────────────────────────────────────
    # on_conflict='tmdb_id,platform' means existing rows are updated in place.
    # hindi_dub, stream_url, trailer_id are NOT touched — preserved from prior runs.
    db     = create_client(SUPABASE_URL, SUPABASE_KEY)
    saved  = 0
    errors = 0

    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i:i + BATCH_SIZE]
        try:
            db.table('discover_content').upsert(
                chunk,
                on_conflict='tmdb_id,platform'
            ).execute()
            saved += len(chunk)
            progress(min(i + BATCH_SIZE, len(rows)), len(rows), 'saving…')
        except Exception as e:
            errors += len(chunk)
            print(f'\n  ⚡  Batch error at row {i}: {e}')

    # FIX FC1 (part 2/2): back-fill category only for rows where it is NULL.
    # This fires one UPDATE per batch (same chunking as the upsert above) —
    # NOT per row — so it stays efficient even for a full 3000+ row catalog run.
    # Rows that already have a category (manual or from a prior run) are untouched.
    cat_filled = 0
    cat_errors = 0
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i:i + BATCH_SIZE]
        # Update per (tmdb_id, platform) pair — NOT by tmdb_id alone.
        # A title on two platforms can have different categories (e.g. indian on
        # Jiohotstar, genre_action on Netflix) — batching by tmdb_id would
        # assign the first-seen category to both rows.
        for row in chunk:
            cat = computed_categories[(row['tmdb_id'], row['platform'])]
            try:
                result = (
                    db.table('discover_content')
                    .update({'category': cat})
                    .eq('tmdb_id', row['tmdb_id'])
                    .eq('platform', row['platform'])
                    .is_('category', 'null')
                    .execute()
                )
                cat_filled += len(result.data) if result.data else 0
            except Exception as e:
                cat_errors += 1
                print(f'\n  ⚡  Category fill error for {row["title"]}: {e}')

    elapsed = time.time() - start
    print(f'\n\n  {"=" * 60}')
    print(f'  ✅  Done in {elapsed:.0f}s')
    print(f'  💾  Saved  : {saved}')
    print(f'  🏷️   Categories filled (new rows only) : {cat_filled}')
    if errors:
        print(f'  ⚠️   Upsert errors : {errors}')
    if cat_errors:
        print(f'  ⚠️   Category fill errors : {cat_errors}')
    print(f'\n  ⚡  Run fetch_hindi_dubs.py next to tag Hindi audio.')
    print(f'  {"=" * 60}\n')


if __name__ == '__main__':
    main()
