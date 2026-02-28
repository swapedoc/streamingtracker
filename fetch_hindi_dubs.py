#!/usr/bin/env python3
"""
fetch_hindi_dubs.py — Checks JustWatch India for Hindi audio availability
and updates your Supabase 'content' and 'discover_content' tables.

SETUP:
  pip install requests python-dotenv supabase

USAGE:
  python3 fetch_hindi_dubs.py

SUPABASE SETUP (run once in Supabase SQL editor):
  ALTER TABLE content          ADD COLUMN IF NOT EXISTS hindi_dub BOOLEAN DEFAULT FALSE;
  ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS hindi_dub BOOLEAN DEFAULT FALSE;
"""

import os
import sys
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
GQL          = 'https://apis.justwatch.com/graphql'

MAX_WORKERS = 20

# ── GraphQL Queries ──────────────────────────────────────────────────────────

SEARCH_Q = """
query SearchTitles($searchQuery: String!, $country: Country!, $first: Int!) {
  popularTitles(country: $country, filter: { searchQuery: $searchQuery }, first: $first) {
    edges {
      node {
        id
        __typename
        ... on Movie { content(country: $country, language: "en") { title originalTitle } }
        ... on Show  { content(country: $country, language: "en") { title originalTitle } }
      }
    }
  }
}
"""

OFFERS_Q = """
query GetOffers($nodeId: ID!, $country: Country!) {
  node(id: $nodeId) {
    ... on Movie {
      offers(country: $country, platform: WEB) {
        audioLanguages
        package { shortName }
      }
    }
    ... on Show {
      offers(country: $country, platform: WEB) {
        audioLanguages
        package { shortName }
      }
    }
  }
}
"""

# ── Thread-safe HTTP sessions ─────────────────────────────────────────────────

_thread_local = threading.local()

def _get_session():
    if not hasattr(_thread_local, 'session'):
        s = requests.Session()
        s.headers.update({
            'User-Agent':      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Content-Type':    'application/json',
            'Accept':          'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Origin':          'https://www.justwatch.com',
            'Referer':         'https://www.justwatch.com/',
            'sec-ch-ua':       '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            'sec-ch-ua-mobile':   '?0',
            'sec-ch-ua-platform': '"macOS"',
            'Sec-Fetch-Dest':  'empty',
            'Sec-Fetch-Mode':  'cors',
            'Sec-Fetch-Site':  'same-site',
        })
        _thread_local.session = s
    return _thread_local.session


def gql(query, variables, retries=3):
    for attempt in range(retries):
        try:
            r = _get_session().post(GQL, json={'query': query, 'variables': variables}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if 'errors' in data:
                    return None
                return data
            if attempt < retries - 1:
                continue
            return None
        except Exception:
            if attempt == retries - 1:
                return None
    return None


# ── Live progress bar ────────────────────────────────────────────────────────

class LiveProgress:
    def __init__(self, total):
        self.total     = total
        self.completed = 0
        self.found     = 0
        self.errors    = 0
        self._lock     = threading.Lock()
        self._start    = time.time()
        self._spinner  = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
        self._tick     = 0
        self._draw()

    def _bar(self):
        filled = int(30 * self.completed / max(self.total, 1))
        return '█' * filled + '░' * (30 - filled)

    def _eta(self):
        elapsed = time.time() - self._start
        if self.completed == 0:
            return '--:--'
        remaining = (elapsed / self.completed) * (self.total - self.completed)
        m, s = divmod(int(remaining), 60)
        return f'{m:02d}:{s:02d}'

    def _draw(self):
        spin = self._spinner[self._tick % len(self._spinner)]
        self._tick += 1
        pct  = self.completed / max(self.total, 1) * 100
        rate = self.completed / max(time.time() - self._start, 0.1)
        line = (
            f'\r  {spin} [{self._bar()}] '
            f'{self.completed}/{self.total} ({pct:.0f}%)  '
            f'✅ {self.found} hindi  '
            f'⚠️  {self.errors} err  '
            f'{rate:.1f}/s  ETA {self._eta()}   '
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def log(self, msg):
        with self._lock:
            sys.stdout.write(f'\r{" " * 110}\r')
            print(msg)
            self._draw()

    def advance(self, found=False, error=False):
        with self._lock:
            self.completed += 1
            if found:  self.found  += 1
            if error:  self.errors += 1
            self._draw()

    def finish(self):
        sys.stdout.write(f'\r{" " * 110}\r')
        sys.stdout.flush()


import re

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(the|a|an)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

def _titles_match(search: str, result: str, original: str = '') -> bool:
    needle = _normalize(search)
    for candidate in filter(None, [_normalize(result), _normalize(original)]):
        if needle == candidate:
            return True
        # Allow substring match only for longer titles (avoids "Kill" matching "Kill Bill")
        if len(needle) >= 6 and (needle in candidate or candidate in needle):
            return True
    return False




def has_hindi_dub(title: str, content_type: str):
    data = gql(SEARCH_Q, {'searchQuery': title, 'country': 'IN', 'first': 5})
    if not data:
        return None

    edges = data.get('data', {}).get('popularTitles', {}).get('edges', [])
    if not edges:
        return False

    want = 'Movie' if content_type != 'tv' else 'Show'

    # Find the top result of the correct type whose title actually matches
    top_match = None
    for e in edges:
        node = e['node']
        if node.get('__typename') != want:
            continue
        content = node.get('content') or {}
        if _titles_match(title, content.get('title', ''), content.get('originalTitle', '')):
            top_match = node['id']
            break

    if not top_match:
        return False

    offer_data = gql(OFFERS_Q, {'nodeId': top_match, 'country': 'IN'})
    if not offer_data:
        return None

    offers = (offer_data.get('data', {}).get('node') or {}).get('offers', [])
    return any('hi' in (offer.get('audioLanguages') or []) for offer in offers)


# ── Concurrent table processor ───────────────────────────────────────────────

def process_table(db, table_name, rows, max_workers=MAX_WORKERS):
    to_process = [r for r in rows if r.get('hindi_dub') is not True]
    skipped    = len(rows) - len(to_process)
    found = not_found = errors = 0

    if skipped:
        print(f'  ⏭  {skipped} already tagged, skipping.')

    if not to_process:
        print('  ✅ Nothing new to process.')
        return 0, 0, 0, skipped

    print(f'  🚀 {len(to_process)} titles — {max_workers} concurrent workers\n')

    progress = LiveProgress(len(to_process))

    def check_one(item):
        return item, has_hindi_dub(item.get('title', '?'), item.get('content_type', 'movie'))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_one, item): item for item in to_process}

        for future in as_completed(futures):
            item, result = future.result()
            title  = item.get('title', '?')
            row_id = item['id']

            if result is None:
                progress.log(f'  ⚠️  {title[:55]} — API error')
                progress.advance(error=True)
                errors += 1
                continue

            try:
                db.table(table_name).update({'hindi_dub': result}).eq('id', row_id).execute()
                if result:
                    progress.log(f'  ✅ {title[:60]}')
                    progress.advance(found=True)
                    found += 1
                else:
                    progress.advance()
                    not_found += 1
            except Exception as e:
                progress.log(f'  ❌ DB error — {title[:40]}: {e}')
                progress.advance(error=True)
                errors += 1

    progress.finish()
    return found, not_found, errors, skipped


# ── Main ─────────────────────────────────────────────────────────────────────

def fetch_all_rows(db, table_name):
    """Fetch ALL rows from a Supabase table using pagination (bypasses 1000-row limit)."""
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        result = (
            db.table(table_name)
            .select('id, title, content_type, hindi_dub')
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data or []
        all_rows.extend(batch)
        print(f'  📄 Fetched {len(all_rows)} rows so far...', end='\r')
        if len(batch) < page_size:
            break
        offset += page_size
    print(f'  📄 Total fetched: {len(all_rows)} rows{" " * 20}')
    return all_rows


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print('❌ Missing SUPABASE_URL or SUPABASE_KEY in .env')
        return

    db = create_client(SUPABASE_URL, SUPABASE_KEY)

    print('\n' + '='*65)
    print('🎬 HINDI DUB FETCHER — JustWatch India')
    print('='*65)
    print()

    # ── Watch Now ─────────────────────────────────────────────────────────────
    print('📺 Processing WATCH NOW (content table)...')
    print('-'*65)
    rows = fetch_all_rows(db, 'content')
    if not rows:
        print('   No rows found.')
    else:
        f, nf, e, s = process_table(db, 'content', rows)
        print(f'   ✅ Hindi dub found: {f}')
        print(f'   ✗  No hindi dub:   {nf}')
        print(f'   ⏭  Already tagged: {s}')
        print(f'   ⚠️  Errors:         {e}')

    print()

    # ── Discover ──────────────────────────────────────────────────────────────
    print('🔍 Processing DISCOVER (discover_content table)...')
    print('-'*65)
    rows2 = fetch_all_rows(db, 'discover_content')
    if not rows2:
        print('   No rows found.')
    else:
        f2, nf2, e2, s2 = process_table(db, 'discover_content', rows2)
        print(f'   ✅ Hindi dub found: {f2}')
        print(f'   ✗  No hindi dub:   {nf2}')
        print(f'   ⏭  Already tagged: {s2}')
        print(f'   ⚠️  Errors:         {e2}')

    print()
    print('='*65)
    print('🎉 Done! hindi_dub column updated in Supabase.')
    print('='*65)


if __name__ == '__main__':
    main()
