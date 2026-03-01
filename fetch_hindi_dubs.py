#!/usr/bin/env python3
"""
fetch_hindi_dubs.py — Checks JustWatch India for Hindi audio availability
and updates your Supabase 'content' and 'discover_content' tables.

SETUP:
  pip install requests python-dotenv supabase

USAGE:
  python3 fetch_hindi_dubs.py

ENV VARS (in .env):
  Required:  SUPABASE_URL, SUPABASE_KEY
  Optional:  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
             — if set, you'll receive a Telegram alert the moment
               JustWatch changes their GraphQL schema.

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

SUPABASE_URL      = os.getenv('SUPABASE_URL')
SUPABASE_KEY      = os.getenv('SUPABASE_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')   # optional — set to enable alerts
TELEGRAM_CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID')     # optional — your chat ID
GQL               = 'https://apis.justwatch.com/graphql'

MAX_WORKERS = 20

# Ensures only one schema-change Telegram alert fires per run, even with 20 concurrent workers.
_schema_alert_sent = threading.Event()


# ── Schema validation + alerting ─────────────────────────────────────────────

def _send_telegram_alert(message: str):
    """Fire a Telegram message if credentials are configured. Never raises.
    Deduplicates: only the first schema error per run sends an alert.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    # Only send once per run — prevents alert storm if schema breaks mid-batch
    if _schema_alert_sent.is_set():
        return
    _schema_alert_sent.set()
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': f'🚨 StreamIQ — {message}'},
            timeout=10,
        )
    except Exception:
        pass  # alerting should never crash the main script


class SchemaError(Exception):
    """Raised when a JustWatch response is missing expected keys."""


def _validate_search_response(data: dict) -> list:
    """
    Validates the search response and returns the edges list.
    Raises SchemaError with a descriptive message if the shape has changed.
    """
    if not isinstance(data, dict):
        raise SchemaError(f"Expected dict, got {type(data).__name__}")

    gql_data = data.get('data')
    if not isinstance(gql_data, dict):
        raise SchemaError(f"Missing 'data' key — top-level keys: {list(data.keys())}")

    popular = gql_data.get('popularTitles')
    # popularTitles=null means the search service returned nothing (e.g. service hiccup),
    # not a schema change. Distinguish key-absent (schema break) from key-null (empty result).
    if popular is None:
        if 'popularTitles' not in gql_data:
            raise SchemaError(
                f"'popularTitles' key missing entirely — available keys: {list(gql_data.keys())}"
            )
        return []  # key present but null — treat as empty result, not a schema error
    if not isinstance(popular, dict):
        raise SchemaError(f"'popularTitles' is {type(popular).__name__}, expected dict")

    edges = popular.get('edges')
    if edges is None:
        if 'edges' not in popular:
            raise SchemaError(
                f"'edges' key missing entirely from popularTitles — available keys: {list(popular.keys())}"
            )
        return []  # edges present but null — empty result
    if not isinstance(edges, list):
        raise SchemaError(f"'edges' is {type(edges).__name__}, expected list")

    return edges


def _validate_offers_response(data: dict) -> list:
    """
    Validates the offers response and returns the offers list.
    Raises SchemaError if the shape has changed.
    """
    if not isinstance(data, dict):
        raise SchemaError(f"Expected dict, got {type(data).__name__}")

    gql_data = data.get('data')
    if not isinstance(gql_data, dict):
        raise SchemaError(f"Missing 'data' key — top-level keys: {list(data.keys())}")

    node = gql_data.get('node')
    # node=null is a valid JustWatch response when the ID is unknown — not a schema error
    if node is None:
        if 'node' not in gql_data:
            raise SchemaError(
                f"'node' key missing entirely from data — available keys: {list(gql_data.keys())}"
            )
        return []  # node key exists but is null — unknown ID, no offers

    if not isinstance(node, dict):
        raise SchemaError(f"'node' is {type(node).__name__}, expected dict or null")

    offers = node.get('offers')
    if offers is None:
        raise SchemaError(
            f"'offers' missing from node — available keys: {list(node.keys())}"
        )
    if not isinstance(offers, list):
        raise SchemaError(f"'offers' is {type(offers).__name__}, expected list")

    return offers

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

    try:
        edges = _validate_search_response(data)
    except SchemaError as e:
        msg = f"JustWatch search schema changed for '{title}': {e}"
        print(f'\n  ⚠️  {msg}')
        _send_telegram_alert(msg)
        return None

    if not edges:
        return False

    want = 'Movie' if content_type != 'tv' else 'Show'

    # Find the top result of the correct type whose title actually matches
    top_match = None
    for e in edges:
        node = e.get('node') or {}
        if node.get('__typename') != want:
            continue
        content = node.get('content') or {}
        if _titles_match(title, content.get('title', ''), content.get('originalTitle', '')):
            top_match = node.get('id')
            break

    if not top_match:
        return False

    offer_data = gql(OFFERS_Q, {'nodeId': top_match, 'country': 'IN'})
    if not offer_data:
        return None

    try:
        offers = _validate_offers_response(offer_data)
    except SchemaError as e:
        msg = f"JustWatch offers schema changed for '{title}': {e}"
        print(f'\n  ⚠️  {msg}')
        _send_telegram_alert(msg)
        return None

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
