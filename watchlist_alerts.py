#!/usr/bin/env python3
"""
watchlist_alerts.py — Daily job: check if tracked titles are now streaming,
send a Telegram notification, and mark them as notified.

SETUP:
  pip install requests python-dotenv supabase

USAGE:
  python3 watchlist_alerts.py              # run once
  # Or add to cron (runs at 9am IST = 3:30am UTC):
  # 30 3 * * * /usr/bin/python3 /path/to/watchlist_alerts.py >> /var/log/streamiq_alerts.log 2>&1

REQUIRED .env:
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_KEY=<service_role_key>          # NOT the anon key — needs UPDATE access
  TELEGRAM_BOT_TOKEN=<bot_token>
  TELEGRAM_BOT_TOKEN=<bot_token>   # one bot for all users
  # No global TELEGRAM_CHAT_ID needed — each user provides their own

HOW IT WORKS:
  1. Fetch all watchlist rows where notified=FALSE and platform IS NULL
     (platform=NULL means "not yet on a service" when user tracked from Discover)
  2. For each unique title, call JustWatch India GraphQL (same as fetch_hindi_dubs.py)
     to check if it has any streaming offers.
  3. If found: update the watchlist row with platform+stream_url, send Telegram alert.
  4. Mark notified=TRUE so we never double-alert.
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

SUPABASE_URL        = os.getenv('SUPABASE_URL')
SUPABASE_KEY        = os.getenv('SUPABASE_KEY')   # service role key
TELEGRAM_BOT_TOKEN  = os.getenv('TELEGRAM_BOT_TOKEN')

GQL         = 'https://apis.justwatch.com/graphql'
MAX_WORKERS = 10

# ── JustWatch GraphQL ────────────────────────────────────────────────────────

SEARCH_Q = """
query SearchTitles($searchQuery: String!, $country: Country!, $first: Int!) {
  popularTitles(country: $country, filter: { searchQuery: $searchQuery }, first: $first) {
    edges {
      node {
        id __typename
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
        package { shortName clearName }
        standardWebURL
      }
    }
    ... on Show {
      offers(country: $country, platform: WEB) {
        audioLanguages
        package { shortName clearName }
        standardWebURL
      }
    }
  }
}
"""

# ── Package → readable name ──────────────────────────────────────────────────

PLATFORM_MAP = {
    'nfx':  ('Netflix',      'https://www.netflix.com'),
    'prv':  ('Prime Video',  'https://www.primevideo.com'),
    'hst':  ('Jiohotstar',   'https://www.jiohotstar.com'),
    'atp':  ('Apple TV+',    'https://tv.apple.com'),
    'jic':  ('JioCinema',    'https://www.jiocinema.com'),
    'mxs':  ('Max',          'https://www.max.com'),
    'dnp':  ('Disney+',      'https://www.disneyplus.com'),
}

# ── Thread-local HTTP session ────────────────────────────────────────────────

_local = threading.local()

def _session():
    if not hasattr(_local, 's'):
        s = requests.Session()
        s.headers.update({
            'User-Agent':   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Content-Type': 'application/json',
            'Origin':       'https://www.justwatch.com',
            'Referer':      'https://www.justwatch.com/',
        })
        _local.s = s
    return _local.s

def gql(query, variables, retries=3):
    for i in range(retries):
        try:
            r = _session().post(GQL, json={'query': query, 'variables': variables}, timeout=15)
            if r.status_code == 200:
                d = r.json()
                return None if 'errors' in d else d
        except Exception:
            pass
        if i < retries - 1:
            time.sleep(1)
    return None

# ── Title matching (same logic as fetch_hindi_dubs.py) ───────────────────────

import re

def _norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(the|a|an)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

def _match(search, result, original=''):
    needle = _norm(search)
    for c in filter(None, [_norm(result), _norm(original)]):
        if needle == c:
            return True
        if len(needle) >= 6 and (needle in c or c in needle):
            return True
    return False

# ── Core check function ──────────────────────────────────────────────────────

def check_streaming(title: str, content_type: str):
    """
    Returns (platform_name, stream_url) if title is now on a streaming service,
    or (None, None) if not found, or raises on API error.
    """
    want = 'Movie' if content_type != 'tv' else 'Show'
    data = gql(SEARCH_Q, {'searchQuery': title, 'country': 'IN', 'first': 5})
    if not data:
        return None, None

    edges = data.get('data', {}).get('popularTitles', {}).get('edges', [])
    top_match = None
    for e in edges:
        node = e['node']
        if node.get('__typename') != want:
            continue
        c = node.get('content') or {}
        if _match(title, c.get('title', ''), c.get('originalTitle', '')):
            top_match = node['id']
            break

    if not top_match:
        return None, None

    offer_data = gql(OFFERS_Q, {'nodeId': top_match, 'country': 'IN'})
    if not offer_data:
        return None, None

    offers = (offer_data.get('data', {}).get('node') or {}).get('offers', [])
    if not offers:
        return None, None

    # Prefer known major platforms in priority order
    priority = ['nfx', 'prv', 'hst', 'atp', 'jic', 'mxs', 'dnp']
    best = None
    for short in priority:
        for o in offers:
            if (o.get('package') or {}).get('shortName') == short:
                best = o
                break
        if best:
            break
    if not best:
        best = offers[0]  # fallback: first available offer

    pkg        = best.get('package') or {}
    short      = pkg.get('shortName', '')
    plat_name  = PLATFORM_MAP.get(short, (pkg.get('clearName', 'Streaming'), ''))[0]
    stream_url = best.get('standardWebURL') or PLATFORM_MAP.get(short, ('', ''))[1]
    return plat_name, stream_url

# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(chat_id: str, message: str) -> bool:
    """Send a message to a specific user's Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    try:
        r = requests.post(url, json={
            'chat_id':    chat_id,
            'text':       message,
            'parse_mode': 'HTML',
        }, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f'  ❌ Telegram error for chat {chat_id}: {e}')
        return False

def send_user_alerts(hits: list[dict]) -> tuple[int, int]:
    """
    Send individual Telegram alerts to each user who tracked a title.
    Each row has its own telegram_chat_id — users get only their titles.
    Returns (sent_count, failed_count).
    """
    if not hits:
        return 0, 0

    # Group hits by telegram_chat_id so each user gets one message
    from collections import defaultdict
    by_user: dict[str, list] = defaultdict(list)
    no_tg = 0
    for h in hits:
        tg = h.get('telegram_chat_id')
        if tg:
            by_user[tg].append(h)
        else:
            no_tg += 1

    if no_tg:
        print(f'  ⚠️  {no_tg} row(s) had no telegram_chat_id — skipped')

    sent = failed = 0
    for chat_id, user_hits in by_user.items():
        lines = ['🎬 <b>StreamIQ — Available to Stream Now!</b>\n']
        for h in user_hits:
            emoji = '🎥' if h['content_type'] != 'tv' else '📺'
            lines.append(
                f'{emoji} <b>{h["title"]}</b>\n'
                f'   ▶ Now on <b>{h["platform"]}</b>\n'
                f'   {h["stream_url"]}\n'
            )
        lines.append('\n<i>Manage your watchlist on StreamIQ</i>')
        ok = send_telegram(chat_id, '\n'.join(lines))
        if ok:
            sent += 1
            print(f'  📨 Sent to chat {chat_id} ({len(user_hits)} title(s))')
        else:
            failed += 1
            print(f'  ❌ Failed to send to chat {chat_id}')

    return sent, failed

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print('❌ Missing SUPABASE_URL or SUPABASE_KEY in .env')
        sys.exit(1)

    db = create_client(SUPABASE_URL, SUPABASE_KEY)

    print('\n' + '='*60)
    print('🔔  StreamIQ Watchlist Alert Runner')
    print('='*60)

    # Fetch un-notified rows that have no streaming platform yet
    print('\n📥 Fetching pending watchlist items...')
    result = (
        db.table('watchlist')
        .select('id, browser_id, title, content_type, platform, stream_url, telegram_chat_id')
        .eq('notified', False)
        .is_('platform', 'null')   # only Discover items (no platform yet)
        .execute()
    )
    rows = result.data or []

    # Also include rows that had a platform but notified=False (newly added Watch Now items)
    result2 = (
        db.table('watchlist')
        .select('id, browser_id, title, content_type, platform, stream_url, telegram_chat_id')
        .eq('notified', False)
        .not_.is_('platform', 'null')
        .execute()
    )
    rows_with_platform = result2.data or []

    # Mark rows that already have a platform as notified immediately
    # (user tracked something that was already streaming — no need to re-alert)
    if rows_with_platform:
        ids = [r['id'] for r in rows_with_platform]
        db.table('watchlist').update({'notified': True}).in_('id', ids).execute()
        print(f'  ✅ Marked {len(ids)} already-streaming items as notified (no alert needed)')

    if not rows:
        print('  ✨ No pending Discover items to check.')
        print('\n' + '='*60)
        return

    # Deduplicate by title (multiple users may track the same title)
    unique: dict[str, list] = {}
    for r in rows:
        unique.setdefault(r['title'], []).append(r)

    print(f'  📋 {len(rows)} rows ({len(unique)} unique titles) to check against JustWatch\n')

    hits    = []   # titles that are now streaming
    misses  = []   # still not found
    errors  = []   # API failures

    def check_one(title_rows):
        title, row_list = title_rows
        ct = row_list[0].get('content_type', 'movie')
        try:
            plat, url = check_streaming(title, ct)
            return title, row_list, plat, url, None
        except Exception as e:
            return title, row_list, None, None, str(e)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(check_one, item): item for item in unique.items()}
        done = 0
        for future in as_completed(futures):
            title, row_list, plat, url, err = future.result()
            done += 1
            pct = done / len(unique) * 100
            print(f'  [{done:3d}/{len(unique)}] ({pct:3.0f}%) {title[:50]}', end=' ')

            if err:
                print(f'⚠️  error: {err}')
                errors.append(title)
                continue

            if plat:
                print(f'✅ {plat}')
                # Update each matching row in DB
                for row in row_list:
                    try:
                        db.table('watchlist').update({
                            'platform':  plat,
                            'stream_url': url or '',
                            'notified':  True,
                        }).eq('id', row['id']).execute()
                    except Exception as e:
                        print(f'     ❌ DB update failed for row {row["id"]}: {e}')
                # Each row gets its own entry so per-user grouping works
                for row in row_list:
                    hits.append({'title': title, 'content_type': row.get('content_type','movie'),
                                 'platform': plat, 'stream_url': url or '',
                                 'telegram_chat_id': row.get('telegram_chat_id')})
            else:
                print('— not yet')
                misses.append(title)

    # Send consolidated Telegram alert
    print(f'\n📊 Summary: {len(hits)} new arrivals · {len(misses)} still pending · {len(errors)} errors')

    if hits:
        print(f'\n📨 Sending Telegram alerts for {len(hits)} row(s)...')
        sent, failed = send_user_alerts(hits)
        print(f'  ✅ Sent to {sent} user(s)' + (f', ❌ {failed} failed' if failed else ''))
    else:
        print('\n💤 No new arrivals — no Telegram message sent')

    print('\n' + '='*60)
    print('🎉 Done!')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
