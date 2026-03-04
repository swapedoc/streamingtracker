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
from constants import JUSTWATCH_PLATFORM_MAP, JUSTWATCH_PRIORITY

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

# ── Package → readable name — imported from constants.py ────────────────────
PLATFORM_MAP = JUSTWATCH_PLATFORM_MAP

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
            if r.status_code == 429:
                # JustWatch rate limit — back off progressively, same as fetch_hindi_dubs.py
                wait = 5 * (i + 1)
                print(f'   ⚡️ JustWatch rate limited (429) — waiting {wait}s')
                time.sleep(wait)
                continue
        except Exception as e:
            print(f'   ⚡️ JustWatch GQL error: {e}')
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
    priority = JUSTWATCH_PRIORITY
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
            # FIX NEW-14: only include the URL line when a stream_url is actually
            # present. When JustWatch returns no standardWebURL the old code
            # rendered an empty line in the Telegram message — no clickable link,
            # just blank whitespace below the platform name.
            url_line = f'   {h["stream_url"]}\n' if h.get('stream_url') else ''
            lines.append(
                f'{emoji} <b>{h["title"]}</b>\n'
                f'   ▶ Now on <b>{h["platform"]}</b>\n'
                + url_line
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

    # FIX WA1: rows_with_platform had a platform set but notified=False.
    # These are titles the user tracked directly from Watch Now — they are
    # already streaming, so we know the platform/url immediately.
    # BUG: the old code silently marked them notified=True with NO Telegram
    # alert. Users who tracked from Watch Now never received any notification.
    # FIX: queue them into watch_now_hits first, send alerts, THEN mark notified.
    watch_now_hits = []
    if rows_with_platform:
        print(f'  📺 {len(rows_with_platform)} Watch Now item(s) tracked — alerting users now...')
        for r in rows_with_platform:
            if r.get('platform') and r.get('telegram_chat_id'):
                watch_now_hits.append({
                    'title':            r['title'],
                    'content_type':     r.get('content_type', 'movie'),
                    'platform':         r['platform'],
                    'stream_url':       r.get('stream_url') or '',
                    'telegram_chat_id': r['telegram_chat_id'],
                })
        if watch_now_hits:
            sent_wn, failed_wn = send_user_alerts(watch_now_hits)
            print(f'  ✅ Watch Now alerts: sent to {sent_wn} user(s)' +
                  (f', ❌ {failed_wn} failed' if failed_wn else ''))
        else:
            print(f'  ℹ️  {len(rows_with_platform)} Watch Now item(s) had no telegram_chat_id — skipped')
        # Mark all notified after alerts are sent (or if no chat_id to notify)
        ids = [r['id'] for r in rows_with_platform]
        db.table('watchlist').update({'notified': True}).in_('id', ids).execute()
        print(f'  ✅ Marked {len(ids)} Watch Now items as notified')

    if not rows:
        print('  ✨ No pending Discover items to check.')
        print('\n' + '='*60)
        return

    # FIX WA2: deduplicate by (title, content_type) not just title.
    # Keying by title alone meant that if User A tracked "Flesh and Blood" as
    # a movie and User B tracked it as a TV show, only ONE JustWatch query
    # fired — using User A's content_type. User B got the wrong result or missed
    # the alert entirely. Tuple key fixes the lookup for both users independently.
    unique: dict[tuple, list] = {}
    for r in rows:
        key = (r['title'], r.get('content_type') or 'movie')
        unique.setdefault(key, []).append(r)

    print(f'  📋 {len(rows)} rows ({len(unique)} unique titles) to check against JustWatch\n')

    hits    = []   # titles that are now streaming
    misses  = []   # still not found
    errors  = []   # API failures

    def check_one(title_rows):
        # FIX WA2: key is now (title, content_type) tuple — unpack both parts
        (title, ct), row_list = title_rows
        try:
            plat, url = check_streaming(title, ct)
            return title, ct, row_list, plat, url, None
        except Exception as e:
            return title, ct, row_list, None, None, str(e)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(check_one, item): item for item in unique.items()}
        done = 0
        for future in as_completed(futures):
            title, ct, row_list, plat, url, err = future.result()
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
