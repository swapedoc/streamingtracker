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

def send_user_alerts(hits: list[dict]) -> tuple[int, int, set[str]]:
    """
    Send individual Telegram alerts to each user who tracked a title.
    Each row has its own telegram_chat_id — users get only their titles.
    Returns (sent_count, failed_count, sent_chat_ids).
    sent_chat_ids: set of chat_ids whose message was delivered successfully.
    Caller uses this to mark only those rows notified=True (Bug #8 fix).
    """
    if not hits:
        return 0, 0, set()

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
    sent_chat_ids: set[str] = set()
    for chat_id, user_hits in by_user.items():
        lines = ['🎬 <b>StreamIQ — Available to Stream Now!</b>\n']
        for h in user_hits:
            emoji = '🎥' if h['content_type'] != 'tv' else '📺'
            # Only add the URL line when a stream_url is present — JustWatch
            # sometimes returns no standardWebURL, and an empty line looks broken.
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
            sent_chat_ids.add(chat_id)
            print(f'  📨 Sent to chat {chat_id} ({len(user_hits)} title(s))')
        else:
            failed += 1
            print(f'  ❌ Failed to send to chat {chat_id}')

    return sent, failed, sent_chat_ids

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
    rows = []
    for _start in range(0, 100_000, 1000):
        page = (
            db.table('watchlist')
            .select('id, browser_id, title, content_type, platform, stream_url, telegram_chat_id')
            .eq('notified', False)
            .is_('platform', 'null')
            .range(_start, _start + 999)
            .execute()
        )
        batch = page.data or []
        rows.extend(batch)
        if len(batch) < 1000:
            break

    # Also include rows that had a platform but notified=False (newly added Watch Now items)
    rows_with_platform = []
    for _start in range(0, 100_000, 1000):
        page = (
            db.table('watchlist')
            .select('id, browser_id, title, content_type, platform, stream_url, telegram_chat_id')
            .eq('notified', False)
            .not_.is_('platform', 'null')
            .range(_start, _start + 999)
            .execute()
        )
        batch = page.data or []
        rows_with_platform.extend(batch)
        if len(batch) < 1000:
            break

    # Rows with a platform already set were tracked directly from Watch Now —
    # they're already streaming. Send Telegram alerts first, then mark notified.
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
            sent_wn, failed_wn, sent_chat_ids_wn = send_user_alerts(watch_now_hits)
            print(f'  ✅ Watch Now alerts: sent to {sent_wn} user(s)' +
                  (f', ❌ {failed_wn} failed' if failed_wn else ''))
        else:
            print(f'  ℹ️  {len(rows_with_platform)} Watch Now item(s) had no telegram_chat_id — skipped')
            sent_chat_ids_wn = set()
        # Only mark rows whose specific chat_id succeeded — a partial Telegram
        # failure (some sent, some failed) must not mark the failed rows as notified.
        ids_no_tg   = [r['id'] for r in rows_with_platform if not r.get('telegram_chat_id')]
        ids_alerted = [r['id'] for r in rows_with_platform
                       if r.get('telegram_chat_id') in sent_chat_ids_wn]
        ids_to_mark = ids_no_tg + ids_alerted
        if ids_to_mark:
            db.table('watchlist').update({'notified': True}).in_('id', ids_to_mark).execute()
            print(f'  ✅ Marked {len(ids_to_mark)} Watch Now items as notified')

    if not rows:
        print('  ✨ No pending Discover items to check.')
        print('\n' + '='*60)
        return

    # Deduplicate by (title, content_type) — different users may track the same
    # title as both a movie and a TV show, requiring separate JustWatch lookups.
    unique: dict[tuple, list] = {}
    for r in rows:
        key = (r['title'], r.get('content_type') or 'movie')
        unique.setdefault(key, []).append(r)

    print(f'  📋 {len(rows)} rows ({len(unique)} unique titles) to check against JustWatch\n')

    hits         = []   # titles that are now streaming
    hit_row_ids  = {}   # row_id → telegram_chat_id; only rows with a successful send are marked notified
    misses       = []   # still not found
    errors       = []   # API failures

    def check_one(title_rows):
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
                # Update platform + stream_url only — do NOT set notified=True here.
                # notified is set in bulk AFTER send_user_alerts() succeeds so that
                # a Telegram failure never silently swallows the alert (mirrors WA1 fix).
                for row in row_list:
                    try:
                        db.table('watchlist').update({
                            'platform':   plat,
                            'stream_url': url or '',
                        }).eq('id', row['id']).execute()
                    except Exception as e:
                        print(f'     ❌ DB update failed for row {row["id"]}: {e}')
                # Collect row_id → chat_id so we can mark only successfully-alerted rows
                for row in row_list:
                    hit_row_ids[row['id']] = row.get('telegram_chat_id')
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
        sent, failed, sent_chat_ids = send_user_alerts(hits)
        print(f'  ✅ Sent to {sent} user(s)' + (f', ❌ {failed} failed' if failed else ''))
        # Only mark rows as notified if their specific chat_id succeeded.
        ids_to_notify = [
            row_id for row_id, chat_id in hit_row_ids.items()
            if chat_id in sent_chat_ids
        ] if isinstance(hit_row_ids, dict) else hit_row_ids
        if ids_to_notify:
            try:
                db.table('watchlist').update({'notified': True}).in_('id', ids_to_notify).execute()
                print(f'  ✅ Marked {len(ids_to_notify)} row(s) as notified')
            except Exception as e:
                print(f'  ❌ Failed to mark rows notified: {e}')
    else:
        print('\n💤 No new arrivals — no Telegram message sent')

    print('\n' + '='*60)
    print('🎉 Done!')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
