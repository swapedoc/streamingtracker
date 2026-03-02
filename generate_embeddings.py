#!/usr/bin/env python3
"""
generate_embeddings.py — Batch Gemini embeddings (100 titles per API call).
At 15 req/min free tier: 3000 titles = ~20 minutes instead of 3.5 hours.

SETUP:  pip install requests python-dotenv supabase
USAGE:
  python3 generate_embeddings.py         # process all missing
  python3 generate_embeddings.py --force # re-embed everything
"""

import os, sys, time, argparse
from dotenv import load_dotenv
from supabase import create_client
import requests

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
GEMINI_KEY   = os.getenv('GEMINI_API_KEY')
MODEL        = 'gemini-embedding-001'   # 3072-dim
BATCH_SIZE   = 100    # Gemini allows up to 100 texts per batchEmbedContents call
MIN_INTERVAL = 4.5    # seconds between API calls (free tier: 15 req/min)


def build_text(row: dict) -> str:
    parts = [f"Title: {row.get('title', '')}"]
    if row.get('genre'):        parts.append(f"Genre: {row['genre']}")
    if row.get('tv_genre'):     parts.append(f"Genre: {row['tv_genre']}")
    if row.get('content_type'): parts.append(f"Type: {'Series' if row['content_type']=='tv' else 'Film'}")
    if row.get('release_year'): parts.append(f"Year: {row['release_year']}")
    if row.get('overview'):     parts.append(f"Synopsis: {str(row['overview'])[:400]}")
    return '. '.join(parts)


def embed_batch(texts: list[str], retries=5) -> list[list[float]] | None:
    """Send up to 100 texts in one batchEmbedContents request."""
    # FIX GE1 / EF9: pass the API key as the x-goog-api-key request header,
    # not as a ?key= URL query param. URL params appear in GitHub Actions logs,
    # server access logs, and any HTTP proxy/debug traces — anyone with log
    # access could extract and abuse the key. The Gemini API accepts the key
    # either way; the header form never appears in logs.
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents'
    headers = {
        'Content-Type':   'application/json',
        'x-goog-api-key': GEMINI_KEY,
    }
    body = {
        'requests': [
            {'model': f'models/{MODEL}', 'content': {'parts': [{'text': t}]}}
            for t in texts
        ]
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=60)
            if r.status_code == 429:
                wait = min(60, 5 * (attempt + 1))
                print(f'\n  ⏳ Rate limited — waiting {wait}s...')
                time.sleep(wait)
                continue
            if not r.ok:
                print(f'\n  ❌ HTTP {r.status_code}: {r.text[:200]}')
                time.sleep(3)
                continue
            data = r.json()
            embeddings = data.get('embeddings', [])
            if len(embeddings) != len(texts):
                print(f'\n  ⚠️  Expected {len(texts)} embeddings, got {len(embeddings)}')
                return None
            return [e['values'] for e in embeddings]
        except Exception as e:
            print(f'\n  ❌ Exception: {e}')
            if attempt == retries - 1:
                return None
            time.sleep(3)
    return None


def fetch_rows(db, force: bool) -> list:
    rows, offset = [], 0
    print('📥 Fetching rows from Supabase...')
    while True:
        q = db.table('discover_content').select('id,title,genre,tv_genre,content_type,release_year,overview')
        if not force:
            q = q.is_('embedding', 'null')
        batch = q.range(offset, offset + 999).execute().data or []
        rows.extend(batch)
        print(f'   {len(rows)} fetched...', end='\r')
        if len(batch) < 1000:
            break
        offset += 1000
    print(f'   Total: {len(rows)} rows{" "*30}')
    return rows


def run_incremental(db) -> tuple[int, int]:
    """
    Embed any discover_content rows that are missing an embedding.
    Designed to be called from streaming_tracker.py at the end of each run
    so new titles are immediately searchable via Vibe Search.

    Returns (success_count, error_count).
    Rate-limited to 15 req/min (MIN_INTERVAL between batches) — free Gemini tier.
    If you're on a paid tier, lower MIN_INTERVAL in this file.
    """
    if not GEMINI_KEY:
        print('\n⚠️  GEMINI_API_KEY not set — skipping embedding step')
        return 0, 0

    rows = fetch_rows(db, force=False)
    if not rows:
        print('\n✅ Embeddings: all discover titles already embedded')
        return 0, 0

    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    print(f'\n🔮 Embedding {len(rows)} new title(s) in {len(batches)} batch call(s)...')

    success = errors = 0
    last_call = 0.0

    for bidx, batch in enumerate(batches):
        texts = [build_text(r) for r in batch]

        wait = MIN_INTERVAL - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.time()

        embeddings = embed_batch(texts)

        if embeddings is None:
            errors += len(batch)
            print(f'   ❌ Batch {bidx + 1}/{len(batches)} failed — {len(batch)} titles skipped')
            continue

        # FIX NEW-13: replace per-row UPDATE calls with a single batch upsert.
        # The old code did one Supabase HTTP round-trip per row — 3000+ calls
        # for a full catalog run, adding 5+ minutes of pure DB write time and
        # risking Supabase connection rate-limiting mid-run.
        # upsert on conflict='id' updates only the embedding column and is
        # equivalent to the old UPDATE ... WHERE id = ? for each row.
        try:
            updates = [{'id': row['id'], 'embedding': emb}
                       for row, emb in zip(batch, embeddings)]
            db.table('discover_content').upsert(updates, on_conflict='id').execute()
            success += len(batch)
        except Exception as e:
            errors += len(batch)
            print(f'   ⚠️  Batch {bidx + 1} DB upsert failed: {e}')

        print(f'   ✅ Batch {bidx + 1}/{len(batches)} done ({success} embedded so far)')

    print(f'   Embedded: {success}  Errors: {errors}')
    return success, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Re-embed all rows')
    args = parser.parse_args()

    if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_KEY]):
        print('❌ Missing SUPABASE_URL, SUPABASE_KEY, or GEMINI_API_KEY in .env')
        sys.exit(1)

    db = create_client(SUPABASE_URL, SUPABASE_KEY)

    total_batches = None
    eta_str = '?'

    print(f'\n{"="*60}')
    print(f'🔮  StreamIQ — Batch Embedding Backfiller')
    print(f'    Model     : {MODEL} (3072 dim)')
    print(f'    Batch size: {BATCH_SIZE} titles per API call')
    print(f'{"="*60}\n')

    rows = fetch_rows(db, args.force)
    if not rows:
        print('✅ All rows already embedded. Use --force to redo.')
        return

    # Split into batches of BATCH_SIZE
    batches = [rows[i:i+BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    total_batches = len(batches)
    est_min = (total_batches * MIN_INTERVAL) / 60
    print(f'   {len(rows)} titles → {total_batches} API calls (~{est_min:.0f} min at free tier)\n')

    success = errors = 0
    start   = time.time()
    last_call = 0.0

    for bidx, batch in enumerate(batches):
        texts = [build_text(r) for r in batch]

        # Throttle — enforce MIN_INTERVAL between calls
        wait = MIN_INTERVAL - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.time()

        embeddings = embed_batch(texts)

        elapsed   = time.time() - start
        calls_done = bidx + 1
        rate      = calls_done / max(elapsed, 0.1)   # calls/sec
        eta_s     = (total_batches - calls_done) / max(rate, 0.001)
        eta       = f'{int(eta_s//60):02d}:{int(eta_s%60):02d}'
        pct       = calls_done / total_batches * 100
        bar       = '█'*int(30*calls_done/total_batches) + '░'*(30-int(30*calls_done/total_batches))

        if embeddings is None:
            errors += len(batch)
            sys.stdout.write(f'\r{" "*110}\r')
            print(f'  ❌ Batch {bidx+1} failed — {len(batch)} titles skipped')
        else:
            # FIX NEW-13: batch upsert instead of per-row UPDATE calls.
            # Same fix as run_incremental() — 1 DB call per 100-row batch
            # instead of 100 individual calls. See run_incremental() for details.
            batch_errors = 0
            try:
                updates = [{'id': row['id'], 'embedding': emb}
                           for row, emb in zip(batch, embeddings)]
                db.table('discover_content').upsert(updates, on_conflict='id').execute()
                success += len(batch)
            except Exception as e:
                batch_errors = len(batch)
                errors += len(batch)
                print(f'\n  ⚠️  Batch {bidx+1} DB upsert failed: {e}')
            if batch_errors:
                print(f'\n  ⚠️  {batch_errors} DB save errors in batch {bidx+1}')

        sys.stdout.write(
            f'\r  [{bar}] batch {calls_done}/{total_batches} ({pct:.0f}%)  '
            f'✅{success}/{len(rows)}  ❌{errors}  ETA {eta}   '
        )
        sys.stdout.flush()

    elapsed_total = time.time() - start
    print(f'\n\n{"="*60}')
    print(f'✅ Done in {elapsed_total/60:.1f} min')
    print(f'   Embedded : {success}')
    print(f'   Errors   : {errors}')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
