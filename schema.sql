-- ─────────────────────────────────────────────────────────────────────────────
-- StreamIQ — Database Schema
-- Run once in Supabase SQL editor to set up all required tables and columns.
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Core discover content table ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS discover_content (
    id BIGSERIAL PRIMARY KEY,
    tmdb_id INTEGER NOT NULL,
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
    stream_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tmdb_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_discover_category ON discover_content(category);
CREATE INDEX IF NOT EXISTS idx_discover_platform ON discover_content(platform);
CREATE INDEX IF NOT EXISTS idx_discover_genre ON discover_content(genre);

-- ── Binge Time + Trailer columns (run once) ───────────────────────────────────
ALTER TABLE content          ADD COLUMN IF NOT EXISTS runtime          INTEGER;  -- minutes (movies)
ALTER TABLE content          ADD COLUMN IF NOT EXISTS seasons          INTEGER;  -- TV only
ALTER TABLE content          ADD COLUMN IF NOT EXISTS episode_count    INTEGER;  -- TV only
ALTER TABLE content          ADD COLUMN IF NOT EXISTS episode_runtime  INTEGER;  -- mins per ep, TV only
ALTER TABLE content          ADD COLUMN IF NOT EXISTS trailer_id       TEXT;     -- YouTube video ID
ALTER TABLE content          ADD COLUMN IF NOT EXISTS genre            TEXT;     -- resolved genre label
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS runtime          INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS seasons          INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS episode_count    INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS episode_runtime  INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS trailer_id       TEXT;

-- ── Leaving Soon columns (run once) ─────────────────────────────────────────
-- Populated by JustWatchFetcher from the validUntil field on each offer.
-- NULL = no expiry date reported by JustWatch (does NOT mean it's staying forever).
ALTER TABLE content          ADD COLUMN IF NOT EXISTS leaving_date DATE;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS leaving_date DATE;

-- ── Vibe Score columns (run once) ────────────────────────────────────────────
ALTER TABLE scores ADD COLUMN IF NOT EXISTS vibe_score FLOAT;   -- 1.0 to 10.0
ALTER TABLE scores ADD COLUMN IF NOT EXISTS vibe_label TEXT;    -- e.g. "Scare Factor", "Laugh Meter"

-- ── Full review text (run once) ───────────────────────────────────────────────
-- review_text was previously truncated to 800-1000 chars before saving.
-- It is now stored at full length so you can re-run LLM prompts without re-scraping.
-- If your column is currently VARCHAR(n), widen it:
ALTER TABLE reviews ALTER COLUMN review_text TYPE TEXT;

-- ── Manual trailer overrides (run once) ──────────────────────────────────────
-- Use when TMDb has no trailer for a title you know has one.
-- Get tmdb_id from: themoviedb.org → search title → number in URL
-- Get trailer_id from: youtube.com/watch?v=XXXXXXXXXX → the part after v=
CREATE TABLE IF NOT EXISTS trailer_overrides (
    tmdb_id    INTEGER PRIMARY KEY,
    trailer_id TEXT NOT NULL,
    note       TEXT,   -- optional: title name for your reference
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Hindi dub column (run once) ─────────────────────────────────────────────
ALTER TABLE content          ADD COLUMN IF NOT EXISTS hindi_dub BOOLEAN DEFAULT FALSE;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS hindi_dub BOOLEAN DEFAULT FALSE;

-- ── Cache invalidation (run once) ────────────────────────────────────────────
-- Single-row table bumped by GitHub Actions after every data pipeline run.
-- Frontend checks this before deciding whether to re-fetch the heavy W/D arrays.
CREATE TABLE IF NOT EXISTS sync_state (
    id           BOOL PRIMARY KEY DEFAULT TRUE CHECK (id),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO sync_state (last_updated) VALUES (NOW())
ON CONFLICT (id) DO NOTHING;
