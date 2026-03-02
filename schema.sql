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

-- ── Watch Now content table ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS content (
    id               BIGSERIAL PRIMARY KEY,
    tmdb_id          INTEGER NOT NULL UNIQUE,
    title            TEXT NOT NULL,
    original_title   TEXT,
    platform         TEXT NOT NULL,
    content_type     TEXT NOT NULL,             -- 'movie' | 'tv'
    release_year     INTEGER,
    imdb_rating      FLOAT,
    poster_path      TEXT,
    overview         TEXT,
    discovery_source TEXT,                       -- 'trending' | 'catalog' etc.
    genre            TEXT,
    tv_genre         TEXT,
    stream_url       TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_content_platform ON content(platform);
CREATE INDEX IF NOT EXISTS idx_content_type     ON content(content_type);

-- ── Reviews table ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reviews (
    id                 BIGSERIAL PRIMARY KEY,
    content_id         BIGINT NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    source             TEXT NOT NULL,            -- 'youtube' | 'reddit' | 'rotten_tomatoes' etc.
    source_id          TEXT NOT NULL,
    source_url         TEXT,
    reviewer           TEXT,
    review_text        TEXT,
    sentiment          INTEGER,                  -- -1 | 0 | 1
    confidence         FLOAT,
    weighted_sentiment FLOAT,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_content_id ON reviews(content_id);
CREATE INDEX IF NOT EXISTS idx_reviews_source      ON reviews(source);

-- ── Scores table ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scores (
    id               BIGSERIAL PRIMARY KEY,
    content_id       BIGINT NOT NULL UNIQUE REFERENCES content(id) ON DELETE CASCADE,
    youtube_score    FLOAT,
    reddit_score     FLOAT,
    imdb_score       FLOAT,
    engagement_score FLOAT DEFAULT 0.0,
    final_score      FLOAT,
    label            TEXT,
    category         TEXT,
    review_count     INTEGER DEFAULT 0,
    positive_ratio   FLOAT,
    is_polarizing    BOOLEAN DEFAULT FALSE,
    sentiment_std    FLOAT,
    vibe_score       FLOAT,                      -- 1.0–10.0 genre-specific intensity
    vibe_label       TEXT,                       -- e.g. 'Scare Factor', 'Laugh Meter'
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scores_final_score ON scores(final_score DESC);

-- ── Watchlist table ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS watchlist (
    id               BIGSERIAL PRIMARY KEY,
    browser_id       TEXT NOT NULL,              -- anonymous browser fingerprint
    title            TEXT NOT NULL,
    content_type     TEXT,                       -- 'movie' | 'tv'
    platform         TEXT,                       -- NULL until streaming is confirmed
    stream_url       TEXT,
    notified         BOOLEAN DEFAULT FALSE,
    telegram_chat_id TEXT,                       -- user's personal Telegram chat ID
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_watchlist_browser_id ON watchlist(browser_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_notified   ON watchlist(notified);

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
