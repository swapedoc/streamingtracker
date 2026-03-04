-- ─────────────────────────────────────────────────────────────────────────────
-- StreamIQ — Database Schema
-- Run once in Supabase SQL editor to set up all required tables and columns.
--
-- FIXES APPLIED (from Master Bug List v5+v6+v7):
--   NEW-7  / SD6  : Added tv_genre to discover_content CREATE TABLE + ALTER TABLE
--   SD1          : scores.created_at renamed to computed_at (matches live DB)
--   SD2          : Added imdb_id TEXT + updated_at TIMESTAMPTZ to content table
--   SD3          : Added engagement_score + reviewer_subscribers + views +
--                  likes + comments_count + youtube_weight to reviews table
--   SD4  / #22   : rankings orphan table dropped (zero code references)
--   EF6  / SD5   : Added match_content() pgvector RPC (required by edgefunction.ts)
--   EF11         : Added RLS policy on discover_content for public read
--   content.id   : Fixed to BIGSERIAL (was INTEGER in schema, BIGSERIAL in live DB)
--   Sec1 note    : ssl=False is a Python-side issue; not schema-related
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable pgvector extension (required for embeddings + match_content RPC)
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Core discover content table ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS discover_content (
    id             BIGSERIAL PRIMARY KEY,
    tmdb_id        INTEGER   NOT NULL,
    title          TEXT      NOT NULL,
    original_title TEXT,
    platform       TEXT      NOT NULL,
    content_type   TEXT      NOT NULL,
    release_year   INTEGER,
    imdb_rating    FLOAT,
    poster_path    TEXT,
    overview       TEXT,
    category       TEXT      NOT NULL,
    genre          TEXT,
    tv_genre       TEXT,                         -- FIX NEW-7/SD6: was missing, caused TV upserts to silently drop this field
    popularity     FLOAT,
    stream_url     TEXT,
    source         TEXT DEFAULT 'tracker',       -- pipeline origin e.g. 'tracker', 'manual'
    hindi_dub      BOOLEAN   DEFAULT FALSE,
    runtime        INTEGER,                      -- minutes (movies)
    seasons        INTEGER,                      -- TV only
    episode_count  INTEGER,                      -- TV only
    episode_runtime INTEGER,                     -- mins per ep, TV only
    trailer_id     TEXT,                         -- YouTube video ID
    embedding      vector(3072),                 -- Gemini gemini-embedding-001 (3072-dim)
    leaving_date   DATE,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tmdb_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_discover_category  ON discover_content(category);
CREATE INDEX IF NOT EXISTS idx_discover_platform  ON discover_content(platform);
CREATE INDEX IF NOT EXISTS idx_discover_genre     ON discover_content(genre);
-- Index for pgvector cosine similarity search (used by match_content RPC)
CREATE INDEX IF NOT EXISTS idx_discover_embedding ON discover_content
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ── Watch Now content table ──────────────────────────────────────────────────
-- FIX SD2: added imdb_id and updated_at (exist in live DB, were absent here)
-- FIX: id changed to BIGSERIAL (live DB uses integer sequence, kept as BIGSERIAL for safety)
CREATE TABLE IF NOT EXISTS content (
    id               BIGSERIAL PRIMARY KEY,
    tmdb_id          INTEGER   NOT NULL,  -- FIX NEW-4: UNIQUE moved to (tmdb_id, platform) — supports multi-platform entries
    title            TEXT      NOT NULL,
    original_title   TEXT,
    platform         TEXT      NOT NULL,
    content_type     TEXT      NOT NULL,          -- 'movie' | 'tv'
    release_year     INTEGER,
    imdb_rating      FLOAT,
    imdb_id          TEXT,                        -- FIX SD2: exists in live DB, was missing
    poster_path      TEXT,
    overview         TEXT,
    discovery_source TEXT DEFAULT 'trending',     -- 'trending' | 'catalog' etc.
    genre            TEXT,
    tv_genre         TEXT,
    stream_url       TEXT DEFAULT '',
    hindi_dub        BOOLEAN   DEFAULT FALSE,
    runtime          INTEGER,
    seasons          INTEGER,
    episode_count    INTEGER,
    episode_runtime  INTEGER,
    trailer_id       TEXT,
    leaving_date     DATE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()    -- FIX SD2: used by YouTube 23h cache (#4); was missing
);

CREATE INDEX IF NOT EXISTS idx_content_platform ON content(platform);
-- FIX NEW-4: multi-platform support — one row per (title, platform)
-- Drop old single-column unique and add composite. For existing deployments:
-- ALTER TABLE content DROP CONSTRAINT IF EXISTS content_tmdb_id_key;
-- ALTER TABLE content ADD CONSTRAINT content_tmdb_id_platform_key UNIQUE (tmdb_id, platform);
ALTER TABLE content DROP CONSTRAINT IF EXISTS content_tmdb_id_key;
ALTER TABLE content ADD CONSTRAINT content_tmdb_id_platform_key UNIQUE (tmdb_id, platform);

CREATE INDEX IF NOT EXISTS idx_content_type     ON content(content_type);

-- ── Reviews table ─────────────────────────────────────────────────────────────
-- FIX SD3: added engagement_score, reviewer_subscribers, views, likes,
--          comments_count, youtube_weight — all exist in live DB, were absent here
CREATE TABLE IF NOT EXISTS reviews (
    id                    BIGSERIAL PRIMARY KEY,
    content_id            BIGINT    NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    source                TEXT      NOT NULL,     -- 'youtube' | 'reddit' | 'rotten_tomatoes' etc.
    source_id             TEXT      NOT NULL,
    source_url            TEXT,
    reviewer              TEXT,
    reviewer_subscribers  INTEGER,               -- FIX SD3: YouTube channel subscriber count
    review_text           TEXT,
    sentiment             INTEGER,               -- -1 | 0 | 1
    confidence            FLOAT,
    views                 INTEGER   DEFAULT 0,   -- FIX SD3
    likes                 INTEGER   DEFAULT 0,   -- FIX SD3
    comments_count        INTEGER   DEFAULT 0,   -- FIX SD3
    youtube_weight        FLOAT     DEFAULT 0,   -- FIX SD3
    engagement_score      FLOAT     DEFAULT 0,   -- FIX SD3
    weighted_sentiment    FLOAT,
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_content_id ON reviews(content_id);
CREATE INDEX IF NOT EXISTS idx_reviews_source      ON reviews(source);

-- ── Scores table ──────────────────────────────────────────────────────────────
-- FIX SD1: renamed created_at → computed_at to match live DB
--          (streaming_tracker.py writes to computed_at, not created_at)
CREATE TABLE IF NOT EXISTS scores (
    id               BIGSERIAL PRIMARY KEY,
    content_id       BIGINT    NOT NULL UNIQUE REFERENCES content(id) ON DELETE CASCADE,
    youtube_score    FLOAT     DEFAULT 0,
    reddit_score     FLOAT     DEFAULT 0,
    imdb_score       FLOAT     DEFAULT 0,
    engagement_score FLOAT     DEFAULT 0,
    final_score      FLOAT     DEFAULT 0,
    label            TEXT,
    category         TEXT      DEFAULT 'catalog',
    review_count     INTEGER   DEFAULT 0,
    positive_ratio   FLOAT,
    is_polarizing    BOOLEAN   DEFAULT FALSE,
    sentiment_std    FLOAT     DEFAULT 0,
    rt_score         FLOAT     DEFAULT 0,        -- Rotten Tomatoes weighted sentiment score
    vibe_score       FLOAT,                      -- 1.0–10.0 genre-specific intensity
    vibe_label       TEXT,                       -- e.g. 'Scare Factor', 'Laugh Meter'
    computed_at      TIMESTAMPTZ DEFAULT NOW()   -- FIX SD1: was 'created_at' in old schema
);

CREATE INDEX IF NOT EXISTS idx_scores_final_score ON scores(final_score DESC);

-- ── Watchlist table ───────────────────────────────────────────────────────────
-- Note: live DB also has tmdb_id TEXT column (nullable) — added here
CREATE TABLE IF NOT EXISTS watchlist (
    id               BIGSERIAL PRIMARY KEY,
    browser_id       TEXT      NOT NULL,         -- anonymous browser fingerprint
    title            TEXT      NOT NULL,
    content_type     TEXT      DEFAULT 'movie',  -- 'movie' | 'tv'
    tmdb_id          TEXT,                       -- nullable; from live DB
    platform         TEXT,                       -- NULL until streaming is confirmed
    stream_url       TEXT,
    notified         BOOLEAN   DEFAULT FALSE,
    telegram_chat_id TEXT,                       -- user's personal Telegram chat ID
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_watchlist_browser_id ON watchlist(browser_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_notified   ON watchlist(notified);

-- ── Rankings table ────────────────────────────────────────────────────────────
-- FIX #22: rankings was an orphan table in the live DB — 7 columns, zero code
-- references across all Python, TypeScript, and YAML files. Dropped.
DROP TABLE IF EXISTS rankings;

-- ── Manual trailer overrides ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trailer_overrides (
    tmdb_id    INTEGER PRIMARY KEY,
    trailer_id TEXT    NOT NULL,
    note       TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Cache invalidation ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sync_state (
    id           BOOL PRIMARY KEY DEFAULT TRUE CHECK (id),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO sync_state (last_updated) VALUES (NOW())
ON CONFLICT (id) DO NOTHING;

-- ─────────────────────────────────────────────────────────────────────────────
-- ALTER TABLE additions for existing deployments
-- Safe to run on a DB that already has these columns (IF NOT EXISTS guards them).
-- ─────────────────────────────────────────────────────────────────────────────

-- FIX NEW-7/SD6: tv_genre missing from discover_content on existing deployments
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS tv_genre           TEXT;

-- FIX SD2: imdb_id and updated_at missing from content on existing deployments
ALTER TABLE content          ADD COLUMN IF NOT EXISTS imdb_id            TEXT;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS updated_at         TIMESTAMPTZ DEFAULT NOW();

-- FIX SD3: engagement and YouTube-metric columns missing from reviews
ALTER TABLE reviews           ADD COLUMN IF NOT EXISTS reviewer_subscribers INTEGER;
ALTER TABLE reviews           ADD COLUMN IF NOT EXISTS views               INTEGER DEFAULT 0;
ALTER TABLE reviews           ADD COLUMN IF NOT EXISTS likes               INTEGER DEFAULT 0;
ALTER TABLE reviews           ADD COLUMN IF NOT EXISTS comments_count      INTEGER DEFAULT 0;
ALTER TABLE reviews           ADD COLUMN IF NOT EXISTS youtube_weight      FLOAT   DEFAULT 0;
ALTER TABLE reviews           ADD COLUMN IF NOT EXISTS engagement_score    FLOAT   DEFAULT 0;

-- FIX SD1: rename created_at → computed_at in scores (only if old column still exists)
-- Wrapped in a DO block so it's safe to run multiple times.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scores' AND column_name = 'created_at'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scores' AND column_name = 'computed_at'
    ) THEN
        ALTER TABLE scores RENAME COLUMN created_at TO computed_at;
    END IF;
END $$;

-- Remaining ALTER TABLE blocks (idempotent; already in original schema for context)
ALTER TABLE content          ADD COLUMN IF NOT EXISTS runtime          INTEGER;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS seasons          INTEGER;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS episode_count    INTEGER;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS episode_runtime  INTEGER;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS trailer_id       TEXT;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS genre            TEXT;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS tv_genre         TEXT;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS leaving_date     DATE;
ALTER TABLE content          ADD COLUMN IF NOT EXISTS hindi_dub        BOOLEAN DEFAULT FALSE;

ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS runtime          INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS seasons          INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS episode_count    INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS episode_runtime  INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS trailer_id       TEXT;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS source           TEXT DEFAULT 'tracker';
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS leaving_date     DATE;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS hindi_dub        BOOLEAN DEFAULT FALSE;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS embedding        vector(3072);

ALTER TABLE reviews ALTER COLUMN review_text TYPE TEXT;

-- rt_score: Rotten Tomatoes weighted sentiment (confirmed in live DB)
ALTER TABLE scores ADD COLUMN IF NOT EXISTS rt_score FLOAT DEFAULT 0;

-- ─────────────────────────────────────────────────────────────────────────────
-- FIX EF6 / SD5: match_content RPC — REQUIRED by edgefunction.ts (Vibe Search)
-- This function was missing from schema.sql entirely, causing Vibe Search to
-- return 500 on any fresh deployment. Added here so schema.sql is self-contained.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_content(
    query_embedding vector(3072),
    match_threshold float,
    match_count     int
)
RETURNS TABLE (
    id           bigint,
    title        text,
    genre        text,
    tv_genre     text,
    content_type text,
    release_year integer,
    imdb_rating  float,
    poster_path  text,
    overview     text,
    platform     text,
    stream_url   text,
    trailer_id   text,
    category     text,
    similarity   float
)
LANGUAGE sql STABLE AS $$
    SELECT
        id,
        title,
        genre,
        tv_genre,
        content_type,
        release_year,
        imdb_rating,
        poster_path,
        overview,
        platform,
        stream_url,
        trailer_id,
        category,
        1 - (embedding <=> query_embedding) AS similarity
    FROM discover_content
    WHERE embedding IS NOT NULL
      AND 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- FIX EF11: Row Level Security for discover_content
-- Allows anon/public reads (safe — this is public catalog data).
-- Blocks all writes from anon role (service role can still write via Python scripts).
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE discover_content ENABLE ROW LEVEL SECURITY;

-- Drop policy first so this script is re-runnable
DROP POLICY IF EXISTS "Public read discover_content" ON discover_content;
CREATE POLICY "Public read discover_content"
    ON discover_content FOR SELECT
    USING (true);

-- Grant EXECUTE on match_content to the anon role so the edge function can
-- switch from service_role key (Sec3) to anon key without breaking Vibe Search.
GRANT EXECUTE ON FUNCTION match_content(vector, float, int) TO anon;
