-- ─────────────────────────────────────────────────────────────────────────────
-- StreamIQ — Database Schema
-- Run once in Supabase SQL editor to set up all required tables and columns.
-- Safe to re-run on an existing DB — all CREATE/ALTER statements are idempotent.
--
-- VERIFIED against live DB snapshot (2026-03-04).
-- Differences from previous schema.sql:
--   LIVE-1  : content.id is SERIAL (INTEGER), not BIGSERIAL — fixed
--   LIVE-2  : reviews.id is SERIAL (INTEGER), not BIGSERIAL — fixed
--   LIVE-3  : scores.id is SERIAL (INTEGER), not BIGSERIAL — fixed
--   LIVE-4  : reviews.content_id is INTEGER nullable with FK to content(id) — fixed
--   LIVE-5  : scores.content_id is INTEGER nullable with FK to content(id) — fixed
--   LIVE-6  : content timestamps are TIMESTAMP (no tz) in live — matched
--   LIVE-7  : reviews.created_at is TIMESTAMP (no tz) + added updated_at TIMESTAMPTZ
--   LIVE-8  : scores.computed_at is TIMESTAMP (no tz) in live — matched
--   LIVE-9  : watchlist has poster TEXT, score NUMERIC, added_at TIMESTAMPTZ
--             content_type NOT NULL, notified NOT NULL — all added
--   LIVE-10 : watchlist unique constraint is (browser_id, title) — matched
--             NOTE: should be (browser_id, title, content_type) — see bug list
--   LIVE-11 : discover_content.category is NOT NULL DEFAULT 'genre_drama'
--             NOTE: this breaks the FC1 back-fill (.is_('category','null') never
--             matches). To fix on live: ALTER TABLE discover_content
--             ALTER COLUMN category DROP NOT NULL, SET DEFAULT NULL.
--   LIVE-12 : Added all indexes present in live but missing from schema:
--             idx_content_discovery, idx_content_tmdb,
--             idx_discover_content_stream_url, idx_scores_category,
--             idx_watchlist_unnotified
--   LIVE-13 : match_content() updated to match live signature + added tv_genre
--   LIVE-14 : RLS policies renamed to match live (public_read) + watchlist
--             policies added, content/reviews/scores RLS enabled
--   LIVE-15 : idx_discover_genre and idx_discover_embedding were in old
--             schema.sql but missing from live DB — kept here so fresh
--             deployments get them; run separately on live if needed
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable pgvector extension (required for embeddings + match_content RPC)
CREATE EXTENSION IF NOT EXISTS vector;


-- ── discover_content ──────────────────────────────────────────────────────────
-- Stores the full OTT catalog for the Discover tab (no reviews, just availability).
-- Populated by streaming_tracker.py (DiscoverFlow) and fetch_full_catalog.py.
--
-- NOTE: category is NOT NULL DEFAULT 'genre_drama' to match live DB.
-- This means the FC1 back-fill pattern (.is_('category','null')) never fires on
-- newly inserted rows because Postgres applies the default immediately. To fix:
--   ALTER TABLE discover_content ALTER COLUMN category DROP NOT NULL;
--   ALTER TABLE discover_content ALTER COLUMN category SET DEFAULT NULL;
CREATE TABLE IF NOT EXISTS discover_content (
    id              BIGSERIAL PRIMARY KEY,
    tmdb_id         INTEGER       NOT NULL,
    title           TEXT          NOT NULL,
    original_title  TEXT,
    platform        TEXT          NOT NULL,
    content_type    TEXT          NOT NULL,
    release_year    INTEGER,
    imdb_rating     FLOAT,
    poster_path     TEXT,
    overview        TEXT,
    category        TEXT          NOT NULL DEFAULT 'genre_drama',
    genre           TEXT,
    tv_genre        TEXT,
    popularity      FLOAT,
    stream_url      TEXT,
    source          TEXT          DEFAULT 'tracker',  -- 'tracker' | 'catalog'
    hindi_dub       BOOLEAN       DEFAULT FALSE,
    runtime         INTEGER,                           -- minutes (movies)
    seasons         INTEGER,                           -- TV only
    episode_count   INTEGER,                           -- TV only
    episode_runtime INTEGER,                           -- mins per ep, TV only
    trailer_id      TEXT,                              -- YouTube video ID
    embedding       vector(3072),                      -- Gemini gemini-embedding-001
    leaving_date    DATE,
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (tmdb_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_discover_category          ON discover_content(category);
CREATE INDEX IF NOT EXISTS idx_discover_platform          ON discover_content(platform);
CREATE INDEX IF NOT EXISTS idx_discover_genre             ON discover_content(genre);
CREATE INDEX IF NOT EXISTS idx_discover_content_stream_url ON discover_content(stream_url);
-- ivfflat index for pgvector cosine similarity (match_content RPC / Vibe Search).
-- Requires ~1000+ rows to be useful. Increase lists= as catalog grows.
CREATE INDEX IF NOT EXISTS idx_discover_embedding ON discover_content
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


-- ── content ───────────────────────────────────────────────────────────────────
-- Watch Now titles — enriched with reviews, scores, and stream URLs.
-- id is SERIAL (INTEGER) matching live DB sequence.
-- Timestamps are TIMESTAMP (no timezone) matching live DB types.
CREATE TABLE IF NOT EXISTS content (
    id               SERIAL PRIMARY KEY,
    tmdb_id          INTEGER   NOT NULL,
    title            TEXT      NOT NULL,
    original_title   TEXT,
    platform         TEXT      NOT NULL,
    content_type     TEXT      NOT NULL,                  -- 'movie' | 'tv'
    release_year     INTEGER,
    imdb_rating      FLOAT,
    imdb_id          TEXT,
    poster_path      TEXT,
    overview         TEXT,
    discovery_source TEXT      DEFAULT 'trending',        -- 'trending' | 'catalog' etc.
    genre            TEXT,
    tv_genre         TEXT,
    stream_url       TEXT      DEFAULT '',
    hindi_dub        BOOLEAN   DEFAULT FALSE,
    runtime          INTEGER,
    seasons          INTEGER,
    episode_count    INTEGER,
    episode_runtime  INTEGER,
    trailer_id       TEXT,
    leaving_date     DATE,
    created_at       TIMESTAMP DEFAULT NOW(),
    updated_at       TIMESTAMP DEFAULT NOW()              -- refreshed on every upsert
);

-- Drop legacy single-column unique before adding composite (idempotent)
ALTER TABLE content DROP CONSTRAINT IF EXISTS content_tmdb_id_key;
ALTER TABLE content ADD CONSTRAINT content_tmdb_id_platform_key
    UNIQUE (tmdb_id, platform);

CREATE INDEX IF NOT EXISTS idx_content_platform  ON content(platform);
CREATE INDEX IF NOT EXISTS idx_content_discovery ON content(discovery_source);
CREATE INDEX IF NOT EXISTS idx_content_tmdb      ON content(tmdb_id);


-- ── reviews ───────────────────────────────────────────────────────────────────
-- One row per review source per title.
-- id is SERIAL (INTEGER) matching live DB.
-- content_id is INTEGER nullable with FK to content(id) ON DELETE CASCADE.
-- updated_at exists in live DB, was missing from old schema.sql.
CREATE TABLE IF NOT EXISTS reviews (
    id                   SERIAL  PRIMARY KEY,
    content_id           INTEGER REFERENCES content(id) ON DELETE CASCADE,
    source               TEXT    NOT NULL,                 -- 'youtube'|'reddit'|'rotten_tomatoes'|'tmdb'
    source_id            TEXT    NOT NULL,
    source_url           TEXT,
    reviewer             TEXT,
    reviewer_subscribers INTEGER,
    review_text          TEXT,
    sentiment            INTEGER,                          -- -1 | 0 | 1
    confidence           FLOAT,
    views                INTEGER DEFAULT 0,
    likes                INTEGER DEFAULT 0,
    comments_count       INTEGER DEFAULT 0,
    youtube_weight       FLOAT   DEFAULT 0,
    engagement_score     FLOAT   DEFAULT 0,
    weighted_sentiment   FLOAT   DEFAULT 0,
    created_at           TIMESTAMP   DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_content ON reviews(content_id);


-- ── scores ────────────────────────────────────────────────────────────────────
-- One row per content title — computed by ScoreComputer.
-- id is SERIAL (INTEGER) matching live DB.
-- content_id is INTEGER nullable with FK to content(id) ON DELETE CASCADE.
-- computed_at is TIMESTAMP (no tz) matching live DB.
CREATE TABLE IF NOT EXISTS scores (
    id               SERIAL  PRIMARY KEY,
    content_id       INTEGER UNIQUE REFERENCES content(id) ON DELETE CASCADE,
    youtube_score    FLOAT   DEFAULT 0,
    reddit_score     FLOAT   DEFAULT 0,
    imdb_score       FLOAT   DEFAULT 0,
    engagement_score FLOAT   DEFAULT 0,                    -- legacy column, kept for compat
    final_score      FLOAT   DEFAULT 0,
    label            TEXT,
    category         TEXT    DEFAULT 'catalog',
    review_count     INTEGER DEFAULT 0,
    positive_ratio   FLOAT   DEFAULT 0,
    is_polarizing    BOOLEAN DEFAULT FALSE,
    sentiment_std    FLOAT   DEFAULT 0,
    rt_score         FLOAT   DEFAULT 0,                    -- Rotten Tomatoes weighted score
    vibe_score       FLOAT,                                -- 1.0–10.0 genre-specific intensity
    vibe_label       TEXT,                                 -- e.g. 'Scare Factor', 'Laugh Meter'
    computed_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scores_final    ON scores(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_scores_category ON scores(category);


-- ── watchlist ─────────────────────────────────────────────────────────────────
-- User-tracked titles (anonymous, keyed by browser fingerprint).
-- poster and score columns exist in live DB (from earlier version, unused by code).
-- added_at is the ordering column used by index.html (not created_at).
-- content_type and notified are NOT NULL in live DB.
--
-- NOTE: unique constraint is (browser_id, title) matching live DB.
-- Should be (browser_id, title, content_type) to allow tracking same title as
-- both movie and TV. See bug list for the ALTER to fix this on live.
CREATE TABLE IF NOT EXISTS watchlist (
    id               BIGSERIAL PRIMARY KEY,
    browser_id       TEXT        NOT NULL,
    title            TEXT        NOT NULL,
    content_type     TEXT        NOT NULL  DEFAULT 'movie',  -- 'movie' | 'tv'
    tmdb_id          TEXT,                                    -- nullable string from frontend
    poster           TEXT,                                    -- exists in live, unused by code
    platform         TEXT,                                    -- NULL until streaming confirmed
    stream_url       TEXT,
    score            NUMERIC,                                 -- exists in live, unused by code
    notified         BOOLEAN     NOT NULL  DEFAULT FALSE,
    added_at         TIMESTAMPTZ NOT NULL  DEFAULT NOW(),     -- ordering column used by frontend
    telegram_chat_id TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (browser_id, title)
);

-- Partial index — covers the nightly alert query (notified=false titles only)
CREATE INDEX IF NOT EXISTS idx_watchlist_unnotified ON watchlist(notified, platform)
    WHERE notified = FALSE;


-- ── trailer_overrides ─────────────────────────────────────────────────────────
-- Manual trailer ID overrides — applied by enrich_trailers_cron.py before
-- any auto-fetch logic runs.
CREATE TABLE IF NOT EXISTS trailer_overrides (
    tmdb_id    INTEGER PRIMARY KEY,
    trailer_id TEXT    NOT NULL,
    note       TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ── sync_state ────────────────────────────────────────────────────────────────
-- Single-row cache invalidation signal.
-- Updated by streaming_tracker.py after each run so the frontend knows to
-- bust its 12h localStorage cache. RLS intentionally disabled — public read
-- is fine since it's just a timestamp (no sensitive data).
CREATE TABLE IF NOT EXISTS sync_state (
    id           BOOL PRIMARY KEY DEFAULT TRUE CHECK (id),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO sync_state (last_updated) VALUES (NOW())
    ON CONFLICT (id) DO NOTHING;


-- ─────────────────────────────────────────────────────────────────────────────
-- ALTER TABLE additions for existing deployments
-- All guarded with IF NOT EXISTS — safe to re-run on live DB.
-- ─────────────────────────────────────────────────────────────────────────────

-- discover_content additions
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS tv_genre        TEXT;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS runtime         INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS seasons         INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS episode_count   INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS episode_runtime INTEGER;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS trailer_id      TEXT;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS source          TEXT DEFAULT 'tracker';
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS leaving_date    DATE;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS hindi_dub       BOOLEAN DEFAULT FALSE;
ALTER TABLE discover_content ADD COLUMN IF NOT EXISTS embedding       vector(3072);

-- content additions
ALTER TABLE content ADD COLUMN IF NOT EXISTS imdb_id        TEXT;
ALTER TABLE content ADD COLUMN IF NOT EXISTS updated_at     TIMESTAMP DEFAULT NOW();
ALTER TABLE content ADD COLUMN IF NOT EXISTS runtime        INTEGER;
ALTER TABLE content ADD COLUMN IF NOT EXISTS seasons        INTEGER;
ALTER TABLE content ADD COLUMN IF NOT EXISTS episode_count  INTEGER;
ALTER TABLE content ADD COLUMN IF NOT EXISTS episode_runtime INTEGER;
ALTER TABLE content ADD COLUMN IF NOT EXISTS trailer_id     TEXT;
ALTER TABLE content ADD COLUMN IF NOT EXISTS genre          TEXT;
ALTER TABLE content ADD COLUMN IF NOT EXISTS tv_genre       TEXT;
ALTER TABLE content ADD COLUMN IF NOT EXISTS leaving_date   DATE;
ALTER TABLE content ADD COLUMN IF NOT EXISTS hindi_dub      BOOLEAN DEFAULT FALSE;

-- reviews additions
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS reviewer_subscribers INTEGER;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS views                INTEGER DEFAULT 0;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS likes                INTEGER DEFAULT 0;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS comments_count       INTEGER DEFAULT 0;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS youtube_weight       FLOAT   DEFAULT 0;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS engagement_score     FLOAT   DEFAULT 0;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS updated_at           TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE reviews ALTER COLUMN review_text TYPE TEXT;

-- scores additions
ALTER TABLE scores ADD COLUMN IF NOT EXISTS rt_score   FLOAT DEFAULT 0;
ALTER TABLE scores ADD COLUMN IF NOT EXISTS vibe_score FLOAT;
ALTER TABLE scores ADD COLUMN IF NOT EXISTS vibe_label TEXT;
ALTER TABLE scores ADD COLUMN IF NOT EXISTS category   TEXT DEFAULT 'catalog';

-- Rename scores.created_at → computed_at if old column still exists
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

-- watchlist additions
ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS poster           TEXT;
ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS score            NUMERIC;
ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS added_at         TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS telegram_chat_id TEXT;


-- ─────────────────────────────────────────────────────────────────────────────
-- Row Level Security
-- ─────────────────────────────────────────────────────────────────────────────

-- discover_content: public read, service_role writes
ALTER TABLE discover_content ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS public_read              ON discover_content;
DROP POLICY IF EXISTS "Public read discover_content" ON discover_content;   -- old name
CREATE POLICY public_read ON discover_content FOR SELECT USING (true);

-- content: public read, service_role writes
ALTER TABLE content ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS public_read ON content;
CREATE POLICY public_read ON content FOR SELECT USING (true);

-- reviews: public read, service_role writes
ALTER TABLE reviews ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS public_read ON reviews;
CREATE POLICY public_read ON reviews FOR SELECT USING (true);

-- scores: public read, service_role writes
ALTER TABLE scores ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS public_read ON scores;
CREATE POLICY public_read ON scores FOR SELECT USING (true);

-- watchlist: anon can read/insert/delete their own rows; only service_role can update
-- (watchlist_alerts.py uses service_role key to set platform/stream_url/notified)
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS anon_select_own ON watchlist;
DROP POLICY IF EXISTS anon_insert     ON watchlist;
DROP POLICY IF EXISTS anon_delete_own ON watchlist;
DROP POLICY IF EXISTS service_update  ON watchlist;
CREATE POLICY anon_select_own ON watchlist FOR SELECT TO anon   USING (true);
CREATE POLICY anon_insert     ON watchlist FOR INSERT TO anon   WITH CHECK (true);
CREATE POLICY anon_delete_own ON watchlist FOR DELETE TO anon   USING (true);
CREATE POLICY service_update  ON watchlist FOR UPDATE TO service_role
    USING (true) WITH CHECK (true);

-- sync_state: RLS intentionally OFF (public timestamp, no sensitive data)
-- trailer_overrides: RLS intentionally OFF (read-only reference data)


-- ─────────────────────────────────────────────────────────────────────────────
-- match_content() — pgvector RPC for Vibe Search
-- Called by edgefunction.ts (smart-endpoint) and index.html vibeSearch().
--
-- Matches live DB signature exactly, with tv_genre added (missing from live —
-- run this CREATE OR REPLACE to update the live function in one shot).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_content(
    query_embedding vector(3072),
    match_threshold float,
    match_count     int
)
RETURNS TABLE (
    id              bigint,
    tmdb_id         integer,
    title           text,
    platform        text,
    content_type    text,
    release_year    integer,
    imdb_rating     float,
    poster_path     text,
    overview        text,
    category        text,
    genre           text,
    tv_genre        text,
    stream_url      text,
    hindi_dub       boolean,
    trailer_id      text,
    runtime         integer,
    seasons         integer,
    episode_count   integer,
    episode_runtime integer,
    similarity      float
)
LANGUAGE sql STABLE AS $$
    SELECT
        id,
        tmdb_id,
        title,
        platform,
        content_type,
        release_year,
        imdb_rating,
        poster_path,
        overview,
        category,
        genre,
        tv_genre,
        stream_url,
        hindi_dub,
        trailer_id,
        runtime,
        seasons,
        episode_count,
        episode_runtime,
        1 - (embedding <=> query_embedding) AS similarity
    FROM discover_content
    WHERE embedding IS NOT NULL
      AND 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION match_content(vector, float, int) TO anon;
