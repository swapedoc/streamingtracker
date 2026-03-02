"""
constants.py — Shared genre maps and platform config used across all scripts.
Import from here instead of duplicating in each file.
"""

# ── Platform config ───────────────────────────────────────────────────────────

# TMDb provider IDs for India (used by fetch_full_catalog, streaming_tracker)
PLATFORMS = {
    'Netflix':     8,
    'Prime Video': 119,
    'Apple TV+':   350,
    'Jiohotstar':  2336,
}

# JustWatch shortName → (display name, fallback URL)
# Used by watchlist_alerts and fetch_hindi_dubs
#
# Disney+ and JioHotstar merged in India — all Disney+/Hotstar shortNames
# resolve to 'Jiohotstar'. JustWatch may return any of these for the same title.
JUSTWATCH_PLATFORM_MAP = {
    'nfx':  ('Netflix',      'https://www.netflix.com'),
    'prv':  ('Prime Video',  'https://www.primevideo.com'),
    'jhs':  ('Jiohotstar',   'https://www.jiohotstar.com'),  # current live shortName
    'hst':  ('Jiohotstar',   'https://www.jiohotstar.com'),  # legacy Hotstar
    'dnp':  ('Jiohotstar',   'https://www.jiohotstar.com'),  # legacy Disney+ (merged)
    'hot':  ('Jiohotstar',   'https://www.jiohotstar.com'),  # legacy Hotstar variant
    'jio':  ('Jiohotstar',   'https://www.jiohotstar.com'),  # legacy JioCinema (merged)
    'atp':  ('Apple TV+',    'https://tv.apple.com'),
    'mxs':  ('Max',          'https://www.max.com'),
}

# Priority order for JustWatch offer selection (most preferred first)
# All Jiohotstar shortNames are grouped — whichever appears first in offers wins.
JUSTWATCH_PRIORITY = ['nfx', 'prv', 'jhs', 'hst', 'dnp', 'hot', 'jio', 'atp', 'mxs']

# ── Movie genre IDs → our label ───────────────────────────────────────────────

MOVIE_GENRE_MAP = {
    28:    'Action',
    27:    'Horror',
    35:    'Comedy',
    18:    'Drama',
    53:    'Thriller',
    878:   'Sci-Fi',
    10749: 'Romance',
    12:    'Action',    # Adventure
    14:    'Drama',     # Fantasy
    36:    'Drama',     # History
    10752: 'Action',    # War
    37:    'Drama',     # Western
    99:    'Drama',     # Documentary
    10402: 'Comedy',    # Music
    9648:  'Thriller',  # Mystery
    10770: 'Drama',     # TV Movie
    16:    'Comedy',    # Animation
    10751: 'Drama',     # Family
    10769: 'Drama',     # Foreign
}

# ── TV genre IDs → our movie-style label (for genre field) ───────────────────

TV_GENRE_MAP = {
    10759: 'Action',    # Action & Adventure
    10765: 'Sci-Fi',    # Sci-Fi & Fantasy
    10766: 'Drama',     # Soap
    10768: 'Action',    # War & Politics
    10762: 'Comedy',    # Kids
    10763: 'Drama',     # News
    10764: 'Drama',     # Reality
    80:    'Thriller',  # Crime
    9648:  'Thriller',  # Mystery
    35:    'Comedy',
    18:    'Drama',
    16:    'Comedy',    # Animation
    99:    'Drama',     # Documentary
    37:    'Drama',     # Western
    10751: 'Drama',     # Family
}

# ── TV genre IDs → human-readable label (for tv_genre field) ─────────────────

TV_GENRE_LABELS = {
    10759: 'Action & Adventure',
    10765: 'Sci-Fi & Fantasy',
    35:    'Comedy',
    18:    'Drama',
    80:    'Crime',
    9648:  'Mystery',
    10751: 'Family',
    16:    'Animation',
    99:    'Documentary',
    10766: 'Soap',
    10768: 'War & Politics',
    37:    'Western',
}

# ── Priority orders (most specific / scarce first) ────────────────────────────

GENRE_PRIORITY = ['Horror', 'Sci-Fi', 'Thriller', 'Romance', 'Comedy', 'Action', 'Drama']

TV_GENRE_PRIORITY = [
    'Action & Adventure', 'Sci-Fi & Fantasy', 'Crime', 'Mystery',
    'Comedy', 'Drama', 'Animation', 'Family', 'Documentary',
    'Soap', 'War & Politics', 'Western',
]

# ── Indian languages ──────────────────────────────────────────────────────────

INDIAN_LANGUAGES = {'hi', 'ta', 'te', 'ml', 'kn', 'bn', 'mr', 'pa'}
