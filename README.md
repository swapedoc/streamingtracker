# StreamIQ — Know What's Worth Watching on Indian OTT

> AI-powered streaming tracker for Netflix, Prime Video, JioHotstar and Apple TV+ in India.  
> Scores every title from real YouTube reviews, Reddit discussions and Rotten Tomatoes ratings.  
> **Free. No signup. Updated twice daily.**

🔗 **[Open StreamIQ →](https://streamiq-india.vercel.app/)**

---

## What is StreamIQ?

StreamIQ is a free streaming intelligence tool built for Indian viewers. Instead of endlessly scrolling through Netflix or Prime Video hoping something looks good, StreamIQ tells you what's actually worth watching — scored from real viewer sentiment, not algorithm-driven promotion.

Twice a day, StreamIQ automatically:
- Pulls trending and curated titles across all major Indian OTT platforms
- Analyses YouTube review videos for sentiment
- Reads Reddit discussions for community opinion
- Pulls Rotten Tomatoes and IMDb scores
- Verifies Hindi dub availability via JustWatch India
- Updates direct streaming links and expiry dates for every title

---

## Features

### 🎯 Watch Now — Trending with Scores
See what's trending this week on Indian streaming, scored by real viewer sentiment. Each title gets a composite score from YouTube, Reddit and critic ratings so you know at a glance whether it's worth your time.

Live stats at the top show total titles, average score, how many are Worth Watching, and how many are Polarising — all clickable to filter instantly.

### 🔍 Discover — 4,000+ Titles
Browse a curated library of 4,000+ titles across:
- **Classics** — Timeless films and shows with 8.0+ IMDb ratings available on Indian OTT
- **Hidden Gems** — Underrated titles with strong scores but low mainstream visibility
- **Hindi Originals** — Indian content in Hindi across all platforms
- **Genre Picks** — Action, Thriller, Horror, Comedy, Drama, Sci-Fi, Romance

### 🔮 Vibe Search — Semantic AI Search
Describe what you're in the mood for in plain English and StreamIQ finds the closest matches across all 4,000+ titles using AI embeddings and cosine similarity. Examples: *"something scary but not gory"*, *"feel-good Hindi comedy"*, *"mind-bending sci-fi with a twist ending"*. Powered by Gemini embeddings via a Supabase Edge Function.

### 🔔 Watchlist + Telegram Alerts
Track any title from the Discover section. When it becomes available on a streaming platform, StreamIQ sends you a **Telegram notification** directly to your chat. Watchlist is browser-based (no account needed) — just add your Telegram Chat ID once to enable alerts.

### ⏳ Leaving Soon
Titles expiring from a platform are flagged with a countdown badge — pulsing orange when under 14 days, red when under 7. The **Last Chance** filter in Discover shows only expiring titles sorted by soonest leaving first.

### ⚡ Vibe Score — Genre-Specific Metric
Each Watch Now title gets a genre-specific intensity score (1–10) extracted by AI from review text:
- Horror → **Scare Factor**
- Thriller → **Tension Meter**
- Action → **Adrenaline**
- Comedy → **Laugh Meter**
- Sci-Fi → **Mind-Bend**
- Romance → **Heart Score**
- Drama → **Emotional Hit**

### ⚠️ Polarising Content Flag
Some titles divide audiences sharply. StreamIQ flags polarising titles so you know before committing 2 hours to something controversial.

### 🎙️ Hindi Dub Filter
Filter any view to only show titles with a **confirmed Hindi audio track** — verified live from JustWatch India, not guessed from the original language. Available in both Watch Now and Discover.

### ⏱️ Binge Time
Every title shows estimated watch time — runtime for movies, season/episode count for series.

### 🎬 Inline Trailers
Watch the official trailer directly in the card or detail panel without leaving the page. Sourced from TMDb (free, no YouTube quota). Falls back to YouTube for titles TMDb doesn't have.

### 🔗 Direct Streaming Links
One click opens the title directly on the right platform. No more hunting across apps to find where something is streaming.

### 📺 LG TV Support
Copy any title to clipboard with one tap and search on your LG TV remote. Designed for the actual couch experience.

---

## Platforms Covered

| Platform | Type |
|---|---|
| Netflix India | Subscription |
| Prime Video India | Subscription |
| JioHotstar | Subscription |
| Apple TV+ | Subscription |

---

## How the Scoring Works

Each title in Watch Now gets a composite score (0–100) built from:

| Source | Weight | What it measures |
|---|---|---|
| YouTube | ~40% | Sentiment from review videos |
| Reddit | ~30% | Community discussion tone |
| IMDb / RT | ~30% | Critic and audience ratings |

Weights shift dynamically based on content age — newer titles lean more on YouTube/Reddit, older titles lean more on IMDb. Scores above **60** are Worth Watching. Scores below 45 are generally not recommended.

---

## Tech Stack

- **Frontend** — Vanilla JS / HTML (Vercel)
- **Data** — TMDb API, YouTube Data API, Reddit, JustWatch GraphQL API, OMDB API
- **Database** — Supabase (PostgreSQL + pgvector for embeddings)
- **Scoring** — Custom sentiment pipeline: Groq (primary) → Gemini (fallback) → VADER
- **Embeddings** — Gemini `gemini-embedding-001` (3072-dim) via Supabase Edge Function
- **Automation** — GitHub Actions (cron every 12 hours)
- **Deployment** — Vercel

---

## Running the Data Pipeline Locally

```bash
# Clone the repo
git clone https://github.com/swapedoc/streamingtracker.git
cd streamingtracker

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in your API keys in .env

# Run the main pipeline (Watch Now + Discover)
python streaming_tracker.py

# Run discover only (skips Watch Now, no YouTube quota used)
python streaming_tracker_v3.py --discover-only

# Enrich trailers + runtime for Discover titles (run weekly)
python enrich_trailers_cron.py

# Backfill Gemini embeddings for Vibe Search
python generate_embeddings.py

# Send watchlist alerts manually
python watchlist_alerts.py
```

### Required API Keys (.env)
```
SUPABASE_URL=
SUPABASE_KEY=
TMDB_API_KEY=
YOUTUBE_API_KEY=
GEMINI_API_KEY=
OMDB_API_KEY=
GROQ_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
```

---

## Cron Schedule

| Schedule | What runs |
|---|---|
| Every 12 hours | Watch Now + Discover tracker |
| Daily 9am IST | Watchlist alerts (Telegram) |
| Every Sunday 2am | Full catalog refresh + trailer enrich + Hindi dubs |

---

## FAQ

**Is StreamIQ free?**
Yes, completely free. No account, no subscription, no ads.

**How often is it updated?**
Twice a day via an automated GitHub Actions cron job.

**Does it work on mobile?**
Yes, the site is responsive and works on mobile browsers.

**Can I filter by Hindi dub only?**
Yes — there's a dedicated Hindi Dub toggle in both the Watch Now and Discover tabs.

**How does Vibe Search work?**
You type a natural language description of what you want to watch. The query is converted to a vector embedding using Gemini and compared against pre-computed embeddings for all 4,000+ Discover titles using cosine similarity. The closest matches are returned ranked by relevance.

**How do Watchlist alerts work?**
Track a title from any Discover card. Enter your Telegram Chat ID once. Each day at 9am IST, StreamIQ checks if any tracked titles are now streaming — if so, it sends you a Telegram message with the platform and direct link.

**Which OTT platforms are supported?**
Netflix, Prime Video, JioHotstar and Apple TV+ — all for India region.

**Is the source code available?**
The frontend and data pipeline code is in this repo.

---

## Keywords

streaming tracker india · what to watch netflix india · prime video india recommendations · jiohotstar best shows · hindi dub filter ott · best movies streaming india 2026 · ott tracker india · bollywood streaming · indian web series recommendations · what to watch tonight india · vibe search movies · ai movie recommendations india

---

*Built with ❤️ for Indian viewers who are tired of scrolling.*
