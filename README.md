# StreamIQ — Know What's Worth Watching on Indian OTT

> AI-powered streaming tracker for Netflix, Prime Video, JioHotstar, Apple TV+ and JioCinema in India.  
> Scores every title from real YouTube reviews, Reddit discussions and Rotten Tomatoes ratings.  
> **Free. No signup. Updated daily.**

🔗 **[Open StreamIQ →](https://movietracker-swapriya.streamlit.app/)**

---

## What is StreamIQ?

StreamIQ is a free streaming intelligence tool built for Indian viewers. Instead of endlessly scrolling through Netflix or Prime Video hoping something looks good, StreamIQ tells you what's actually worth watching — scored from real viewer sentiment, not algorithm-driven promotion.

Every 24 hours, StreamIQ automatically:
- Pulls trending and curated titles across all major Indian OTT platforms
- Analyses YouTube review videos for sentiment
- Reads Reddit discussions for community opinion
- Pulls Rotten Tomatoes and IMDb scores
- Verifies Hindi dub availability via JustWatch India
- Updates direct streaming links for every title

---

## Features

### 🎯 Watch Now — Trending with Scores
See what's trending this week on Indian streaming, scored by real viewer sentiment. Each title gets a composite score from YouTube, Reddit and critic ratings so you know at a glance whether it's worth your time.

### 🔍 Discover — 900+ Titles
Browse a curated library of 900+ titles across:
- **Classics** — Timeless films and shows available on Indian OTT
- **Hidden Gems** — Underrated titles with strong scores but low mainstream visibility
- **Hindi Originals** — Indian content in Hindi across all platforms
- **Genre Picks** — Action, Thriller, Horror, Comedy, Drama, Sci-Fi, Romance

### 🎙️ Hindi Dub Filter
Filter any view to only show titles with a **confirmed Hindi audio track** — verified live from JustWatch India, not guessed from the original language.

### ⚡ Polarising Content Flag
Some titles divide audiences sharply. StreamIQ flags polarising titles so you know before committing 2 hours to something controversial.

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
| JioCinema | Free + Premium |

---

## How the Scoring Works

Each title in Watch Now gets a composite score (0–100) built from:

| Source | Weight | What it measures |
|---|---|---|
| YouTube | ~40% | Sentiment from review videos |
| Reddit | ~30% | Community discussion tone |
| IMDb / RT | ~30% | Critic and audience ratings |

Scores above **60** are flagged as Worth Watching. Scores below 45 are generally not recommended.

---

## Tech Stack

- **Frontend** — Streamlit (Python)
- **Data** — TMDb API, YouTube Data API, Reddit RSS, JustWatch GraphQL API
- **Database** — Supabase (PostgreSQL)
- **Scoring** — Custom sentiment pipeline with Gemini AI
- **Automation** — GitHub Actions (daily cron)
- **Deployment** — Streamlit Community Cloud

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/swapedoc/streaming-tracker.git
cd streaming-tracker

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in your API keys in .env

# Run the data pipeline
python streaming_tracker_v3.py

# Launch the dashboard
streamlit run dashboard_v3.py
```

### Required API Keys (.env)
```
SUPABASE_URL=
SUPABASE_KEY=
TMDB_API_KEY=
YOUTUBE_API_KEY=
GEMINI_API_KEY=
OMDB_API_KEY=
```

---

## FAQ

**Is StreamIQ free?**  
Yes, completely free. No account, no subscription, no ads.

**How often is it updated?**  
Every 24 hours via an automated GitHub Actions cron job.

**Does it work on mobile?**  
Yes, the dashboard is responsive and works on mobile browsers.

**Can I filter by Hindi dub only?**  
Yes — there's a dedicated Hindi Dub toggle in both the Watch Now and Discover tabs.

**Which OTT platforms are supported?**  
Netflix, Prime Video, JioHotstar, Apple TV+ and JioCinema — all for India region.

**Is the source code available?**  
The dashboard and data pipeline code is in this repo.

---

## Keywords

streaming tracker india · what to watch netflix india · prime video india recommendations · jiohotstar best shows · hindi dub filter ott · best movies streaming india 2025 · ott tracker india · bollywood streaming · indian web series recommendations · what to watch tonight india

---

*Built with ❤️ for Indian viewers who are tired of scrolling.*
