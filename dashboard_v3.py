# dashboard_lg_tv_fixed.py — StreamIQ with LG TV Integration + All Features
# streamlit run dashboard_lg_tv_fixed.py

import streamlit as st
import os, json, math
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv
import streamlit.components.v1 as components

load_dotenv()

st.set_page_config(page_title="StreamIQ", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# Custom loading screen CSS - Sci-Fi Cyberpunk version
loading_screen = """
<script>
(function(){
  var fe = window.frameElement;
  if(fe){
    fe.style.cssText = 'position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;z-index:999999!important;border:none!important;background:transparent';
    setTimeout(function(){ fe.style.opacity='0'; fe.style.transition='opacity 0.8s ease'; }, 4000);
    setTimeout(function(){ fe.style.display='none'; }, 4800);
  }
})();
</script>

<style>
/* Hide Streamlit's default loading */
div[data-testid="stStatusWidget"] {
    display: none !important;
}

#loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: #000000;
    z-index: 99999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeOut 0.8s ease 4s forwards;
    pointer-events: none;
    overflow: hidden;
}

@keyframes fadeOut {
    to { opacity: 0; visibility: hidden; }
}

/* Matrix-style digital rain background */
#matrix-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0.15;
}

/* Radial gradient overlay */
.radial-overlay {
    position: absolute;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at center, transparent 0%, rgba(0, 0, 0, 0.8) 70%);
}

/* Main container */
.load-container {
    position: relative;
    z-index: 10;
    text-align: center;
}

/* Holographic display */
.holo-display {
    position: relative;
    padding: 50px 70px;
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid;
    border-image: linear-gradient(135deg, #E8C547, #00ffff, #E8C547) 1;
    box-shadow: 0 0 30px rgba(232, 197, 71, 0.3), inset 0 0 30px rgba(0, 255, 255, 0.1);
    clip-path: polygon(0% 5%, 5% 0%, 95% 0%, 100% 5%, 100% 95%, 95% 100%, 5% 100%, 0% 95%);
    animation: holoPulse 3s ease-in-out infinite;
}

@keyframes holoPulse {
    0%, 100% {
        box-shadow: 0 0 30px rgba(232, 197, 71, 0.3), inset 0 0 30px rgba(0, 255, 255, 0.1);
    }
    50% {
        box-shadow: 0 0 50px rgba(232, 197, 71, 0.5), inset 0 0 50px rgba(0, 255, 255, 0.2);
    }
}

/* Corner indicators */
.corner-indicator {
    position: absolute;
    width: 30px;
    height: 30px;
    border: 2px solid #E8C547;
    opacity: 0.6;
}

.corner-tl {
    top: -15px;
    left: -15px;
    border-right: none;
    border-bottom: none;
    animation: cornerPulse 2s ease-in-out infinite;
}

.corner-tr {
    top: -15px;
    right: -15px;
    border-left: none;
    border-bottom: none;
    animation: cornerPulse 2s ease-in-out infinite 0.5s;
}

.corner-bl {
    bottom: -15px;
    left: -15px;
    border-right: none;
    border-top: none;
    animation: cornerPulse 2s ease-in-out infinite 1s;
}

.corner-br {
    bottom: -15px;
    right: -15px;
    border-left: none;
    border-top: none;
    animation: cornerPulse 2s ease-in-out infinite 1.5s;
}

@keyframes cornerPulse {
    0%, 100% {
        opacity: 0.4;
        transform: scale(1);
    }
    50% {
        opacity: 1;
        transform: scale(1.2);
    }
}

/* Logo */
.logo-container {
    margin-bottom: 40px;
}

.logo-main {
    font-size: 54px;
    font-weight: 700;
    letter-spacing: 12px;
    position: relative;
    display: inline-block;
    font-family: 'Courier New', monospace;
}

.stream-text {
    color: #ffffff;
    text-shadow: 0 0 10px rgba(255, 255, 255, 1), 0 0 20px rgba(0, 255, 255, 0.8), 0 0 30px rgba(0, 255, 255, 0.6);
    animation: cyberGlow 2s ease-in-out infinite;
}

.iq-text {
    color: #E8C547;
    text-shadow: 0 0 10px rgba(232, 197, 71, 1), 0 0 20px rgba(232, 197, 71, 0.8), 0 0 30px rgba(255, 165, 0, 0.6);
    animation: goldGlow 2s ease-in-out infinite;
}

@keyframes cyberGlow {
    0%, 100% {
        text-shadow: 0 0 10px rgba(255, 255, 255, 1), 0 0 20px rgba(0, 255, 255, 0.8);
    }
    50% {
        text-shadow: 0 0 20px rgba(255, 255, 255, 1), 0 0 40px rgba(0, 255, 255, 1), 0 0 60px rgba(0, 255, 255, 0.8);
    }
}

@keyframes goldGlow {
    0%, 100% {
        text-shadow: 0 0 10px rgba(232, 197, 71, 1), 0 0 20px rgba(232, 197, 71, 0.8);
    }
    50% {
        text-shadow: 0 0 20px rgba(232, 197, 71, 1), 0 0 40px rgba(255, 165, 0, 1), 0 0 60px rgba(255, 165, 0, 0.8);
    }
}

.subtitle {
    font-size: 9px;
    color: #00ffff;
    letter-spacing: 4px;
    margin-top: 10px;
    text-transform: uppercase;
    opacity: 0.7;
    text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
    font-family: 'Courier New', monospace;
}

/* About section */
.about-section {
    margin: 30px 0;
    padding: 18px 25px;
    background: rgba(232, 197, 71, 0.05);
    border: 1px solid rgba(232, 197, 71, 0.2);
    border-left: 3px solid #E8C547;
    text-align: left;
    max-width: 480px;
    margin-left: auto;
    margin-right: auto;
    animation: fadeInUp 1s ease-out 0.5s both;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.about-title {
    font-size: 10px;
    color: #E8C547;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-weight: 600;
    font-family: 'Courier New', monospace;
}

.about-text {
    font-size: 12px;
    color: #aaa;
    line-height: 1.6;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.about-sources {
    margin-top: 10px;
    font-size: 10px;
    color: #00ffff;
    letter-spacing: 1px;
    font-family: 'Courier New', monospace;
}

.about-sources span {
    display: inline-block;
    padding: 3px 8px;
    margin: 3px;
    background: rgba(0, 255, 255, 0.1);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 2px;
}

/* Central orb */
.central-orb {
    width: 90px;
    height: 90px;
    margin: 0 auto 30px;
    position: relative;
}

.orb-core {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #E8C547 0%, #FFA500 30%, transparent 70%);
    box-shadow: 0 0 30px rgba(232, 197, 71, 0.8), 0 0 60px rgba(232, 197, 71, 0.5), inset 0 0 30px rgba(232, 197, 71, 0.3);
    animation: orbPulse 2s ease-in-out infinite;
}

@keyframes orbPulse {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 30px rgba(232, 197, 71, 0.8), 0 0 60px rgba(232, 197, 71, 0.5);
    }
    50% {
        transform: scale(1.1);
        box-shadow: 0 0 50px rgba(232, 197, 71, 1), 0 0 100px rgba(232, 197, 71, 0.7);
    }
}

.orb-ring {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 120%;
    height: 120%;
    border: 2px solid rgba(0, 255, 255, 0.5);
    border-radius: 50%;
    animation: ringRotate 4s linear infinite;
}

.orb-ring:nth-child(2) {
    width: 140%;
    height: 140%;
    animation-duration: 3s;
    animation-direction: reverse;
    border-color: rgba(232, 197, 71, 0.3);
}

@keyframes ringRotate {
    0% {
        transform: translate(-50%, -50%) rotate(0deg);
    }
    100% {
        transform: translate(-50%, -50%) rotate(360deg);
    }
}

/* Data stream visualization */
.data-stream {
    margin: 25px 0;
    height: 80px;
    position: relative;
    overflow: hidden;
}

.stream-line {
    position: absolute;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #E8C547 20%, #00ffff 50%, #E8C547 80%, transparent 100%);
    animation: streamFlow 3s linear infinite;
    box-shadow: 0 0 10px rgba(232, 197, 71, 0.8);
}

.stream-line:nth-child(1) {
    top: 20%;
    width: 80%;
    animation-delay: 0s;
}

.stream-line:nth-child(2) {
    top: 40%;
    width: 60%;
    animation-delay: 0.5s;
    opacity: 0.7;
}

.stream-line:nth-child(3) {
    top: 60%;
    width: 70%;
    animation-delay: 1s;
    opacity: 0.5;
}

.stream-line:nth-child(4) {
    top: 80%;
    width: 50%;
    animation-delay: 1.5s;
    opacity: 0.6;
}

@keyframes streamFlow {
    0% {
        left: -100%;
    }
    100% {
        left: 100%;
    }
}

/* Terminal output */
.terminal {
    margin-top: 30px;
    text-align: left;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    color: #00ff00;
    max-width: 450px;
}

.terminal-line {
    margin: 5px 0;
    opacity: 0;
    animation: terminalType 0.5s ease-out forwards;
}

@keyframes terminalType {
    from {
        opacity: 0;
        transform: translateX(-5px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.terminal-line:nth-child(1) { animation-delay: 0.5s; }
.terminal-line:nth-child(2) { animation-delay: 1s; }
.terminal-line:nth-child(3) { animation-delay: 1.5s; }
.terminal-line:nth-child(4) { animation-delay: 2s; }

.cursor {
    display: inline-block;
    width: 8px;
    height: 14px;
    background: #00ff00;
    animation: cursorBlink 1s step-end infinite;
    margin-left: 2px;
}

@keyframes cursorBlink {
    0%, 50% {
        opacity: 1;
    }
    51%, 100% {
        opacity: 0;
    }
}

.success {
    color: #00ff00;
}

.warning {
    color: #E8C547;
}

.info {
    color: #00ffff;
}

/* System info overlay */
.system-info {
    position: absolute;
    top: 20px;
    left: 20px;
    font-family: monospace;
    font-size: 9px;
    color: #00ff00;
    opacity: 0.5;
    text-align: left;
    line-height: 1.6;
}

/* Responsive */
@media (max-width: 768px) {
    .holo-display {
        padding: 40px 30px;
    }

    .logo-main {
        font-size: 36px;
        letter-spacing: 6px;
    }

    .about-section {
        padding: 15px 20px;
        max-width: 100%;
    }

    .about-text {
        font-size: 11px;
    }

    .about-sources span {
        font-size: 9px;
        padding: 2px 6px;
    }

    .terminal {
        max-width: 280px;
        font-size: 9px;
    }

    .central-orb {
        width: 70px;
        height: 70px;
    }

    .data-stream {
        height: 60px;
    }
}
</style>

<div id="loading-screen">
    <canvas id="matrix-canvas"></canvas>
    <div class="radial-overlay"></div>

    <div class="system-info">
        SYS_ID: STRM-IQ-4200<br>
        STATUS: ONLINE<br>
        CONN: SECURED
    </div>

    <div class="load-container">
        <div class="holo-display">
            <div class="corner-indicator corner-tl"></div>
            <div class="corner-indicator corner-tr"></div>
            <div class="corner-indicator corner-bl"></div>
            <div class="corner-indicator corner-br"></div>

            <div class="logo-container">
                <div class="logo-main">
                    <span class="stream-text">STREAM</span><span class="iq-text">IQ</span>
                </div>
                <div class="subtitle">// QUANTUM INTELLIGENCE PLATFORM //</div>
            </div>

            <div class="central-orb">
                <div class="orb-ring"></div>
                <div class="orb-ring"></div>
                <div class="orb-core"></div>
            </div>

            <div class="about-section">
                <div class="about-title">// System Purpose</div>
                <div class="about-text">
                    Smart streaming tracker powered by AI. Discover what's worth watching right now with aggregated insights from YouTube reviews, Reddit discussions, and Rotten Tomatoes ratings. Get direct links to stream instantly.
                </div>
                <div class="about-sources">
                    <span>YouTube</span>
                    <span>Reddit</span>
                    <span>Rotten Tomatoes</span>
                    <span>Direct Links</span>
                </div>
            </div>

            <div class="data-stream">
                <div class="stream-line"></div>
                <div class="stream-line"></div>
                <div class="stream-line"></div>
                <div class="stream-line"></div>
            </div>

            <div class="terminal">
                <div class="terminal-line">&gt; <span class="success">INIT:</span> Neural network initialized...</div>
                <div class="terminal-line">&gt; <span class="info">LOAD:</span> Loading quantum protocols...</div>
                <div class="terminal-line">&gt; <span class="warning">SYNC:</span> Synchronizing data streams...</div>
                <div class="terminal-line">&gt; <span class="success">READY:</span> System online<span class="cursor"></span></div>
            </div>
        </div>
    </div>
</div>

<script>
// Matrix digital rain effect
(function() {
    const canvas = document.getElementById('matrix-canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const chars = 'STREAMIQ01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops = Array(Math.floor(columns)).fill(1);

    function drawMatrix() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#E8C547';
        ctx.font = fontSize + 'px monospace';

        for (let i = 0; i < drops.length; i++) {
            const text = chars[Math.floor(Math.random() * chars.length)];
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);

            if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }
    }

    const interval = setInterval(drawMatrix, 50);
    
    // Stop after 4 seconds (when loading screen fades)
    setTimeout(() => clearInterval(interval), 4000);
})();
</script>
"""

components.html(loading_screen, height=1)

st.markdown("""<style>
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stSidebar"]{display:none!important}
.block-container{padding:0!important;max-width:100%!important}
body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#07070A!important}
iframe{border:none!important}
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_all_data():
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    if not url or not key: return None, None, "No credentials"
    try:
        db = create_client(url, key)
        sr = db.table('scores').select('*, content(*)').order('final_score', desc=True).execute()
        dr = db.table('discover_content').select('*').order('popularity', desc=True).execute()
        watch = []
        for row in (sr.data or []):
            c = row.pop('content', {}) or {}
            watch.append({**row,
                'title': str(c.get('title','?')), 'platform': str(c.get('platform','?')),
                'content_type': str(c.get('content_type','?')), 'release_year': c.get('release_year'),
                'tmdb_id': c.get('tmdb_id'), 'poster_path': c.get('poster_path'),
                'overview': str(c.get('overview') or '')[:300],
                'imdb_rating': c.get('imdb_rating'), 'discovery_source': c.get('discovery_source','catalog'),
                'stream_url': str(c.get('stream_url') or ''),
                'hindi_dub': bool(c.get('hindi_dub', False)),
            })
        return watch, dr.data or [], None
    except Exception as e:
        return None, None, str(e)

watch_data, discover_data, err = load_all_data()
if err or not watch_data:
    st.error(f"Data error: {err}")
    st.stop()

def safe(v):
    if v is None: return ''
    try:
        if isinstance(v, float) and math.isnan(v): return ''
    except: pass
    return str(v)

def to_js(data, disc=False):
    out = []
    for r in data:
        if disc:
            rating = r.get('imdb_rating')
            try:
                if rating is not None and math.isnan(float(rating)): rating = 0
            except: rating = 0
            out.append({'title': safe(r.get('title')), 'platform': safe(r.get('platform')),
                'type': safe(r.get('content_type')), 'year': safe(r.get('release_year')),
                'rating': float(rating or 0), 'poster': safe(r.get('poster_path')),
                'overview': safe(r.get('overview',''))[:280], 'category': safe(r.get('category')),
                'tmdb_id': safe(r.get('tmdb_id')),
                'stream_url': safe(r.get('stream_url', '')),
                'hindi_dub': bool(r.get('hindi_dub', False))})
        else:
            out.append({'title': safe(r.get('title')), 'platform': safe(r.get('platform')),
                'type': safe(r.get('content_type')), 'year': safe(r.get('release_year')),
                'score': float(r.get('final_score') or 0), 'label': safe(r.get('label')),
                'poster': safe(r.get('poster_path')), 'overview': safe(r.get('overview',''))[:280],
                'tmdb_id': safe(r.get('tmdb_id')),
                'trending': safe(r.get('discovery_source')).lower() == 'trending',
                'polarizing': bool(r.get('is_polarizing', False)),
                'yt': float(r.get('youtube_score') or 0), 'reddit': float(r.get('reddit_score') or 0),
                'imdb': float(r.get('imdb_score') or 0), 'reviews': int(r.get('review_count') or 0),
                'stream_url': safe(r.get('stream_url', '')),
                'hindi_dub': bool(r.get('hindi_dub', False))})
    return json.dumps(out)

WJ = to_js(watch_data)
DJ = to_js(discover_data, disc=True)
TS = datetime.now().strftime('%d %b %Y · %H:%M')


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;500;700;800&family=Syne+Mono&family=Spectral:ital,wght@0,300;0,400;1,300;1,400&display=swap" rel="stylesheet"/>
<title>StreamIQ</title>
<style>

/* ═══════════════════════════════════════════════════
   DESIGN SYSTEM — "Projection Room"
   One accent. Absolute void. Cinematic typography.
═══════════════════════════════════════════════════ */
:root {
  /* the void */
  --v0:  #07070A;
  --v1:  #0C0C10;
  --v2:  #111116;
  --v3:  #18181F;
  --v4:  #222230;

  /* projected light — one color only */
  --gold:   #E8C547;
  --gold2:  #F5D76E;
  --gold-d: #A8852A;
  --gold-g: rgba(232,197,71,0.06);

  /* text scale — bumped up for readability */
  --t-bright: #F5F5F0;
  --t-mid:    #B0B0C0;
  --t-dim:    #707080;
  --t-void:   #505060;

  /* platform colours */
  --c-nf: #E50914;
  --c-pv: #00ADEF;
  --c-ap: #6BB3F7;
  --c-jh: #A855F7;
  --c-jc: #F97316;

  /* typography */
  --serif:  'DM Serif Display', Georgia, serif;
  --sans:   'Syne', sans-serif;
  --mono:   'Syne Mono', 'Courier New', monospace;
  --body:   'Spectral', Georgia, serif;

  --ease:   cubic-bezier(0.16, 1, 0.3, 1);
  --ease2:  cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; -webkit-font-smoothing: antialiased; color-scheme: dark; }

body {
  background: var(--v0);
  color: var(--t-mid);
  font-family: var(--body);
  min-height: 100vh;
  overflow-x: hidden;
  cursor: default;
}

/* cinematic vignette */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: radial-gradient(ellipse 100% 100% at 50% 50%, transparent 40%, rgba(0,0,0,0.7) 100%);
}

/* grain — ultra subtle */
body::after {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.022;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size: 256px;
}

::-webkit-scrollbar { width: 2px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--v4); border-radius: 1px; }

/* custom cursor */
* { cursor: none !important; }
#cursor {
  position: fixed; z-index: 9999; pointer-events: none;
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--gold);
  transform: translate(-50%, -50%);
  transition: transform 0.1s, width 0.25s var(--ease), height 0.25s var(--ease), opacity 0.2s;
  mix-blend-mode: difference;
}
#cursor-ring {
  position: fixed; z-index: 9998; pointer-events: none;
  width: 36px; height: 36px; border-radius: 50%;
  border: 1px solid rgba(232,197,71,0.35);
  transform: translate(-50%, -50%);
  transition: transform 0.45s var(--ease), width 0.4s var(--ease), height 0.4s var(--ease), opacity 0.3s, border-color 0.3s;
}
body:has(.card:hover) #cursor-ring,
body:has(.dc:hover) #cursor-ring {
  width: 60px; height: 60px;
  border-color: rgba(232,197,71,0.6);
}

#app { position: relative; z-index: 1; max-width: 1600px; margin: 0 auto; padding: 0 56px 140px; }

/* ═══════════════════════════════════════════════════
   MASTHEAD
═══════════════════════════════════════════════════ */
.mast {
  padding: 52px 0 0;
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: end;
  position: relative;
}

/* horizontal rule — like a film frame edge */
.mast::after {
  content: '';
  position: absolute; bottom: -28px; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, var(--gold) 0%, rgba(232,197,71,0.15) 30%, transparent 70%);
}

.brand { display: flex; flex-direction: column; gap: 10px; }

.logo-lockup {
  display: flex; align-items: baseline; gap: 16px;
}
.logo-word {
  font-family: var(--serif);
  font-size: clamp(3.8rem, 6vw, 6rem);
  font-style: italic;
  font-weight: 400;
  line-height: 0.9;
  letter-spacing: -0.03em;
  color: var(--t-bright);
  position: relative;
}
/* the IQ — gold, upright, smaller */
.logo-word .iq {
  font-style: normal;
  color: var(--gold);
  font-size: 0.55em;
  vertical-align: super;
  letter-spacing: 0.08em;
  font-family: var(--sans);
  font-weight: 800;
}

.logo-sub {
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--t-dim);
  display: flex; align-items: center; gap: 10px;
}
.logo-sub::before {
  content: '';
  display: inline-block; width: 20px; height: 1px;
  background: var(--gold);
}

/* live pulse */
.pulse-dot {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: 0.54rem;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--gold); opacity: 0.7;
}
.pulse-dot::before {
  content: ''; width: 4px; height: 4px; border-radius: 50%;
  background: var(--gold);
  box-shadow: 0 0 6px var(--gold);
  animation: heartbeat 2.4s ease-in-out infinite;
}
@keyframes heartbeat {
  0%,100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.3; transform: scale(0.6); }
}

.mast-right {
  display: flex; flex-direction: column; align-items: flex-end; gap: 8px;
  padding-bottom: 4px;
}
.mast-ts {
  font-family: var(--mono); font-size: 0.6rem;
  color: var(--t-void); letter-spacing: 0.06em;
}
.mast-kpi {
  display: flex; gap: 28px;
}
.kpi {
  text-align: right;
  display: flex; flex-direction: column; gap: 2px;
}
.kpi-n {
  font-family: var(--serif); font-style: italic;
  font-size: 2rem; line-height: 1;
  color: var(--t-bright); letter-spacing: -0.03em;
}
.kpi-l {
  font-family: var(--mono); font-size: 0.5rem;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--t-dim);
}

/* ═══════════════════════════════════════════════════
   NAV TABS
═══════════════════════════════════════════════════ */
.tabs {
  display: flex; align-items: center; gap: 0;
  padding: 40px 0 0;
  margin-bottom: 40px;
  border-bottom: 1px solid var(--v3);
  position: relative;
}
.tab {
  font-family: var(--sans); font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--t-dim); background: none; border: none;
  padding: 14px 32px 12px; position: relative;
  transition: color 0.25s;
}
.tab::after {
  content: ''; position: absolute;
  bottom: -1px; left: 0; right: 0; height: 1px;
  background: var(--gold);
  box-shadow: 0 0 10px var(--gold), 0 0 20px rgba(232,197,71,0.4);
  transform: scaleX(0); transform-origin: left;
  transition: transform 0.4s var(--ease);
}
.tab.on { color: var(--gold); }
.tab.on::after { transform: scaleX(1); }
.tab:hover { color: var(--t-mid); }
.tab-sep {
  width: 1px; height: 14px;
  background: var(--v4); margin: 0 4px;
  align-self: center;
}

/* frame count decoration */
.tabs-frames {
  margin-left: auto;
  font-family: var(--mono); font-size: 0.5rem;
  letter-spacing: 0.14em; color: var(--t-void);
  display: flex; align-items: center; gap: 6px;
  padding-bottom: 12px;
}
.frame-holes {
  display: flex; gap: 4px;
}
.fh {
  width: 5px; height: 5px; border-radius: 1px;
  border: 1px solid var(--v4);
}

/* ═══════════════════════════════════════════════════
   STATS RIBBON — editorial numbers
═══════════════════════════════════════════════════ */
.stats-ribbon {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1px;
  background: var(--v3);
  border: 1px solid var(--v3);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 36px;
}
.srib {
  background: var(--v1);
  padding: 22px 28px;
  position: relative;
  overflow: hidden;
  transition: background 0.2s;
}
.srib::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: var(--c, transparent);
  opacity: 0.6;
}
.srib::after { display: none; }
.srib.on { background: rgba(255,255,255,0.05); outline: 1px solid var(--c, #fff); }
.srib.on::before { opacity: 1; height: 2px; }
.srib-n {
  font-family: var(--serif); font-style: italic;
  font-size: 2.8rem; line-height: 1; font-weight: 400;
  letter-spacing: -0.04em;
  color: var(--c, var(--t-bright));
  margin-bottom: 6px;
  transition: color 0.3s;
}
.srib-l {
  font-family: var(--mono); font-size: 0.52rem;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--t-dim);
}

/* ═══════════════════════════════════════════════════
   CONTROLS — minimal filter bar
═══════════════════════════════════════════════════ */
.ctrl {
  display: flex; flex-wrap: wrap;
  align-items: center; gap: 6px;
  margin-bottom: 32px;
}
.pill {
  font-family: var(--mono); font-size: 0.56rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--t-mid); background: transparent;
  border: 1px solid var(--v4); border-radius: 1px;
  padding: 6px 14px;
  transition: all 0.18s; white-space: nowrap;
}
.pill:hover { color: var(--t-bright); border-color: var(--t-dim); }
.pill.on {
  color: var(--v0); background: var(--gold);
  border-color: var(--gold);
  box-shadow: 0 0 16px rgba(232,197,71,0.3);
}
.ctrl-sep { width: 1px; height: 20px; background: var(--t-dim); margin: 0 10px; opacity: 0.5; }

.search-wrap { margin-left: auto; position: relative; }
.search-icon {
  position: absolute; left: 12px; top: 50%; transform: translateY(-50%);
  color: var(--t-void); font-size: 0.8rem; pointer-events: none;
  font-family: var(--mono);
}
.search {
  font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.06em;
  color: var(--t-mid); background: var(--v1);
  border: 1px solid var(--v3); border-radius: 1px;
  padding: 8px 14px 8px 32px; width: 200px; outline: none;
  transition: border-color 0.2s, width 0.35s var(--ease);
}
.search::placeholder { color: var(--t-void); }
.search:focus { border-color: var(--gold-d); width: 260px; }

.sort-sel {
  font-family: var(--mono); font-size: 0.56rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--t-void); background: var(--v1);
  border: 1px solid var(--v3); border-radius: 1px;
  padding: 8px 12px; outline: none;
}

/* ═══════════════════════════════════════════════════
   WATCH GRID
═══════════════════════════════════════════════════ */
.watch-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 14px;
}

/* ═══════════════════════════════════════════════════
   CARD — the centrepiece
═══════════════════════════════════════════════════ */
.card {
  position: relative;
  background: var(--v1);
  border: 1px solid var(--v3);
  border-radius: 2px;
  overflow: hidden;
  transition:
    transform 0.45s var(--ease),
    border-color 0.3s,
    box-shadow 0.45s var(--ease);
  will-change: transform;
}
.card:hover {
  transform: translateY(-10px) scale(1.015);
  border-color: rgba(232,197,71,0.22);
  box-shadow:
    0 30px 60px rgba(0,0,0,0.7),
    0 0 0 1px rgba(232,197,71,0.1),
    0 0 40px rgba(232,197,71,0.04);
  z-index: 3;
}

/* platform accent — top edge only, clean line */
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--plat, var(--v3));
  z-index: 2;
  opacity: 0.55;
  transition: opacity 0.3s;
}
.card:hover::before { opacity: 1; }

/* POSTER */
.poster {
  position: relative;
  aspect-ratio: 2/3;
  overflow: hidden;
  background: var(--v2);
}
.poster img {
  width: 100%; height: 100%;
  object-fit: cover; display: block;
  transition: transform 0.7s var(--ease), filter 0.5s;
}
.card:hover .poster img {
  transform: scale(1.06);
  filter: brightness(0.28) saturate(0.5);
}
.poster-ph {
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  font-size: 2.8rem; color: var(--v4);
}

/* overview reveal on hover */
.poster-reveal {
  position: absolute; inset: 0; z-index: 3;
  display: flex; align-items: center; justify-content: center;
  padding: 20px 16px;
  opacity: 0;
  transform: translateY(8px);
  transition: opacity 0.4s var(--ease), transform 0.4s var(--ease);
}
.card:hover .poster-reveal { opacity: 1; transform: translateY(0); }
.poster-reveal-text {
  font-family: var(--body); font-style: italic;
  font-size: 0.72rem; line-height: 1.65;
  color: rgba(245,245,240,0.85);
  text-align: center;
  display: -webkit-box; -webkit-line-clamp: 7; -webkit-box-orient: vertical;
  overflow: hidden;
}

/* SCORE — the editorial number, overlapping poster and body */
.score-flag {
  position: absolute;
  bottom: -1px; right: 0;
  z-index: 4;
  width: 52px;
  display: flex; flex-direction: column; align-items: center;
  background: var(--v1);
  border-top: 1px solid var(--v3);
  border-left: 1px solid var(--v3);
  padding: 6px 0 4px;
  transition: background 0.3s;
}
.card:hover .score-flag { background: var(--v2); }
.score-num {
  font-family: var(--serif); font-style: italic;
  font-size: 1.45rem; line-height: 1; font-weight: 400;
  letter-spacing: -0.04em;
  color: var(--sc, var(--t-mid));
  transition: color 0.3s;
}
.score-label {
  font-family: var(--mono); font-size: 0.38rem;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--t-void); margin-top: 1px;
}

/* badges */
.badges {
  position: absolute; top: 10px; left: 10px; z-index: 4;
  display: flex; flex-direction: column; gap: 4px;
}
.badge {
  font-family: var(--mono); font-size: 0.46rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  padding: 3px 7px; border-radius: 1px;
  backdrop-filter: blur(12px);
}
.badge-hot {
  background: rgba(232,197,71,0.12); color: var(--gold);
  border: 1px solid rgba(232,197,71,0.25);
}
.badge-pol {
  background: rgba(255,51,51,0.12); color: #FF4455;
  border: 1px solid rgba(255,51,51,0.25);
}

/* CARD BODY */
.card-body {
  padding: 14px 16px 14px 16px;
  position: relative;
}
.card-title {
  font-family: var(--sans); font-size: 0.85rem; font-weight: 700;
  letter-spacing: 0.01em; line-height: 1.3;
  color: var(--t-bright); margin-bottom: 8px;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  transition: color 0.2s;
  padding-right: 48px; /* make room for score flag */
}
.card:hover .card-title { color: var(--gold2); }

.card-meta {
  display: flex; align-items: center; gap: 6px; margin-bottom: 10px;
}
.card-plat {
  font-family: var(--mono); font-size: 0.5rem;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--pl, var(--t-mid));
  border-bottom: 1px solid var(--pl, var(--t-mid));
  padding-bottom: 1px;
}
.card-type {
  font-family: var(--mono); font-size: 0.5rem;
  color: var(--t-dim); letter-spacing: 0.06em;
}

/* score bar — thin golden line */
.score-line {
  height: 1px; background: var(--v3);
  margin-bottom: 8px; position: relative; overflow: visible;
}
.score-line-fill {
  position: absolute; top: 0; left: 0; height: 1px;
  background: var(--sc, var(--t-void));
  box-shadow: 0 0 6px var(--sc, transparent);
  transition: width 1s var(--ease);
}

.card-verdict {
  font-family: var(--body); font-style: italic;
  font-size: 0.65rem; line-height: 1;
  color: var(--sc, var(--t-dim));
}

/* ═══════════════════════════════════════════════════
   OVERLAY + DETAIL PANEL
═══════════════════════════════════════════════════ */
#overlay {
  position: absolute; inset: 0; z-index: 100;
  background: rgba(0,0,0,0); backdrop-filter: blur(0px);
  pointer-events: none;
  transition: background 0.5s, backdrop-filter 0.5s;
  visibility: hidden;
}
#overlay.on {
  background: rgba(0,0,0,0.82); backdrop-filter: blur(10px);
  pointer-events: all;
  visibility: visible;
}

#panel {
  position: absolute; top: 0; right: 0;
  width: min(620px, 100vw); height: 100%;
  min-height: 100%;
  visibility: hidden;
  background: var(--v1);
  border-left: 1px solid var(--v3);
  z-index: 101;
  transform: translateX(100%);
  transition: transform 0.5s var(--ease);
  overflow-y: auto; overflow-x: hidden;
  display: flex; flex-direction: column;
  overflow-anchor: none;
}
#panel.on { transform: translateX(0); visibility: visible; }
#panel::-webkit-scrollbar { width: 2px; }
#panel::-webkit-scrollbar-thumb { background: var(--v4); }

/* close bar */
.panel-topbar {
  position: sticky; top: 0; z-index: 10;
  display: flex; justify-content: space-between; align-items: center;
  padding: 20px 28px;
  background: linear-gradient(to bottom, var(--v1) 60%, transparent);
}
.panel-close-btn {
  font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--t-void); background: transparent;
  border: 1px solid var(--v3); border-radius: 1px;
  padding: 7px 14px; transition: color 0.2s, border-color 0.2s;
}
.panel-close-btn:hover { color: var(--t-mid); border-color: var(--v4); }
.panel-frame-id {
  font-family: var(--mono); font-size: 0.5rem;
  color: var(--t-void); letter-spacing: 0.1em;
}

/* panel poster — cinematic wide with fade */
.panel-poster-wrap {
  position: relative; width: 100%;
  aspect-ratio: 16/9; overflow: hidden;
  margin-top: -60px;
}
.panel-poster-wrap img {
  width: 100%; height: 100%;
  object-fit: cover;
  filter: brightness(0.35) saturate(0.6);
}
.panel-poster-wrap::after {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(to bottom,
    transparent 0%, rgba(12,12,16,0.6) 50%, var(--v1) 100%);
}
.panel-poster-ph {
  width: 100%; height: 100%;
  background: var(--v2);
  display: flex; align-items: center; justify-content: center;
  font-size: 5rem;
}

/* panel content */
.panel-body {
  padding: 0 36px 48px;
  display: flex; flex-direction: column; gap: 0;
}
.panel-plat {
  font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--pl, var(--t-void));
  display: inline-flex; align-items: center; gap: 8px;
  margin-bottom: 14px;
}
.panel-plat::before {
  content: ''; width: 16px; height: 1px;
  background: var(--pl, var(--t-void));
}

.panel-title {
  font-family: var(--serif); font-style: italic;
  font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 400;
  line-height: 0.95; letter-spacing: -0.03em;
  color: var(--t-bright); margin-bottom: 20px;
}

.panel-meta-row {
  display: flex; flex-wrap: wrap; gap: 20px;
  margin-bottom: 28px;
  padding-bottom: 28px;
  border-bottom: 1px solid var(--v3);
}
.pm-item {
  font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--t-dim);
}
.pm-item span { color: var(--t-bright); display: block; margin-top: 3px; }

/* big score section */
.panel-score-section {
  display: flex; align-items: flex-start; gap: 24px;
  padding-bottom: 28px;
  margin-bottom: 28px;
  border-bottom: 1px solid var(--v3);
}
.panel-big-score {
  font-family: var(--serif); font-style: italic;
  font-size: 7rem; font-weight: 400;
  line-height: 0.85; letter-spacing: -0.05em;
  color: var(--sc, var(--t-mid));
  flex-shrink: 0;
  text-shadow: 0 0 80px var(--sc-a, transparent);
}
.panel-score-meta {
  display: flex; flex-direction: column; gap: 6px;
  padding-top: 12px;
}
.panel-verdict-text {
  font-family: var(--body); font-style: italic;
  font-size: 1.4rem; line-height: 1.2;
  color: var(--t-bright);
}
.panel-verdict-sub {
  font-family: var(--mono); font-size: 0.5rem;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--t-void);
}
.panel-tags {
  display: flex; gap: 6px; margin-top: 8px;
  flex-wrap: wrap;
}
.ptag {
  font-family: var(--mono); font-size: 0.5rem;
  letter-spacing: 0.12em; text-transform: uppercase;
  padding: 4px 9px; border-radius: 1px;
}

/* overview */
.panel-overview {
  font-family: var(--body); font-size: 0.9rem;
  font-weight: 300; line-height: 1.8;
  color: var(--t-mid);
  margin-bottom: 28px;
}

/* breakdown */
.breakdown { margin-bottom: 28px; }
.breakdown-hd {
  font-family: var(--mono); font-size: 0.52rem;
  letter-spacing: 0.24em; text-transform: uppercase;
  color: var(--t-void); margin-bottom: 16px;
  display: flex; align-items: center; gap: 10px;
}
.breakdown-hd::after {
  content: ''; flex: 1; height: 1px; background: var(--v3);
}
.bd-row {
  display: grid; grid-template-columns: 80px 1fr 38px;
  align-items: center; gap: 14px; margin-bottom: 12px;
}
.bd-lbl {
  font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--t-void);
}
.bd-track {
  height: 1px; background: var(--v3); position: relative;
}
.bd-fill {
  position: absolute; top: 0; left: 0; height: 1px;
  transition: width 0.9s var(--ease);
  box-shadow: 0 0 8px currentColor;
}
.bd-val {
  font-family: var(--serif); font-style: italic;
  font-size: 1rem; color: var(--t-mid); text-align: right;
}

.panel-link {
  font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--gold); text-decoration: none;
  display: inline-flex; align-items: center; gap: 8px;
  border-bottom: 1px solid rgba(232,197,71,0.25);
  padding-bottom: 2px;
  transition: opacity 0.2s;
}
.panel-link:hover { opacity: 0.65; }
.panel-actions {
  display: flex; align-items: center; gap: 20px;
  margin-top: 10px; flex-wrap: wrap;
}
.panel-watch-btn {
  font-family: var(--mono); font-size: 0.6rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  text-decoration: none; padding: 7px 18px;
  display: inline-flex; align-items: center; gap: 6px;
  font-weight: 700; transition: opacity 0.2s;
}
.panel-watch-btn:hover { opacity: 0.8; }

/* ═══════════════════════════════════════════════════
   DISCOVER GRID
═══════════════════════════════════════════════════ */
.disc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 12px;
}
.dc {
  display: flex; flex-direction: column;
  background: var(--v1); border: 1px solid var(--v3);
  border-radius: 2px; overflow: hidden;
  color: inherit;
  transition: transform 0.4s var(--ease), border-color 0.25s, box-shadow 0.4s var(--ease);
  position: relative;
}
.dc::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: var(--cc, transparent);
  opacity: 0.7; z-index: 2;
}
.dc:hover {
  transform: translateY(-6px);
  border-color: rgba(232,197,71,0.15);
  box-shadow: 0 20px 40px rgba(0,0,0,0.6);
}
.dc-poster {
  aspect-ratio: 2/3; overflow: hidden;
  background: var(--v2); position: relative;
}
.dc-poster img {
  width: 100%; height: 100%; object-fit: cover; display: block;
  transition: transform 0.55s var(--ease), filter 0.3s;
}
.dc:hover .dc-poster img { transform: scale(1.05); filter: brightness(0.85); }
.dc-poster-ph {
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem; color: var(--v4);
}
.dc-body {
  padding: 10px 12px 12px;
  flex: 1; display: flex; flex-direction: column; gap: 5px;
}
.dc-title {
  font-family: var(--sans); font-size: 0.75rem; font-weight: 700;
  letter-spacing: 0.01em; color: var(--t-bright); line-height: 1.3;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.dc-sub {
  font-family: var(--mono); font-size: 0.5rem;
  color: var(--t-void); letter-spacing: 0.06em;
}
.dc-foot {
  display: flex; align-items: center; gap: 6px; margin-top: auto;
}
.dc-rating {
  font-family: var(--serif); font-style: italic;
  font-size: 0.8rem; color: var(--gold);
}
.dc-tag {
  font-family: var(--mono); font-size: 0.48rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  padding: 2px 6px; border-radius: 1px;
  background: rgba(255,255,255,0.03);
  border: 1px solid currentColor; opacity: 0.8;
}
.jw-link {
  font-family: var(--mono); font-size: 0.48rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--gold); text-decoration: none;
  display: block; text-align: center;
  padding: 2px 0;
  opacity: 0.7;
  transition: opacity 0.2s;
}
.jw-link:hover { opacity: 1; }

/* ═══════════════════════════════════════════════════
   UTILITIES
═══════════════════════════════════════════════════ */
.empty {
  padding: 100px 40px; text-align: center;
  font-family: var(--mono); font-size: 0.7rem;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--t-void);
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fu { animation: fadeUp 0.6s var(--ease) both; }

/* ── section label ── */
.section-label {
  font-family: var(--mono); font-size: 0.52rem;
  letter-spacing: 0.26em; text-transform: uppercase;
  color: var(--t-void); margin-bottom: 14px;
  display: flex; align-items: center; gap: 12px;
}
.section-label::after {
  content: ''; flex: 1; height: 1px; background: var(--v3);
}


/* ═══════════════════════════════════════════════════
   MOBILE RESPONSIVE
═══════════════════════════════════════════════════ */
@media (max-width: 768px) {
  #app { padding: 0 16px 80px; }

  /* hide cursor on mobile */
  #cursor, #cursor-ring { display: none; }
  * { cursor: auto !important; }

  /* masthead */
  .mast { grid-template-columns: 1fr; gap: 12px; padding: 24px 0 0; }
  .mast-right { align-items: flex-start; flex-direction: row; gap: 20px; }
  .mast-kpi { gap: 16px; }
  .logo-word { font-size: 2.8rem; }
  .kpi-n { font-size: 1.4rem; }
  .mast-ts { display: none; }

  /* tabs */
  .tabs { padding: 20px 0 0; margin-bottom: 24px; }
  .tab { padding: 12px 16px 10px; font-size: 0.6rem; }
  .tabs-frames { display: none; }

  /* stats ribbon - 2 columns on mobile */
  .stats-ribbon { grid-template-columns: repeat(3, 1fr); }
  .srib { padding: 14px 12px; }
  .srib-n { font-size: 1.8rem; }
  .srib::after { display: none; }
.srib.on { background: rgba(255,255,255,0.05); outline: 1px solid var(--c, #fff); }
.srib.on::before { opacity: 1; height: 2px; }

  /* controls - wrap nicely */
  .ctrl { gap: 6px; margin-bottom: 20px; }
  .pill { font-size: 0.5rem; padding: 5px 10px; }
  .search-wrap { margin-left: 0; width: 100%; }
  .search { width: 100% !important; }
  .sort-sel { font-size: 0.5rem; }

  /* card grid - 2 columns on mobile */
  .watch-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .disc-grid  { grid-template-columns: repeat(2, 1fr); gap: 8px; }

  /* panel - full screen on mobile */
  #panel {
    width: 100vw !important;
    height: 100% !important;
  }
  .panel-title { font-size: 1.6rem; }
  .panel-big-score { font-size: 5rem; }
  .panel-body { padding: 0 20px 40px; }

  /* close button - big and obvious on mobile */
  .panel-topbar {
    padding: 16px 20px;
    background: var(--v1);
    position: sticky;
    top: 0;
    z-index: 20;
  }
  .panel-close-btn {
    font-size: 0.7rem;
    padding: 10px 20px;
    color: var(--t-bright) !important;
    border-color: var(--t-dim) !important;
    background: var(--v3) !important;
  }
  .panel-frame-id { font-size: 0.45rem; }
}


/* ═══════════════════════════════════════════════════
   LG TV INTEGRATION STYLES
═══════════════════════════════════════════════════ */

.tv-send-btn {
  flex: 1;
  min-width: 200px;
  padding: 12px 20px;
  background: rgba(232,197,71,0.08);
  color: var(--gold);
  border: 1px solid rgba(232,197,71,0.25);
  border-radius: 2px;
  font-family: var(--mono);
  font-weight: 600;
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  transition: all 0.25s var(--ease);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(8px);
}

.tv-send-btn:hover {
  background: rgba(232,197,71,0.15);
  border-color: var(--gold);
  box-shadow: 0 0 20px rgba(232,197,71,0.15);
  transform: translateY(-1px);
}

.tv-send-btn:active {
  transform: translateY(0);
}

.tv-send-btn.sending {
  background: rgba(16,185,129,0.12);
  border-color: rgba(16,185,129,0.4);
  color: #10B981;
  pointer-events: none;
}

.tv-send-btn.error {
  background: rgba(239,68,68,0.12);
  border-color: rgba(239,68,68,0.4);
  color: #EF4444;
}

.tv-icon {
  font-size: 14px;
  opacity: 0.8;
  transition: opacity 0.3s;
}

.tv-send-btn:hover .tv-icon {
  opacity: 1;
}

@keyframes subtle-pulse {
  0%, 100% { opacity: 0.8; }
  50% { opacity: 1; }
}

.tv-send-btn.sending .tv-icon {
  animation: subtle-pulse 1.5s ease-in-out infinite;
}

/* Toast Notification */
.toast {
  position: fixed;
  bottom: 30px;
  right: 30px;
  background: rgba(12,12,16,0.95);
  border: 1px solid rgba(232,197,71,0.35);
  border-radius: 2px;
  padding: 16px 24px;
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.08em;
  color: var(--gold);
  z-index: 10000;
  backdrop-filter: blur(12px);
  box-shadow: 0 4px 24px rgba(0,0,0,0.6), 0 0 1px rgba(232,197,71,0.5);
  transform: translateY(150%);
  transition: transform 0.35s var(--ease);
  display: flex;
  align-items: center;
  gap: 12px;
}

.toast.show {
  transform: translateY(0);
}

.toast-icon {
  font-size: 18px;
  opacity: 0.9;
}

/* Compact TV button for Discover cards */
.dc .tv-send-btn {
  font-size: 0.56rem;
  padding: 8px 14px;
  min-width: auto;
  width: 100%;
  letter-spacing: 0.14em;
}

.dc .tv-icon {
  font-size: 12px;
}
</style>
</head>
<body>

<!-- custom cursor -->
<div id="cursor"></div>
<div id="cursor-ring"></div>

<div id="app">

<!-- ── MASTHEAD ── -->
<div class="mast">
  <div class="brand">
    <div class="logo-lockup">
      <div class="logo-word">Stream<span class="iq">IQ</span></div>
    </div>
    <div class="logo-sub">
      <span class="pulse-dot">Live</span>
      &nbsp;Streaming Intelligence
    </div>
  </div>
  <div class="mast-right">
    <div class="mast-ts" id="ts"></div>
    <div class="mast-kpi" id="hkpi"></div>
  </div>
</div>

<!-- ── TABS ── -->
<div class="tabs">
  <button class="tab on" onclick="showTab('watch',this)">Watch Now</button>
  <div class="tab-sep"></div>
  <button class="tab" onclick="showTab('disc',this)">Discover</button>

  <div class="tabs-frames">
    <div class="frame-holes">
      <div class="fh"></div><div class="fh"></div><div class="fh"></div>
      <div class="fh"></div><div class="fh"></div><div class="fh"></div>
    </div>
    <span id="nav-ts"></span>
    <div class="frame-holes">
      <div class="fh"></div><div class="fh"></div><div class="fh"></div>
      <div class="fh"></div><div class="fh"></div><div class="fh"></div>
    </div>
  </div>
</div>

<!-- ── WATCH TAB ── -->
<div id="tab-watch">
  <div class="stats-ribbon" id="ws"></div>
  <div class="ctrl">
    <button class="pill on" data-p="all"           onclick="sP('all',this)">All</button>
    <button class="pill" data-p="Netflix"           onclick="sP('Netflix',this)">Netflix</button>
    <button class="pill" data-p="Prime Video"       onclick="sP('Prime Video',this)">Prime</button>
    <button class="pill" data-p="Jiohotstar"        onclick="sP('Jiohotstar',this)">Jiohotstar</button>
    <button class="pill" data-p="Apple TV+"         onclick="sP('Apple TV+',this)">Apple TV+</button>
    <div class="ctrl-sep"></div>
    <button class="pill on" data-t="all"   onclick="sT('all',this)">All Types</button>
    <button class="pill" data-t="movie"    onclick="sT('movie',this)">Films</button>
    <button class="pill" data-t="tv"       onclick="sT('tv',this)">Series</button>
    <div class="ctrl-sep"></div>
    <button class="pill" id="w-hindi-btn" onclick="toggleHindiW(this)">🎙 Hindi Dub</button>
    <div class="search-wrap">
      <span class="search-icon">↳</span>
      <input class="search" placeholder="Search titles…" oninput="wSr(this.value)"/>
    </div>
    <select class="sort-sel" onchange="wSo(this.value)">
      <option value="score">Score ↓</option>
      <option value="title">A – Z</option>
      <option value="year">Newest</option>
    </select>
  </div>
  <div class="watch-grid" id="wg"></div>
</div>

<!-- ── DISCOVER TAB ── -->
<div id="tab-disc" style="display:none">
  <div class="stats-ribbon" id="ds"></div>
  <div class="ctrl">
    <button class="pill on" data-dp="all"         onclick="sdP('all',this)">All</button>
    <button class="pill" data-dp="Netflix"         onclick="sdP('Netflix',this)">Netflix</button>
    <button class="pill" data-dp="Prime Video"     onclick="sdP('Prime Video',this)">Prime</button>
    <button class="pill" data-dp="Jiohotstar"      onclick="sdP('Jiohotstar',this)">Jiohotstar</button>
    <button class="pill" data-dp="Apple TV+"       onclick="sdP('Apple TV+',this)">Apple TV+</button>
    <div class="ctrl-sep"></div>
    <button class="pill on" data-dt="all"   onclick="sdT('all',this)">All Types</button>
    <button class="pill" data-dt="movie"    onclick="sdT('movie',this)">Films</button>
    <button class="pill" data-dt="tv"       onclick="sdT('tv',this)">Series</button>
    <div class="ctrl-sep"></div>
    <button class="pill" id="d-hindi-btn" onclick="toggleHindiD(this)">🎙 Hindi Dub</button>
  </div>
  <div class="ctrl">
    <button class="pill on" data-c="all"              onclick="sC('all',this)">All</button>
    <button class="pill" data-c="classics"             onclick="sC('classics',this)">Classics</button>
    <button class="pill" data-c="underdog"             onclick="sC('underdog',this)">Hidden Gems</button>
    <button class="pill" data-c="indian"               onclick="sC('indian',this)">Hindi</button>
    <button class="pill" data-c="genre_action"         onclick="sC('genre_action',this)">Action</button>
    <button class="pill" data-c="genre_thriller"       onclick="sC('genre_thriller',this)">Thriller</button>
    <button class="pill" data-c="genre_horror"         onclick="sC('genre_horror',this)">Horror</button>
    <button class="pill" data-c="genre_comedy"         onclick="sC('genre_comedy',this)">Comedy</button>
    <button class="pill" data-c="genre_drama"          onclick="sC('genre_drama',this)">Drama</button>
    <button class="pill" data-c="genre_sci-fi"         onclick="sC('genre_sci-fi',this)">Sci-Fi</button>
    <div class="search-wrap">
      <span class="search-icon">↳</span>
      <input class="search" placeholder="Search…" oninput="dSr(this.value)"/>
    </div>
  </div>
  <div class="disc-grid" id="dg"></div>
</div>

</div><!-- #app -->

<!-- PANEL -->
<div id="overlay" onclick="closeP()"></div>
<div id="panel">
  <div class="panel-topbar">
    <div class="panel-frame-id" id="panel-fid">TITLE / 000</div>
    <button class="panel-close-btn" onclick="closeP()">✕ &nbsp;Close</button>
  </div>
  <div id="pi"></div>

</div>

<script>
/* ── DATA ── */
const W = __WJ__;
const D = __DJ__;

/* ── PLATFORM COLORS ── */
const PC = {
  'Netflix':     '#E50914',
  'Prime Video': '#00ADEF',
  'Apple TV+':   '#6BB3F7',
  'Jiohotstar':  '#A855F7',
  'JioCinema':   '#F97316'
};

/* ── DISCOVER COLORS ── */
const CC = {
  'classics':       '#E8C547',
  'underdog':       '#4DBFFF',
  'indian':         '#FF7043',
  'genre_action':   '#FF3355',
  'genre_thriller': '#B88EFF',
  'genre_horror':   '#00F5C4',
  'genre_comedy':   '#FFB830',
  'genre_drama':    '#4DBFFF',
  'genre_sci-fi':   '#C6F135',
  'genre_romance':  '#FF9EAA',
};
const CL = {
  'classics': 'Classic', 'underdog': 'Gem', 'indian': 'Hindi',
  'genre_action': 'Action', 'genre_thriller': 'Thriller',
  'genre_horror': 'Horror', 'genre_comedy': 'Comedy',
  'genre_drama': 'Drama', 'genre_sci-fi': 'Sci-Fi', 'genre_romance': 'Romance',
};

/* ── SCORE COLOR ── */
function sc(s) {
  if (s >= 75) return '#E8C547';
  if (s >= 60) return '#8CB4CC';
  if (s >= 45) return '#7A7A90';
  return '#553535';
}
function scA(s) {
  if (s >= 75) return 'rgba(232,197,71,0.35)';
  if (s >= 60) return 'rgba(140,180,204,0.25)';
  if (s >= 45) return 'rgba(122,122,144,0.2)';
  return 'rgba(85,53,53,0.2)';
}
function verdict(s) {
  if (s >= 75) return 'Essential Viewing';
  if (s >= 60) return 'Worth Your Time';
  if (s >= 45) return 'For Fans Only';
  return 'Skip It';
}
function pc(p) { return PC[p] || '#666'; }

/* ── STATE ── */
let wPlat='all', wType='all', wSearch='', wSort='score';
let dCat='all',  dSearch='';

/* ── CUSTOM CURSOR ── */
const cur = document.getElementById('cursor');
const ring = document.getElementById('cursor-ring');
let mx=0, my=0, rx=0, ry=0;
document.addEventListener('mousemove', e => {
  mx = e.clientX; my = e.clientY;
  cur.style.left = mx+'px'; cur.style.top = my+'px';
});
(function animRing() {
  rx += (mx - rx) * 0.14;
  ry += (my - ry) * 0.14;
  ring.style.left = rx+'px'; ring.style.top = ry+'px';
  requestAnimationFrame(animRing);
})();

/* ── FILTER ── */
function getFW() {
  let base = W;
  if (window._wFilter === 'worth') base = base.filter(d => d.score >= 60);
  if (window._wFilter === 'polar') base = base.filter(d => d.polarizing);
  if (window._wHindi) base = base.filter(d => d.hindi_dub === true);
  return base
    .filter(d => wPlat==='all' || d.platform===wPlat)
    .filter(d => wType==='all'  || d.type===wType)
    .filter(d => !wSearch || d.title.toLowerCase().includes(wSearch.toLowerCase()))
    .sort((a,b) => {
      if (wSort==='score') return b.score - a.score;
      if (wSort==='title') return a.title.localeCompare(b.title);
      return (b.year||0)-(a.year||0);
    });
}

/* ── STATS RIBBON ── */
var SRIB_COLORS = {
  'ds': ['#F5F5F0','#E8C547','#4DBFFF','#FF7043','#B88EFF'],
  'ws': ['#F5F5F0','#E8C547','#8CB4CC','#FF4455','#7A7A90']
};
function setSribActive(elId, idx) {
  var colors = SRIB_COLORS[elId] || [];
  document.querySelectorAll('#'+elId+' .srib').forEach(function(x,i){
    if (i === idx) {
      var c = colors[i] || '#fff';
      x.style.background = 'rgba(255,255,255,0.07)';
      x.style.borderTop = '2px solid ' + c;
      x.style.boxShadow = '0 0 12px ' + c + '33';
    } else {
      x.style.background = '';
      x.style.borderTop = '';
      x.style.boxShadow = '';
    }
  });
}
function renderStats(data, elId, isDisc) {
  let items;
  if (!isDisc) {
    const avg   = data.length ? (data.reduce((s,d)=>s+(d.score||0),0)/data.length).toFixed(0) : 0;
    const worth = data.filter(d=>d.score>=60).length;
    const pol   = data.filter(d=>d.polarizing).length;
    const revs  = data.reduce((s,d)=>s+d.reviews,0);
    items = [
      { n: data.length, l: 'Titles',        c: '#F5F5F0', fn: "window._wFilter=null;renderWatch();setSribActive('ws',0)" },
      { n: avg,         l: 'Avg Score',      c: '#E8C547' },
      { n: worth,       l: 'Worth Watching', c: '#8CB4CC', fn: "wWorth();setSribActive('ws',2)" },
      { n: pol,         l: 'Polarising',     c: '#FF4455', fn: "wPolar();setSribActive('ws',3)" },
      { n: revs,        l: 'Reviews',        c: '#7A7A90' },
    ];
    // update masthead KPIs
    document.getElementById('hkpi').innerHTML = `
      <div class="kpi"><div class="kpi-n">${data.length}</div><div class="kpi-l">Titles</div></div>
      <div class="kpi"><div class="kpi-n">${avg}</div><div class="kpi-l">Avg Score</div></div>
      <div class="kpi"><div class="kpi-n">${worth}</div><div class="kpi-l">Worth It</div></div>
    `;
  } else {
    const cats = {};
    D.forEach(d => cats[d.category] = (cats[d.category]||0)+1);
    const genres = Object.keys(cats).filter(k=>k.startsWith('genre_')).reduce((s,k)=>s+(cats[k]||0),0);
    items = [
      { n: data.length,         l: 'Titles',      c: '#F5F5F0', fn: "sC('all',null);setSribActive('ds',0)"       },
      { n: cats['classics']||0, l: 'Classics',    c: '#E8C547', fn: "sC('classics',null);setSribActive('ds',1)"  },
      { n: cats['underdog']||0, l: 'Hidden Gems', c: '#4DBFFF', fn: "sC('underdog',null);setSribActive('ds',2)"  },
      { n: cats['indian']||0,   l: 'Hindi',       c: '#FF7043', fn: "sC('indian',null);setSribActive('ds',3)"    },
      { n: genres,              l: 'Genre Picks', c: '#B88EFF', fn: "dCat='genre_picks';document.querySelectorAll('[data-c]').forEach(x=>x.classList.remove('on'));renderDisc();setSribActive('ds',4);"},
    ];
  }
  document.getElementById(elId).innerHTML = items.map(s =>
    `<div class="srib" data-n="${s.n}" style="--c:${s.c}${s.fn ? ';cursor:pointer' : ''}" ${s.fn ? `onclick="${s.fn}"` : ''}>
      <div class="srib-n">${s.n}</div>
      <div class="srib-l">${s.l}${s.fn ? ' ↗' : ''}</div>
    </div>`
  ).join('');
}

/* ── CARD ── */
function makeCard(d, i, wIdx) {
  const col = sc(d.score);
  const img = d.poster ? 'https://image.tmdb.org/t/p/w300'+d.poster : null;
  const platCol = pc(d.platform);
  const delay = (i % 24) * 0.025;
  const pct = d.score;

  return `
<div class="card fu" style="animation-delay:${delay}s;--plat:${platCol};--sc:${col};--sc-pct:${pct}%"
  data-widx="${wIdx}" onclick="openPanelIdx(this.dataset.widx, this)">
  <div class="poster">
    ${img
      ? `<img src="${img}" loading="lazy" alt=""/>`
      : `<div class="poster-ph">🎬</div>`}
    <div class="poster-reveal">
      <div class="poster-reveal-text"></div>
    </div>
    <div class="badges">
      ${d.trending   ? '<span class="badge badge-hot">Hot</span>' : ''}
      ${d.polarizing ? '<span class="badge badge-pol">Split</span>' : ''}
    </div>
    <div class="score-flag">
      <div class="score-num" style="color:${col}">${d.score.toFixed(0)}</div>
      <div class="score-label">IQ</div>
    </div>
  </div>
  <div class="card-body">
    <div class="card-title"></div>
    <div class="card-meta">
      <span class="card-plat" style="--pl:${platCol}"></span>
      <span class="card-type">${d.type==='tv'?'Series':'Film'} · ${d.year||'—'}</span>
    </div>
    <div class="score-line">
      <div class="score-line-fill" style="width:${pct}%;background:${col};box-shadow:0 0 8px ${col}"></div>
    </div>
    <div class="card-verdict">${verdict(d.score)}</div>
  </div>
</div>`;
}

/* ── STABLE INDEX MAP — built once, used by all renders ── */
const W_IDX_MAP = new Map();
W.forEach((d, i) => {
  W_IDX_MAP.set((d.tmdb_id||'')+'|'+(d.platform||'')+'|'+(d.title||''), i);
});
function wIdxFor(d) {
  const k = (d.tmdb_id||'')+'|'+(d.platform||'')+'|'+(d.title||'');
  const v = W_IDX_MAP.get(k);
  return v !== undefined ? v : -1;
}

/* ── RENDER WATCH ── */
function renderWatch() {
  const data = getFW();
  renderStats(W, 'ws', false);
  const el = document.getElementById('wg');
  if (!data.length) { el.innerHTML = '<div class="empty">No titles match your filters</div>'; return; }
  el.innerHTML = data.slice(0,80).map((d, i) => makeCard(d, i, wIdxFor(d))).join('');
  el.querySelectorAll('.card').forEach(card => {
    const idx = parseInt(card.dataset.widx);
    if (isNaN(idx) || idx < 0 || idx >= W.length) return;
    const d = W[idx];
    card.querySelector('.card-title').textContent = d.title || '';
    card.querySelector('.card-plat').textContent  = d.platform || '';
    card.querySelector('.poster-reveal-text').textContent = d.overview || d.title || '';
  });
}
/* ── PANEL ── */
/* ── PANEL — safe index-based lookup ── */
function openPanelIdx(idxStr, el) {
  const d = W[parseInt(idxStr)];
  if (!d) return;
  openPanel(d, el);
}

let panelCount = 0;
function openPanel(d, clickedEl) {
  panelCount++;
  const col  = sc(d.score);
  const colA = scA(d.score);
  const img  = d.poster ? 'https://image.tmdb.org/t/p/w780'+d.poster : null;
  const platCol = pc(d.platform);
  const tmdb = d.tmdb_id
    ? `https://www.themoviedb.org/${d.type==='tv'?'tv':'movie'}/${d.tmdb_id}`
    : null;

  document.getElementById('panel-fid').textContent =
    `${(d.title||'').substring(0,22).toUpperCase()} / ${String(panelCount).padStart(3,'0')}`;

  function brow(lbl, val, c) {
    return `<div class="bd-row">
      <span class="bd-lbl">${lbl}</span>
      <div class="bd-track"><div class="bd-fill" style="width:${val}%;background:${c};color:${c}"></div></div>
      <span class="bd-val" style="color:${c}">${val.toFixed(0)}</span>
    </div>`;
  }

  // Build shell with placeholder IDs — never interpolate user data into innerHTML
  document.getElementById('pi').innerHTML = `
    <div class="panel-poster-wrap">
      ${img ? `<img src="${img}" alt=""/>` : `<div class="panel-poster-ph">🎬</div>`}
    </div>
    <div class="panel-body">
      <div class="panel-plat" id="pp-plat" style="--pl:${platCol};color:${platCol}"></div>
      <div class="panel-title" id="pp-title"></div>
      <div class="panel-meta-row">
        <div class="pm-item">Type<span>${d.type==='tv'?'Series':'Film'}</span></div>
        <div class="pm-item">Year<span>${d.year||'—'}</span></div>
        <div class="pm-item">Reviews<span>${d.reviews}</span></div>
      </div>
      <div class="panel-score-section">
        <div class="panel-big-score" style="color:${col};text-shadow:0 0 80px ${colA}">${d.score.toFixed(0)}</div>
        <div class="panel-score-meta">
          <div class="panel-verdict-text">${verdict(d.score)}</div>
          <div class="panel-verdict-sub">StreamIQ Score</div>
          <div class="panel-tags">
            ${d.trending   ? `<span class="ptag" style="background:rgba(232,197,71,0.1);color:#E8C547;border:1px solid rgba(232,197,71,0.2)">↑ Trending</span>` : ''}
            ${d.polarizing ? `<span class="ptag" style="background:rgba(255,68,85,0.1);color:#FF4455;border:1px solid rgba(255,68,85,0.2)">⚡ Polarising</span>` : ''}
            ${d.hindi_dub  ? `<span class="ptag" style="background:rgba(255,112,67,0.1);color:#FF7043;border:1px solid rgba(255,112,67,0.2)">🎙 Hindi Dub</span>` : ''}
          </div>
        </div>
      </div>
      <div class="panel-overview" id="pp-overview"></div>
      <div class="breakdown">
        <div class="breakdown-hd">Score Breakdown</div>
        ${brow('YouTube',  d.yt,     '#FF4040')}
        ${brow('Reddit',   d.reddit, '#FF7A30')}
        ${brow('IMDb',     d.imdb,   '#E8C547')}
      </div>
      <div class="panel-actions">
        ${d.stream_url ? (() => {
          const u = d.stream_url;
          const cfg = {
            'netflix':    {bg:'#E50914', fg:'#fff'},
            'primevideo': {bg:'#00A8E1', fg:'#fff'},
            'hotstar':    {bg:'#1A1A2E', fg:'#00D4FF'},
            'jiocinema':  {bg:'#7B2FBE', fg:'#fff'},
            'apple':      {bg:'#f5f5f7', fg:'#000'},
          };
          const key = Object.keys(cfg).find(k => u.includes(k)) || '';
          const c = cfg[key] || {bg:'#E8C547', fg:'#07070A'};
          return `
            <button class="tv-send-btn" onclick="copyTitle('${d.title.replace(/'/g, "\\'")}', this)">
              <span class="tv-icon">📋</span>
              Copy Title
            </button>
            <a class="panel-watch-btn" href="${u}" target="_blank"
               onclick="tryDeepLink(event,this)" style="background:${c.bg};color:${c.fg}">&#9654; Watch on ${d.platform}</a>`;
        })() : ''}
        ${tmdb ? `<a class="panel-link" href="${tmdb}" target="_blank">View on TMDb →</a>` : ''}
      </div>
    </div>`;

  // Safe text — works even if title/overview contains quotes, apostrophes, HTML, etc.
  document.getElementById('pp-plat').textContent    = d.platform || '';
  document.getElementById('pp-title').textContent   = d.title    || '';
  document.getElementById('pp-overview').textContent = d.overview || '';

  const panelEl = document.getElementById('panel');
  const overlayEl = document.getElementById('overlay');
  // Scroll is on the PARENT window (Streamlit page), not the iframe
  // Use getBoundingClientRect on clicked card — works on all platforms
  // because it gives position relative to the VISIBLE viewport, not the document
  let offsetTop = 0;
  let viewH = window.innerHeight || 800;
  try { viewH = window.parent.innerHeight || viewH; } catch(e) {}
  if (clickedEl) {
    const rect = clickedEl.getBoundingClientRect();
    // rect.top is relative to iframe viewport
    // We need to convert to absolute position in the full iframe document
    offsetTop = rect.top + (window.scrollY || document.documentElement.scrollTop || 0);
    // Clamp so panel doesn't go below where content ends
    offsetTop = Math.max(0, offsetTop - 60);
  }
  panelEl.style.top = offsetTop + 'px';
  panelEl.style.height = viewH + 'px';
  overlayEl.style.top = offsetTop + 'px';
  overlayEl.style.height = viewH + 'px';
  panelEl.scrollTop = 0;
  panelEl.classList.add('on');
  overlayEl.classList.add('on');
}

function closeP() {
  const p = document.getElementById('panel');
  p.classList.remove('on');
  document.getElementById('overlay').classList.remove('on');
  p.style.top = '';
  p.style.height = '';
  document.getElementById('overlay').style.top = '';
  document.getElementById('overlay').style.height = '';
}
document.addEventListener('keydown', e => { if (e.key==='Escape') closeP(); });

/* ── FILTER HANDLERS ── */
function sP(v,b) { wPlat=v; document.querySelectorAll('[data-p]').forEach(x=>x.classList.toggle('on',x===b)); renderWatch(); }
function sT(v,b) { wType=v; document.querySelectorAll('[data-t]').forEach(x=>x.classList.toggle('on',x===b)); renderWatch(); }
function wSr(v)  { wSearch=v; renderWatch(); }
function wSo(v)  { wSort=v; renderWatch(); }

/* ── DISCOVER ── */
let dPlat = 'all';
let dType = 'all';

function getFD() {
  return D
    .filter(d => !window._dHindi || d.hindi_dub === true)
    .filter(d => {
      if (dCat === 'all') return true;
      if (dCat === 'genre_picks') return d.category && d.category.startsWith('genre_');
      return d.category === dCat;
    })
    .filter(d => dPlat==='all' || d.platform===dPlat)
    .filter(d => dType==='all' || d.type===dType)
    .filter(d => !dSearch || d.title.toLowerCase().includes(dSearch.toLowerCase()))
    .sort((a,b) => b.rating - a.rating);
}

function renderDisc() {
  const data = getFD();
  renderStats(D, 'ds', true);
  const el = document.getElementById('dg');
  if (!data.length) { el.innerHTML='<div class="empty">Nothing found</div>'; return; }
  el.innerHTML = data.slice(0,150).map((d,i) => {
    const img    = d.poster ? 'https://image.tmdb.org/t/p/w200'+d.poster : null;
    const catCol  = CC[d.category] || '#666';
    const catLbl  = CL[d.category] || d.category;
    const platCol = pc(d.platform);
    const tmdb    = d.tmdb_id ? `https://www.themoviedb.org/${d.type==='tv'?'tv':'movie'}/${d.tmdb_id}` : '#';
    const jwQuery = encodeURIComponent(d.title);
    const jwUrl   = `https://www.justwatch.com/in/search?q=${jwQuery}`;
    const delay  = (i%30)*0.016;
    return `
<div class="dc fu" style="animation-delay:${delay}s;--cc:${catCol}">
  <a href="${tmdb}" target="_blank" style="text-decoration:none;color:inherit;display:flex;flex-direction:column;flex:1">
  <div class="dc-poster">
    ${img ? `<img src="${img}" loading="lazy" alt="${d.title}"/>` : `<div class="dc-poster-ph">🎬</div>`}
  </div>
  <div class="dc-body">
    <div class="dc-title">${d.title}</div>
    <div class="dc-sub">${d.type==='tv'?'Series':'Film'} · ${d.year} · ${d.platform}</div>
    <div class="dc-foot">
      ${d.rating ? `<span class="dc-rating">✦ ${d.rating.toFixed(1)}</span>` : ''}
      <span class="dc-tag" style="color:${catCol};border-color:${catCol}">${catLbl}</span>
      ${d.hindi_dub ? '<span class="dc-tag" style="color:#FF7043;border-color:#FF7043">🎙 Hindi</span>' : ''}
    </div>
  </div>
  </a>
  <div style="padding:6px 12px 10px;border-top:1px solid var(--v3);display:flex;flex-direction:column;gap:8px">
    ${d.stream_url
      ? `<button class="tv-send-btn" style="min-width:auto;padding:10px 16px;font-size:13px" onclick="event.stopPropagation();copyTitle('${d.title.replace(/'/g, "\\'")}', this)">
           <span class="tv-icon">📋</span> Copy Title
         </button>
         <a href="${d.stream_url}" target="_blank" class="jw-link" style="color:${platCol}" onclick="tryDeepLink(event,this)">&#9654; Watch on ${d.platform}</a>`
      : `<a href="${jwUrl}" target="_blank" class="jw-link">&#9654; Find on JustWatch</a>`
    }
  </div>
</div>`;
  }).join('');
}

function sC(v,b) { dCat=v; document.querySelectorAll('[data-c]').forEach(x=>x.classList.toggle('on',x===b)); renderDisc(); }
function sdP(v,b) { dPlat=v; document.querySelectorAll('[data-dp]').forEach(x=>x.classList.toggle('on',x===b)); renderDisc(); }
function sdT(v,b) { dType=v; document.querySelectorAll('[data-dt]').forEach(x=>x.classList.toggle('on',x===b)); renderDisc(); }
function dSr(v)  { dSearch=v; renderDisc(); }

/* ── TAB SWITCH ── */
function showTab(t, btn) {
  document.getElementById('tab-watch').style.display  = t==='watch' ? '' : 'none';
  document.getElementById('tab-disc').style.display   = t==='disc'  ? '' : 'none';
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('on', b===btn));
  if (t==='disc')  renderDisc();
  }


/* ── INIT ── */
const now = new Date();
const ts = now.toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'})
         + ' · ' + now.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
document.getElementById('ts').textContent = ts;
document.getElementById('nav-ts').textContent = ts;

renderWatch();
renderDisc();


/* ═══════════════════════════════════════════════════
   LG TV INTEGRATION FUNCTIONS
═══════════════════════════════════════════════════ */

let toastTimeout;
function showToast(icon, message, duration = 3000) {
  let toast = document.getElementById('tv-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'tv-toast';
    toast.className = 'toast';
    toast.innerHTML = '<span class="toast-icon"></span><span class="toast-msg"></span>';
    document.body.appendChild(toast);
  }
  
  toast.querySelector('.toast-icon').textContent = icon;
  toast.querySelector('.toast-msg').textContent = message;
  toast.classList.add('show');
  
  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove('show');
  }, duration);
}

function copyTitle(title, btn) {
  var orig = btn.innerHTML;
  var doIt = function() {
    btn.innerHTML = '<span class="tv-icon">✓</span> Copied!';
    showToast('📋', '"' + title + '" copied — search on TV remote', 3000);
    setTimeout(function() { btn.innerHTML = orig; }, 2000);
  };
  if (navigator.clipboard) {
    navigator.clipboard.writeText(title).then(doIt).catch(function(){
      var ta = document.createElement('textarea');
      ta.value = title; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta); doIt();
    });
  } else {
    var ta = document.createElement('textarea');
    ta.value = title; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta); doIt();
  }
}
/* ── STAT CARD FILTER HELPERS ── */

/* ── HINDI DUB FILTER ── */
function toggleHindiW(btn) {
  window._wHindi = !window._wHindi;
  btn.classList.toggle('on', window._wHindi);
  renderWatch();
}
function toggleHindiD(btn) {
  window._dHindi = !window._dHindi;
  btn.classList.toggle('on', window._dHindi);
  renderDisc();
}

function wWorth() {
  wSort = 'score';
  window._wFilter = 'worth';
  showTab('watch', document.querySelector('.tab'));
  renderWatch();
}
function wPolar() {
  window._wFilter = 'polar';
  showTab('watch', document.querySelector('.tab'));
  renderWatch();
}

/* ── DEEP LINK: open in app on mobile, browser fallback ── */
function tryDeepLink(e, href, platform) {
  if (!href) return;
  var isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  if (!isMobile) return; // desktop: let default href work normally
  e.preventDefault();
  var appUrl = href;
  if (href.includes('netflix.com')) {
    // Extract Netflix title ID and use nflx:// scheme
    var m = href.match(/netflix\.com\/(?:title|watch)\/([\d]+)/);
    if (m) appUrl = 'nflx://www.netflix.com/title/' + m[1];
    else appUrl = 'nflx://www.netflix.com/';
  } else if (href.includes('primevideo.com')) {
    appUrl = href.replace('https://www.primevideo.com', 'aiv://aiv');
  } else if (href.includes('hotstar.com') || href.includes('jiohotstar.com')) {
    appUrl = href.replace(/https:\/\/(www\.)?jiohotstar\.com/, 'hotstar://').replace(/https:\/\/(www\.)?hotstar\.com/, 'hotstar://');
  }
  // Try app scheme, fall back to browser after 1.5s
  var fallbackTimer = setTimeout(function() {
    window.open(href, '_blank');
  }, 1500);
  window.addEventListener('blur', function onBlur() {
    clearTimeout(fallbackTimer);
    window.removeEventListener('blur', onBlur);
  });
  window.location.href = appUrl;
}

/* ── KEEP-ALIVE: ping every 10 min to prevent Streamlit sleep ── */
(function keepAlive() {
  setInterval(function() {
    fetch(window.location.href, { method: 'HEAD', cache: 'no-store' }).catch(function(){});
  }, 10 * 60 * 1000);
})();

</script>
</body>
</html>"""

HTML = HTML.replace('__WJ__', WJ).replace('__DJ__', DJ)
components.html(HTML, height=4400, scrolling=True)
