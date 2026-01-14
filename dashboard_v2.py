# dashboard_v2.py - Streamlit Dashboard for V2.5 (Complete)
# Run with: streamlit run dashboard_v2.py

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Streaming Tracker V2.5",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .polarizing-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .classic-badge {
        background-color: #ffd700;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .genre-badge {
        background-color: #9333ea;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from Supabase"""
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        return None, None, None, "No Supabase credentials"
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Load scores with content info
        scores_result = supabase.table('scores') \
            .select('*, content(*)') \
            .order('final_score', desc=True) \
            .execute()
        
        if not scores_result.data:
            return None, None, None, "No data found. Run streaming_tracker_v25.py first"
        
        # Parse nested content
        scores_data = []
        for row in scores_result.data:
            content = row.pop('content', {})
            scores_data.append({
                **row,
                'title': content.get('title', 'Unknown'),
                'platform': content.get('platform', 'Unknown'),
                'content_type': content.get('content_type', 'unknown'),
                'release_year': content.get('release_year'),
                'tmdb_id': content.get('tmdb_id'),
                'poster_path': content.get('poster_path'),
                'overview': content.get('overview'),
                'imdb_rating': content.get('imdb_rating'),
                'discovery_source': content.get('discovery_source', 'unknown')
            })
        
        scores_df = pd.DataFrame(scores_data)
        
        # Load reviews
        reviews_result = supabase.table('reviews') \
            .select('*, content(title, platform)') \
            .order('created_at', desc=True) \
            .execute()
        
        reviews_data = []
        for row in reviews_result.data:
            content = row.pop('content', {})
            reviews_data.append({
                **row,
                'content_title': content.get('title', 'Unknown'),
                'content_platform': content.get('platform', 'Unknown')
            })
        
        reviews_df = pd.DataFrame(reviews_data)
        
        # Load content
        content_result = supabase.table('content').select('*').execute()
        content_df = pd.DataFrame(content_result.data)
        
        return scores_df, reviews_df, content_df, None
        
    except Exception as e:
        return None, None, None, f"Database error: {str(e)}"


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🎬 Streaming Tracker V2.5</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Source Discovery • AI Sentiment • Real-Time Verification</p>', unsafe_allow_html=True)
    
    # Load data
    scores_df, reviews_df, content_df, error = load_data()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("""
        **To fix this:**
        1. Make sure `.env` file has SUPABASE_URL and SUPABASE_KEY
        2. Run: `python3 streaming_tracker_v25.py`
        3. Wait for it to complete
        4. Refresh this page
        """)
        return
    
    # Success message with breakdown
    total_content = len(scores_df)
    total_reviews = len(reviews_df) if reviews_df is not None else 0
    
    # Count by category
    category_counts = {}
    if 'category' in scores_df.columns:
        for cat in scores_df['category'].unique():
            if cat == 'trending':
                category_counts['🔥 Trending'] = len(scores_df[scores_df['category'] == 'trending'])
            elif cat == 'underdog':
                category_counts['💎 Hidden Gems'] = len(scores_df[scores_df['category'] == 'underdog'])
            elif cat == 'classics':
                category_counts['⭐ Classics'] = len(scores_df[scores_df['category'] == 'classics'])
            elif cat.startswith('genre_'):
                genre = cat.replace('genre_', '').title()
                category_counts[f'🎭 {genre}'] = len(scores_df[scores_df['category'] == cat])
    
    breakdown = " • ".join([f"{k}: {v}" for k, v in category_counts.items()])
    st.success(f"✅ Loaded {total_content} titles • {total_reviews} reviews")
    if breakdown:
        st.info(f"📊 {breakdown}")
    
    # ========================================================================
    # SIDEBAR FILTERS
    # ========================================================================
    
    st.sidebar.header("🔍 Filters")
    
    # View Mode - ENHANCED with all discovery sources
    view_mode = st.sidebar.radio(
        "📊 View Mode",
        [
            "📺 All Content",
            "🔥 Trending (Recent)", 
            "💎 Hidden Gems",
            "⭐ Classics (IMDB 8.0+)",
            "🎭 By Genre"
        ],
        help="Filter content by discovery source"
    )
    
    # Platform filter
    platforms = ['All'] + sorted(scores_df['platform'].unique().tolist())
    selected_platform = st.sidebar.selectbox("Platform", platforms)
    
    # Content type filter
    content_types = ['All'] + sorted(scores_df['content_type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Type", content_types)
    
    # Score range
    min_score = st.sidebar.slider("Minimum Score", 0, 100, 0)
    
    # Show polarizing only
    show_polarizing_only = st.sidebar.checkbox("🧨 Polarizing Content Only")
    
    # Apply filters
    filtered_df = scores_df.copy()
    
    # Apply view mode filter
    if view_mode == "🔥 Trending (Recent)":
        if 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'] == 'trending']
        else:
            # Fallback: show content from 2024-2026
            filtered_df = filtered_df[
                (filtered_df['release_year'] >= 2024) | 
                (filtered_df['release_year'].isna())
            ]
    
    elif view_mode == "💎 Hidden Gems":
        if 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'] == 'underdog']
        else:
            st.warning("No hidden gems data available. Run the tracker to discover underdogs.")
    
    elif view_mode == "⭐ Classics (IMDB 8.0+)":
        if 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'] == 'classics']
        else:
            # Fallback to IMDB rating
            filtered_df = filtered_df[
                (filtered_df['imdb_rating'].notna()) & 
                (filtered_df['imdb_rating'] >= 8.0)
            ]
    
    elif view_mode == "🎭 By Genre":
        if 'category' in filtered_df.columns:
            genre_categories = [cat for cat in filtered_df['category'].unique() if cat.startswith('genre_')]
            if genre_categories:
                # Show genre selector
                genres = sorted([cat.replace('genre_', '').title() for cat in genre_categories])
                selected_genre = st.sidebar.selectbox("Select Genre", genres)
                filtered_df = filtered_df[filtered_df['category'] == f'genre_{selected_genre.lower()}']
            else:
                st.warning("No genre data available. Run the tracker to discover by genre.")
    
    # "All Content" = no filter (DEFAULT)
    
    if selected_platform != 'All':
        filtered_df = filtered_df[filtered_df['platform'] == selected_platform]
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['content_type'] == selected_type]
    
    filtered_df = filtered_df[filtered_df['final_score'] >= min_score]
    
    if show_polarizing_only:
        filtered_df = filtered_df[filtered_df['is_polarizing'] == True]
    
    # Sort by score
    filtered_df = filtered_df.sort_values('final_score', ascending=False)
    
    # ========================================================================
    # KEY METRICS
    # ========================================================================
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Content",
            len(filtered_df),
            f"{len(filtered_df) - len(scores_df)}" if len(filtered_df) != len(scores_df) else None
        )
    
    with col2:
        avg_score = filtered_df['final_score'].mean() if not filtered_df.empty else 0
        st.metric("Avg Score", f"{avg_score:.1f}")
    
    with col3:
        must_watch = len(filtered_df[filtered_df['final_score'] >= 80])
        st.metric("🔥 Must Watch", must_watch)
    
    with col4:
        polarizing = len(filtered_df[filtered_df['is_polarizing'] == True])
        st.metric("🧨 Polarizing", polarizing)
    
    st.divider()
    
    # ========================================================================
    # TOP RANKED CONTENT
    # ========================================================================
    
    st.header("🏆 Top Ranked Content")
    
    if filtered_df.empty:
        st.warning("No content matches the current filters")
        return
    
    # Display table
    display_df = filtered_df.head(50).copy()
    
    # Format columns
    display_df['Score'] = display_df['final_score'].apply(lambda x: f"{x:.1f}")
    display_df['Reviews'] = display_df['review_count']
    display_df['Label'] = display_df['label']
    display_df['Platform'] = display_df['platform']
    display_df['Type'] = display_df['content_type'].apply(lambda x: '📺 TV' if x == 'tv' else '🎬 Movie')
    
    # Add year column
    display_df['Year'] = display_df['release_year'].apply(
        lambda x: str(int(x)) if pd.notna(x) else 'N/A'
    )
    
    # Add category badge
    def get_category_badge(row):
        cat = row.get('category', '')
        if cat == 'classics':
            return '⭐ Classic'
        elif cat == 'underdog':
            return '💎 Gem'
        elif cat == 'trending':
            return '🔥 Trending'
        elif cat.startswith('genre_'):
            genre = cat.replace('genre_', '').title()
            return f'🎭 {genre}'
        return ''
    
    display_df['Category'] = display_df.apply(get_category_badge, axis=1)
    
    # Add status indicators
    display_df['Status'] = display_df.apply(
        lambda row: '🧨 Polarizing' if row['is_polarizing'] 
                   else '⚠️ Low Reviews' if row['review_count'] < 3
                   else '✓', 
        axis=1
    )
    
    # Select columns to show
    table_cols = ['title', 'Platform', 'Type', 'Category', 'Year', 'Score', 'Label', 'Reviews', 'Status']
    
    st.dataframe(
        display_df[table_cols],
        column_config={
            "title": st.column_config.TextColumn("Title", width="large"),
            "Platform": st.column_config.TextColumn("Platform", width="medium"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Category": st.column_config.TextColumn("Source", width="medium"),
            "Year": st.column_config.TextColumn("Year", width="small"),
            "Score": st.column_config.TextColumn("Score", width="small"),
            "Label": st.column_config.TextColumn("Label", width="medium"),
            "Reviews": st.column_config.NumberColumn("Reviews", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small")
        },
        height=600,
        width="stretch",
        hide_index=True
    )
    
    # ========================================================================
    # DETAILED VIEW - EXPANDABLE CARDS
    # ========================================================================
    
    st.subheader("📋 Detailed Rankings")
    
    for idx, row in display_df.head(20).iterrows():
        # Get category emoji
        cat = row.get('category', '')
        if cat == 'classics':
            cat_emoji = '⭐'
        elif cat == 'underdog':
            cat_emoji = '💎'
        elif cat == 'trending':
            cat_emoji = '🔥'
        elif cat.startswith('genre_'):
            cat_emoji = '🎭'
        else:
            cat_emoji = ''
        
        polarizing_emoji = '🧨 ' if row['is_polarizing'] else ''
        
        with st.expander(f"{cat_emoji} {polarizing_emoji}{row['title']} - {row['label']}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Poster
                if row.get('poster_path'):
                    poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}"
                    st.image(poster_url, width=150)
                
                # TMDb link
                tmdb_id = row.get('tmdb_id')
                content_type = row.get('content_type')
                if tmdb_id and content_type:
                    tmdb_url = f"https://www.themoviedb.org/{'tv' if content_type == 'tv' else 'movie'}/{tmdb_id}"
                    st.markdown(f"[View on TMDb]({tmdb_url})")
                
                # Show category badge
                if row.get('Category'):
                    st.markdown(f"**Source:** {row['Category']}")
            
            with col2:
                # Overview
                if row.get('overview'):
                    st.write("**Overview:**")
                    st.write(row['overview'][:300] + "..." if len(row['overview']) > 300 else row['overview'])
                
                # Score breakdown
                st.write("**Score Breakdown:**")
                
                score_data = {
                    'Source': ['YouTube', 'Reddit', 'IMDB'],
                    'Score': [
                        row.get('youtube_score', 0),
                        row.get('reddit_score', 0),
                        row.get('imdb_score', 0)
                    ]
                }
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=score_data['Source'],
                        y=score_data['Score'],
                        marker_color=['#FF6B6B', '#FF8C42', '#4ECDC4'],
                        text=[f"{s:.1f}" for s in score_data['Score']],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis_range=[0, 100],
                    yaxis_title="Score",
                    showlegend=False
                )
                
                st.plotly_chart(fig, width="stretch", key=f"chart_{idx}")
                
                # Stats
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Reviews", row['review_count'])
                with col_b:
                    st.metric("Positive %", f"{row['positive_ratio']*100:.0f}%")
                with col_c:
                    if row['is_polarizing']:
                        st.markdown('<span class="polarizing-badge">POLARIZING</span>', unsafe_allow_html=True)
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    st.divider()
    st.header("📊 Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        
        fig = px.histogram(
            filtered_df,
            x='final_score',
            nbins=20,
            labels={'final_score': 'Score', 'count': 'Count'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("Platform Comparison")
        
        platform_avg = filtered_df.groupby('platform')['final_score'].mean().sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=platform_avg.index,
                y=platform_avg.values,
                marker_color='#764ba2',
                text=[f"{v:.1f}" for v in platform_avg.values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_range=[0, 100],
            yaxis_title="Avg Score"
        )
        
        st.plotly_chart(fig, width="stretch")
    
    # Category distribution (NEW)
    st.subheader("Content by Discovery Source")
    
    if 'category' in filtered_df.columns:
        category_counts = filtered_df['category'].value_counts()
        
        # Map categories to readable names
        category_names = {
            'trending': '🔥 Trending',
            'underdog': '💎 Hidden Gems',
            'classics': '⭐ Classics'
        }
        
        for cat in category_counts.index:
            if cat.startswith('genre_'):
                genre = cat.replace('genre_', '').title()
                category_names[cat] = f'🎭 {genre}'
        
        readable_names = [category_names.get(cat, cat) for cat in category_counts.index]
        
        fig = px.pie(
            values=category_counts.values,
            names=readable_names,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
    
    # ========================================================================
    # RECENT REVIEWS
    # ========================================================================
    
    st.divider()
    st.header("📝 Recent Reviews")
    
    if reviews_df is not None and not reviews_df.empty:
        recent_reviews = reviews_df.head(10)
        
        for idx, review in recent_reviews.iterrows():
            sentiment_emoji = {
                1: '✅ Positive',
                0: '➖ Neutral',
                -1: '❌ Negative'
            }.get(review.get('sentiment'), '❓ Unknown')
            
            source = review.get('source', 'unknown')
            source_emoji = '📺' if source == 'youtube' else '💬' if source == 'reddit' else '📝'
            
            with st.expander(f"{source_emoji} {sentiment_emoji} - {review.get('content_title', 'Unknown')} ({review.get('content_platform', 'Unknown')})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Reviewer:** {review.get('reviewer', 'Anonymous')}")
                    st.write(f"**Source:** {source.title()}")
                    
                    if review.get('reviewer_subscribers', 0) > 0:
                        subs = review['reviewer_subscribers']
                        if subs >= 1_000_000:
                            st.write(f"👥 {subs/1_000_000:.1f}M subscribers")
                        elif subs >= 1_000:
                            st.write(f"👥 {subs/1_000:.1f}K subscribers")
                        else:
                            st.write(f"👥 {subs} subscribers")
                    
                    if review.get('review_text'):
                        text = review['review_text'][:200]
                        st.write(f"**Review:** {text}...")
                    
                    if review.get('source_url'):
                        st.markdown(f"[View Source]({review['source_url']})")
                
                with col2:
                    st.metric("Confidence", f"{review.get('confidence', 0):.0%}")
                    if review.get('views', 0) > 0:
                        st.metric("Views", f"{review.get('views', 0):,}")
                    
                    weight = review.get('youtube_weight', 0)
                    if weight > 0:
                        st.metric("Authority", f"{weight:.2f}")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col3:
        st.caption(f"💡 Total: {len(scores_df)} titles")
    
    st.caption("🚀 To update data, run: `python3 streaming_tracker_v25.py`")


if __name__ == "__main__":
    main()
