# app.py
import streamlit as st
from newsapi import NewsApiClient
from transformers import pipeline
import yfinance as yf
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import time
import requests
import altair as alt

# Load API keys from .env
load_dotenv()

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

# Replace the sentiment pipeline initialization with:
sentiment_pipeline = pipeline(
    "text-classification",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)

# ------------------------
# Core Functions (with Caching)
# ------------------------

@st.cache_data(ttl=10800)
def get_company_name(ticker):
    """Get company name with retries and custom headers"""
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0'})
            stock = yf.Ticker(ticker, session=session)
            company_name = stock.info.get('longName')
            if company_name:
                return company_name
            raise ValueError("Company name not found in yfinance data")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"Failed after {max_retries} attempts: {str(e)}")
            raise

@st.cache_data(ttl=1800)
def fetch_esg_news(ticker="TSLA"):
    """Fetch news specific to the selected company's ESG factors"""
    company_name = get_company_name(ticker)
    query = f"({company_name} OR {ticker}) AND (ESG OR sustainability OR governance)"

    from_date = datetime.now() - timedelta(days=20)
    from_param = from_date.strftime("%Y-%m-%d") # Format date as YYYY-MM-DD

    articles = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=10,
        from_param=from_param # Use formatted date
    )
    return articles["articles"]

@st.cache_data(ttl=1800)
def fetch_stock_data(ticker):
    """Fetch stock data with retries and custom headers"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0'})
            stock = yf.Ticker(ticker, session=session)
            data = stock.history(period="3mo")
            return data.reset_index()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"Failed to fetch stock data: {str(e)}")
            raise

def analyze_news_sentiment(articles):
    """Analyze sentiment with robust date parsing"""
    results = []
    for article in articles:
        try:
            # Extract text content
            text = f"{article.get('title', '')}. {article.get('description', '')}"[:512]
            
            # Handle date parsing with multiple fallbacks
            pub_date = article.get('publishedAt')
            if not pub_date:
                continue  # Skip articles without dates
                
            try:
                # Try ISO format parser first
                date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try different format variants
                    if len(pub_date) >= 10:
                        date = datetime.strptime(pub_date[:10], "%Y-%m-%d")
                    else:
                        continue
                except:
                    continue

            # Process sentiment scoring
            result = sentiment_pipeline(text)[0]
            score = result["score"] if result["label"].lower() in ["positive", "label_1"] else -result["score"]
            
            results.append({
                "date": date,
                "text": text,
                "sentiment": result["label"].capitalize(),
                "score": score
            })
        except Exception as e:
            st.warning(f"Skipped article due to error: {str(e)}")
            continue
            
    return pd.DataFrame(results)

def prepare_training_data(sentiment_df, stock_df):
    """Align sentiment and price data with timezone normalization"""
    # Convert all dates to UTC
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_convert('UTC')
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_convert('UTC')
    
    merged = pd.merge_asof(
        stock_df.sort_values('Date'),
        sentiment_df.sort_values('date'),
        left_on='Date',
        right_on='date',
        direction='forward'
    )
    
    # Create features (keep existing code)
    merged['price_change'] = merged['Close'].pct_change() * 100
    merged['sentiment_ma'] = merged['score'].rolling(3).mean()
    merged['news_count'] = merged.groupby('date')['score'].transform('count')
    merged['target'] = merged['price_change'].shift(-1)
    merged = merged.dropna(subset=['score', 'sentiment_ma', 'news_count', 'target'])
    merged = merged[merged['news_count'] > 0]  # Filter days with news
    
    return merged.dropna()

# ------------------------
# Streamlit UI - Redesigned
# ------------------------

st.set_page_config(
    layout="wide",
    page_title="ESG Impact Analyzer",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "ESG News Impact Analyzer for Stock Market Analysis"
    }
)

# Dark mode custom CSS - Simplified and fixed
st.markdown("""
<style>
    /* Base Theme */
    [data-testid="stAppViewContainer"] {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }
    
    /* Main content spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #333;
    }
    
    /* Input elements */
    [data-testid="stTextInput"] > div > div {
        background-color: #333;
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif;
        margin-bottom: 1rem;
    }
    
    /* Cards and containers */
    .card {
        background-color: #222;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #333;
    }
    
    /* Metric styling */
    .metric-card {
        background-color: #232323;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #333;
        height: 100%;
    }
    
    /* Charts */
    .chart-container {
        background-color: #232323;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    
    /* News headline cards */
    .news-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #555;
    }
    
    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: #333;
        border-radius: 6px 6px 0 0;
        border: 1px solid #444;
        color: #bbb;
    }
    
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background-color: #424242;
        color: #fff;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] th {
        background-color: #333;
        color: #e0e0e0;
    }
    
    /* Messages */
    .info-box {
        background-color: #152238;
        color: #99ccff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3366cc;
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background-color: #332b10;
        color: #ffd699;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #cc9933;
        margin-bottom: 1rem;
    }
    
    .error-box {
        background-color: #391515;
        color: #ff9999;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #cc3333;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("""
<div style="border-bottom: 1px solid #444; margin-bottom: 1.5rem; padding-bottom: 0.5rem;">
    <h1 style="color: #e0e0e0; font-weight: 600; margin-bottom: 0.5rem;">ESG News Impact Analyzer 🌱📈</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls - Simplified
with st.sidebar:
    st.markdown("## Configuration ⚙️")
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    
    ticker_input = st.text_input(
        "**Stock Ticker**",
        value="TSLA",
        help="Enter a valid stock ticker symbol",
        placeholder="e.g. TSLA",
    ).upper()
    
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    
    forecast_days = st.slider(
        "**Forecast Horizon**",
        1, 5, 3,
        help="Number of days to predict price movement"
    )
    
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    
    model_type = st.selectbox(
        "**Analysis Model**",
        ["Random Forest", "Linear Regression"],
        index=0,
        help="Select machine learning model type"
    )

# Main Content
try:
    # Header with Company Info - Simplified
    with st.spinner("🔍 Fetching company data..."):
        company_name = get_company_name(ticker_input)
        stock_data = fetch_stock_data(ticker_input)
    
    st.markdown(f"""
    <div class="card">
        <h3 style="margin: 0; font-weight: 500;">{company_name} Analysis Dashboard</h3>
        <p style="margin: 0.5rem 0 0 0; color: #999; font-size: 0.9rem;">Ticker: {ticker_input} | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Stock Performance Section
    st.markdown("### 📊 Stock Performance Trend")
    
    with st.container():
        tab1, tab2 = st.tabs(["3-Month Trend", "Raw Data"])
        
        with tab1:
            # Custom chart configuration
            stock_chart = alt.Chart(stock_data).mark_line(color='#4dabf7').encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Close:Q', title='Price ($)'),
                tooltip=['Date:T', 'Close:Q', 'Volume:Q']
            ).properties(
                height=350,
                background='#232323'
            ).configure_axis(
                labelColor='#e0e0e0',
                titleColor='#e0e0e0',
                grid=True,
                gridColor='#333'
            ).configure_view(
                strokeWidth=0
            )
            
            st.altair_chart(stock_chart, use_container_width=True)
            
        with tab2:
            st.dataframe(
                stock_data[['Date', 'Close', 'Volume']].style.format({
                    'Close': '${:.2f}',
                    'Volume': '{:,.0f}'
                }),
                hide_index=True,
                use_container_width=True
            )

    # AI Impact Predictions - Horizontal Layout
    st.markdown("### 🤖 AI Impact Predictions")
    
    try:
        articles = fetch_esg_news(ticker_input)
        sentiment_df = analyze_news_sentiment(articles) if articles else None
        
        if sentiment_df is not None and len(sentiment_df) > 5:
            merged_data = prepare_training_data(sentiment_df, stock_data)
            
            # Model Training
            with st.spinner("🧠 Training prediction model..."):
                X = merged_data[['score', 'sentiment_ma', 'news_count']]
                y = merged_data['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                model = RandomForestRegressor(n_estimators=100) if model_type == "Random Forest" else LinearRegression()
                model.fit(X_train, y_train)
                prediction = model.predict(X.tail(1).values)[0]
                mae = mean_absolute_error(y_test, model.predict(X_test))
            
            # Prediction Metrics - Horizontal layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #bbb; font-size: 0.9rem; margin-bottom: 0.3rem;">📉 Predicted Change</div>
                    <div style="font-size: 1.8rem; font-weight: 600; color: {'#4dabf7' if prediction > 0 else '#ff6b6b'}; margin-bottom: 0.3rem;">{prediction:.2f}%</div>
                    <div style="color: #777; font-size: 0.8rem;">{forecast_days}-day forecast</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #bbb; font-size: 0.9rem; margin-bottom: 0.3rem;">🎯 Model Accuracy</div>
                    <div style="font-size: 1.8rem; font-weight: 600; color: #aaa; margin-bottom: 0.3rem;">{mae:.2f}% MAE</div>
                    <div style="color: #777; font-size: 0.8rem;">Test set performance</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #bbb; font-size: 0.9rem; margin-bottom: 0.3rem;">🌡️ Sentiment Score</div>
                    <div style="font-size: 1.8rem; font-weight: 600; color: {'#66bb6a' if sentiment_df['score'].mean() > 0 else '#ef5350'}; margin-bottom: 0.3rem;">{sentiment_df['score'].mean():.2f}</div>
                    <div style="color: #777; font-size: 0.8rem;">3-day average</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Model Insights - Refined
            st.markdown("#### 🔍 Model Insights")
            
            if model_type == "Random Forest":
                try:
                    importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Custom chart for dark theme
                    importance_chart = alt.Chart(importance).mark_bar(color='#4dabf7').encode(
                        x=alt.X('Importance:Q', title='Importance'),
                        y=alt.Y('Feature:N', sort='-x', title='Feature'),
                        tooltip=['Feature', 'Importance']
                    ).properties(
                        height=200,
                        background='#232323'
                    ).configure_axis(
                        labelColor='#e0e0e0',
                        titleColor='#e0e0e0',
                        grid=True,
                        gridColor='#333'
                    ).configure_view(
                        strokeWidth=0
                    )
                    
                    st.altair_chart(importance_chart, use_container_width=True)
                    
                except AttributeError:
                    st.warning("Feature importance unavailable")
                    
            elif model_type == "Linear Regression":
                try:
                    coefficients = pd.DataFrame({
                        'Feature': X.columns,
                        'Coefficient': model.coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    # Custom chart for dark theme
                    chart = alt.Chart(coefficients).mark_bar().encode(
                        x=alt.X('Coefficient:Q', title="Impact Strength"),
                        y=alt.Y('Feature:N', sort='-x', title="Predictive Factors"),
                        color=alt.condition(
                            alt.datum.Coefficient > 0,
                            alt.value("#66bb6a"),  # Positive = green
                            alt.value("#ef5350")   # Negative = red
                        ),
                        tooltip=['Feature', 'Coefficient']
                    ).properties(
                        height=200,
                        title="Feature Impact on Stock Price",
                        background='#232323'
                    ).configure_axis(
                        labelColor='#e0e0e0',
                        titleColor='#e0e0e0',
                        grid=True,
                        gridColor='#333'
                    ).configure_view(
                        strokeWidth=0
                    ).configure_title(
                        color='#e0e0e0'
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                except AttributeError:
                    st.warning("Coefficients not available for current model configuration")
        else:
            st.markdown("""
            <div class="warning-box">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 1.5rem; margin-right: 1rem;">⚠️</div>
                    <div>Insufficient data for modeling - Need at least 5 news articles</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not generate predictions: {str(e)}")

    # ESG News Analysis - Full Width Dedicated Section
    st.markdown("### 🌍 ESG News Monitor")
    
    articles = fetch_esg_news(ticker_input) if 'articles' not in locals() else articles
    sentiment_df = analyze_news_sentiment(articles) if articles and 'sentiment_df' not in locals() else sentiment_df
    
    if not articles:
        st.markdown("""
        <div class="info-box">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 1.5rem; margin-right: 1rem;">ℹ️</div>
                <div>No recent ESG news found for this ticker</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Two columns for sentiment timeline and news headlines
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📈 Sentiment Timeline")
            sentiment_chart = sentiment_df.resample('D', on='date').mean(numeric_only=True)
            
            # Custom chart for dark theme
            timeline_chart = alt.Chart(sentiment_chart.reset_index()).mark_area(
                opacity=0.6,
                color='#4dabf7'
            ).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('score:Q', title='Sentiment Score'),
                tooltip=['date:T', 'score:Q']
            ).properties(
                height=250,
                background='#232323'
            ).configure_axis(
                labelColor='#e0e0e0',
                titleColor='#e0e0e0'
            )
            
            st.altair_chart(timeline_chart, use_container_width=True)
        
        with col2:
            st.markdown("#### 📰 Latest Headlines")
            
            # Display a scrollable container for headlines
            if len(sentiment_df) > 0:
                for _, row in sentiment_df.tail(5).iterrows():
                    sentiment_color = "#66bb6a" if row['score'] > 0 else "#ef5350"
                    emoji = "✅" if row['score'] > 0 else "⚠️"
                    
                    st.markdown(f"""
                    <div style="background-color: #1c1c1c; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.7rem; border-left: 4px solid {sentiment_color};">
                        <div style="display: flex; align-items: flex-start;">
                            <div style="font-size: 1.2rem; margin-right: 0.8rem;">{emoji}</div>
                            <div style="flex-grow: 1;">
                                <div style="color: #999; font-size: 0.8rem; margin-bottom: 0.3rem;">{row['date'].strftime('%b %d, %Y')}</div>
                                <div style="font-weight: 500; margin-bottom: 0.5rem;">{row['text'][:80]}...</div>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <div style="background-color: #333; height: 8px; border-radius: 4px; flex-grow: 1; margin-right: 10px;">
                                        <div style="background-color: {sentiment_color}; width: {abs(row['score']*100)}%; height: 8px; border-radius: 4px;"></div>
                                    </div>
                                    <div style="color: #bbb; font-size: 0.75rem; white-space: nowrap;">
                                        {abs(row['score']*100):.1f}%
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    No headlines available
                </div>
                """, unsafe_allow_html=True)

    # Additional Features Section - Simplified
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"🚨 System Error: {str(e)}")
    st.markdown("""
    <div class="error-box">
        <div style="margin-bottom: 0.8rem; font-weight: 500;">💡 Troubleshooting Tips:</div>
        <ul style="margin-left: 1.5rem; padding-left: 0;">
            <li style="margin-bottom: 0.3rem;">Verify internet connection</li>
            <li style="margin-bottom: 0.3rem;">Check ticker symbol validity</li>
            <li>Wait 1 minute before retrying</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ------------------------
# Additional Features
# ------------------------
with st.expander("Methodology Details"):
    st.markdown("""
    **Model Details:**
    - Uses news sentiment scores with 3-day moving averages
    - Predicts next-day price changes using:
      - Random Forest Regressor (default)
      - Linear Regression (comparison)
    - Features include: sentiment score, sentiment moving average, news volume
    """)

with st.expander("Latest Raw Data"):
    if 'sentiment_df' in locals():
        st.dataframe(sentiment_df[['date', 'text', 'score']])
    if 'stock_data' in locals():
        st.dataframe(stock_data[['Date', 'Close', 'Volume']])