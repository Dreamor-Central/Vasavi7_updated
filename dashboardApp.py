import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from plotly.subplots import make_subplots
import uuid

# Database connection configuration
@st.cache_resource
def init_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root", 
        password="password",
        database="chat_history"
    )

# Cache data queries
@st.cache_data(ttl=300)
def get_data():
    conn = init_connection()
    query = """
    SELECT 
        id, timestamp, session_id, user_message, response, 
        intent, products, categories, sentiment, keywords,
        returns_count, exchanges_count
    FROM chat_log
    ORDER BY timestamp DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def main():
    st.set_page_config(
        page_title="Customer Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Aesthetic CSS with refined color palette and modern design
    st.markdown("""
    <style>
    body {
        background-color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .metric-card {
        background: linear-gradient(135deg, #FF4081 0%, #FF6EC7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 0.5rem;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .metric-card h4 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-card p {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .section-header {
        background: linear-gradient(90deg, #00BCD4 0%, #26C6DA 100%);
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        color: #FFFFFF;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .section-header h2 {
        font-size: 1.8rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #E1F5FE;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #B3E5FC;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #01579B;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #81D4FA;
        color: #01579B;
    }
    .control-panel {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F4F8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .control-panel h3 {
        font-size: 1.5rem;
        color: #2D3748;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF5722 0%, #FF7043 100%);
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 500;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stSelectbox, .stSlider, .stDateInput {
        background: #F7FAFC;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>🛍️ Customer Analytics Dashboard</h1><p>Elegant Insights into Customer Behavior and Trends</p></div>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = get_data()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Control Panel for Filters
        with st.container():
            st.markdown('<div class="control-panel"><h3>🎛️ Control Panel</h3></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                date_range = st.date_input(
                    "📅 Date Range",
                    value=(datetime.now() - timedelta(days=30), datetime.now()),
                    key="date_range"
                )
            
            with col2:
                intents = df['intent'].dropna().unique()
                selected_intents = st.multiselect("🎯 Intents", intents, default=intents)
            
            with col3:
                sentiment_range = st.slider("😊 Sentiment Range", 0.0, 10.0, (0.0, 10.0), step=0.1)
            
            # Apply filters
            mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
            filtered_df = df.loc[mask]
            filtered_df = filtered_df[filtered_df['intent'].isin(selected_intents)]
            filtered_df = filtered_df[
                (filtered_df['sentiment'] >= sentiment_range[0]) & 
                (filtered_df['sentiment'] <= sentiment_range[1])
            ]
            
            st.markdown(f"<div style='text-align: center; color: #2D3748; font-weight: 500;'>**📊 Records:** {len(filtered_df):,} | **👥 Unique Sessions:** {filtered_df['session_id'].nunique():,}</div>", unsafe_allow_html=True)
        
        # === 1. KEY PERFORMANCE INDICATORS ===
        st.markdown('<div class="section-header"><h2>📈 Key Metrics</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h4>💬 Conversations</h4><p>{len(filtered_df):,}</p></div>', unsafe_allow_html=True)
        
        with col2:
            unique_sessions = filtered_df['session_id'].nunique()
            st.markdown(f'<div class="metric-card"><h4>👥 Sessions</h4><p>{unique_sessions:,}</p></div>', unsafe_allow_html=True)
        
        with col3:
            avg_messages = len(filtered_df) / unique_sessions if unique_sessions > 0 else 0
            st.markdown(f'<div class="metric-card"><h4>⏱️ Avg Session</h4><p>{avg_messages:.1f} msgs</p></div>', unsafe_allow_html=True)
        
        with col4:
            avg_sentiment = filtered_df['sentiment'].mean()
            satisfaction = (avg_sentiment / 10) * 100 if not pd.isna(avg_sentiment) else 0
            st.markdown(f'<div class="metric-card"><h4>😊 Satisfaction</h4><p>{satisfaction:.1f}%</p></div>', unsafe_allow_html=True)
        
        with col5:
            total_returns = filtered_df['returns_count'].sum()
            return_rate = (total_returns / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            st.markdown(f'<div class="metric-card"><h4>📦 Return Rate</h4><p>{return_rate:.1f}%</p></div>', unsafe_allow_html=True)
        
        # === 2. CONVERSATION ANALYTICS ===
        st.markdown('<div class="section-header"><h2>📊 Conversation Insights</h2></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📈 Sessions", "📋 Volume", "⏰ Patterns"])
        
        with tab1:
            daily_sessions = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg({
                'session_id': 'nunique',
                'id': 'count'
            }).reset_index()
            daily_sessions.columns = ['date', 'unique_sessions', 'total_conversations']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=daily_sessions['date'], y=daily_sessions['unique_sessions'],
                          name="Sessions", line=dict(color='#FF1744', width=4),
                          fill='tonexty', fillcolor='rgba(255, 23, 68, 0.2)'),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=daily_sessions['date'], y=daily_sessions['total_conversations'],
                          name="Conversations", line=dict(color='#FF9800', width=4),
                          fill='tonexty', fillcolor='rgba(255, 152, 0, 0.2)'),
                secondary_y=True,
            )
            fig.update_layout(
                title="Sessions & Conversations Over Time",
                height=450,
                template="plotly_white",
                showlegend=True,
                hovermode="x unified",
                font=dict(family="Inter", color="#2D3748"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            weekly_volume = filtered_df.groupby(filtered_df['timestamp'].dt.to_period('W')).size().reset_index()
            weekly_volume['timestamp'] = weekly_volume['timestamp'].dt.start_time
            weekly_volume.columns = ['week', 'message_count']
            
            fig = px.bar(weekly_volume, x='week', y='message_count',
                        title="Weekly Message Volume",
                        color='message_count',
                        color_continuous_scale=['#FF5722', '#FF9800', '#FFC107', '#FFEB3B', '#CDDC39', '#8BC34A'])
            fig.update_layout(height=450, template="plotly_white", 
                            font=dict(family="Inter", color="#2D3748"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            hourly_pattern = filtered_df.groupby(filtered_df['timestamp'].dt.hour).size().reset_index()
            hourly_pattern.columns = ['hour', 'messages']
            
            fig = px.area(hourly_pattern, x='hour', y='messages',
                         title="Hourly Message Patterns",
                         color_discrete_sequence=['#E91E63'])
            fig.update_traces(fill='tonexty', fillcolor='rgba(233, 30, 99, 0.3)')
            fig.update_layout(height=450, template="plotly_white", 
                            font=dict(family="Inter", color="#2D3748"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # === 3. INTENT ANALYSIS ===
        st.markdown('<div class="section-header"><h2>🎯 Intent Insights</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            intent_distribution = filtered_df['intent'].value_counts().head(10)
            bright_colors = ['#FF4081', '#00BCD4', '#FF5722', '#8BC34A', '#FFC107', 
                           '#E91E63', '#00E676', '#FF6D00', '#651FFF', '#00C853']
            fig = px.pie(values=intent_distribution.values, names=intent_distribution.index,
                        title="Top Customer Intents",
                        color_discrete_sequence=bright_colors)
            fig.update_layout(height=450, font=dict(family="Inter", color="#2D3748"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            intent_sentiment = filtered_df.groupby('intent')['sentiment'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=intent_sentiment.values, y=intent_sentiment.index,
                        orientation='h', title="Sentiment by Intent",
                        color=intent_sentiment.values,
                        color_continuous_scale=['#FF1744', '#FF5722', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50'])
            fig.update_layout(height=450, font=dict(family="Inter", color="#2D3748"),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # === 4. PRODUCT & SENTIMENT INTELLIGENCE ===
        st.markdown('<div class="section-header"><h2>🛍️ Product Insights</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'products' in filtered_df.columns:
                products_data = []
                for products in filtered_df['products'].dropna():
                    if products:
                        products_data.extend([p.strip() for p in str(products).split(',')])
                
                if products_data:
                    products_df = pd.DataFrame(products_data, columns=['product'])
                    product_counts = products_df['product'].value_counts().head(15)
                    
                    fig = px.bar(x=product_counts.values, y=product_counts.index,
                                orientation='h', title="Top Discussed Products",
                                color=product_counts.values,
                                color_continuous_scale=['#9C27B0', '#E91E63', '#FF5722', '#FF9800', '#FFC107'])
                    fig.update_layout(height=500, font=dict(family="Inter", color="#2D3748"),
                                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No product data available")
        
        with col2:
            daily_sentiment = filtered_df.groupby(filtered_df['timestamp'].dt.date)['sentiment'].mean().reset_index()
            daily_sentiment.columns = ['date', 'avg_sentiment']
            
            fig = px.line(daily_sentiment, x='date', y='avg_sentiment',
                         title="Sentiment Trends",
                         line_shape='spline', color_discrete_sequence=['#00BCD4'])
            fig.update_traces(line=dict(width=4))
            fig.add_hline(y=5, line_dash="dash", line_color="#FF5722", line_width=3,
                         annotation_text="Neutral (5.0)", annotation_font_color="#FF5722")
            fig.update_layout(height=500, template="plotly_white", 
                            font=dict(family="Inter", color="#2D3748"),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # === 5. OPERATIONAL METRICS ===
        st.markdown('<div class="section-header"><h2>📦 Operations</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            returns_data = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg({
                'returns_count': 'sum',
                'exchanges_count': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=returns_data['timestamp'], y=returns_data['returns_count'],
                mode='lines+markers', name='Returns',
                line=dict(color='#FF1744', width=4),
                marker=dict(size=8, color='#FF1744'),
                fill='tonexty', fillcolor='rgba(255, 23, 68, 0.3)'
            ))
            fig.add_trace(go.Scatter(
                x=returns_data['timestamp'], y=returns_data['exchanges_count'],
                mode='lines+markers', name='Exchanges',
                line=dict(color='#00E676', width=4),
                marker=dict(size=8, color='#00E676'),
                fill='tonexty', fillcolor='rgba(0, 230, 118, 0.3)'
            ))
            fig.update_layout(title="Returns & Exchanges", height=450, template="plotly_white", 
                            font=dict(family="Inter", color="#2D3748"),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sentiment_returns = filtered_df.groupby('sentiment').agg({
                'returns_count': 'sum',
                'exchanges_count': 'sum'
            }).reset_index()
            
            fig = px.scatter(sentiment_returns, x='sentiment', y='returns_count',
                           size='exchanges_count', title="Sentiment vs Returns",
                           color='exchanges_count', 
                           color_continuous_scale=['#9C27B0', '#E91E63', '#FF5722', '#FF9800', '#FFC107'])
            fig.update_traces(marker=dict(line=dict(width=2, color='white')))
            fig.update_layout(height=450, template="plotly_white", 
                            font=dict(family="Inter", color="#2D3748"),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # === 6. DETAILED ANALYTICS TABLES ===
        st.markdown('<div class="section-header"><h2>📋 Detailed Reports</h2></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🎯 Intents", "🛍️ Products", "📊 Sessions"])
        
        with tab1:
            intent_analytics = filtered_df.groupby('intent').agg({
                'session_id': 'nunique',
                'sentiment': ['mean', 'std'],
                'returns_count': 'sum',
                'exchanges_count': 'sum',
                'id': 'count'
            }).round(2)
            intent_analytics.columns = ['Unique Sessions', 'Avg Sentiment', 'Sentiment Std', 'Total Returns', 'Total Exchanges', 'Total Messages']
            st.dataframe(intent_analytics, use_container_width=True)
        
        with tab2:
            if 'categories' in filtered_df.columns:
                category_performance = filtered_df.groupby('categories').agg({
                    'session_id': 'nunique',
                    'sentiment': 'mean',
                    'returns_count': 'sum'
                }).round(2)
                category_performance.columns = ['Unique Sessions', 'Avg Sentiment', 'Total Returns']
                st.dataframe(category_performance, use_container_width=True)
            else:
                st.info("Category data not available")
        
        with tab3:
            recent_sessions = filtered_df.nlargest(20, 'sentiment')[
                ['timestamp', 'session_id', 'intent', 'sentiment', 'returns_count', 'user_message']
            ]
            st.dataframe(recent_sessions, use_container_width=True)
    
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        st.info("Please verify your database configuration.")
        st.markdown("### Expected Data Structure:")
        st.code("""
        Columns: id, timestamp, session_id, user_message, response, 
                intent, products, categories, sentiment, keywords,
                returns_count, exchanges_count
        """)

if __name__ == "__main__":
    main()