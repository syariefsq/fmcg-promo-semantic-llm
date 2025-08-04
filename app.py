"""
Simple Streamlit Application for FMCG Promotion Copy Re-use Recommender

This is a simplified version designed for learning and portfolio purposes.
All complex syntax has been removed to make it easy to understand and follow.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# Import our simple modules
from data import DataCleaner
from search import VectorDatabase
from analysis import NLPAnalyzer

class PromotionApp:
    """
    Simple promotion recommender application class.
    Easy to understand and follow for learning purposes.
    """
    
    def __init__(self):
        """Initialize the application with default settings."""
        self.data_cleaner = None
        self.vector_db = None
        self.nlp_analyzer = None
        self.promos_data = None
        self.enhanced_promos = None
        self.skus_data = None
        self.is_ready = False
        
    def setup(self):
        """Set up all components of the application."""
        print("Setting up the application...")
        
        try:
            # Step 1: Load enhanced datasets
            print("Step 1: Loading enhanced datasets...")
            try:
                self.enhanced_promos = pd.read_csv("data/enhanced_promos.csv")
                self.enhanced_skus = pd.read_csv("data/enhanced_skus.csv")
                print("Enhanced datasets v2 loaded successfully")
                
                # Use enhanced data as primary data
                self.promos_data = self.enhanced_promos.copy()
                self.skus_data = self.enhanced_skus.copy()
                
            except Exception as e:
                print(f"Could not load enhanced data: {e}")
                # Fallback to original data cleaning
                print("Falling back to original data...")
                self.data_cleaner = DataCleaner()
                self.promos_data, self.skus_data = self.data_cleaner.clean_all_data(
                    "data/promos.csv", "data/sku_master.csv"
                )
                self.enhanced_promos = self.promos_data.copy()
                self.enhanced_skus = self.skus_data.copy()
            
            # Step 2: Set up vector search with enhanced data
            print("Step 2: Setting up search engine with enhanced data...")
            self.vector_db = VectorDatabase()
            self.vector_db.initialize()
            self.vector_db.add_promos_to_db(self.promos_data)
            
            # Step 3: Set up text analysis
            print("Step 3: Setting up text analysis...")
            self.nlp_analyzer = NLPAnalyzer()
            
            self.is_ready = True
            print("Application setup complete!")
            return self
            
        except Exception as e:
            print(f"Error during setup: {e}")
            return None

def get_app():
    """Get the application instance (cached for performance)."""
    if 'app_instance' not in st.session_state:
        st.session_state.app_instance = PromotionApp().setup()
    return st.session_state.app_instance

def search_promotions(query, filters=None):
    """
    Search for similar promotional headlines.
    Simple function that's easy to understand.
    """
    app = get_app()
    if not app or not app.is_ready:
        return [], []
    
    try:
        # Search for similar promotions
        search_filters = {}
        if filters:
            if filters.get('category') and filters['category'] != 'All':
                search_filters['category'] = filters['category']
            if filters.get('touchpoint') and filters['touchpoint'] != 'All':
                search_filters['touchpoint'] = filters['touchpoint']
        
        similar_promos = app.vector_db.search_similar_promos(
            query=query,
            n_results=filters.get('n_results', 10) if filters else 10,
            category_filter=search_filters.get('category'),
            touchpoint_filter=search_filters.get('touchpoint'),
            min_kpi_lift=filters.get('min_kpi_lift') if filters else None,
            min_kpi_roi=filters.get('min_kpi_roi') if filters else None
        )
        
        # Get SKU suggestions
        sku_suggestions = []
        if similar_promos:
            selected_promo = similar_promos[0]
            category = selected_promo.get('category', '')
            
            # Find SKUs in same category
            same_category_skus = app.skus_data[
                app.skus_data['category'] == category
            ].head(5)
            
            for _, row in same_category_skus.iterrows():
                sku_suggestions.append({
                    'sku': row['sku'],
                    'brand': row['brand'],
                    'product_name': row['product_name'],
                    'pack_size': row['pack_size'],
                    'priority': 'High'
                })
        
        return similar_promos, sku_suggestions
        
    except Exception as e:
        st.error(f"Search error: {e}")
        return [], []

def process_agent_request(user_input):
    """
    Process agent requests in simple, easy to understand way.
    No complex logic, just straightforward responses.
    """
    user_input = user_input.lower()
    app = get_app()
    
    if not app or not app.is_ready:
        return "Sorry, the system is not ready yet. Please wait for initialization."
    
    try:
        # Search requests
        if 'search' in user_input or 'find' in user_input:
            # Extract search query (simple approach)
            if '"' in user_input:
                query = user_input.split('"')[1]
            else:
                query = "DISKON 50%"  # Default
            
            similar_promos, sku_suggestions = search_promotions(query)
            
            response = f"I found {len(similar_promos)} similar promotions for '{query}':\\n\\n"
            
            for i, promo in enumerate(similar_promos[:3], 1):
                response += f"{i}. {promo['headline']}\\n"
                response += f"   Category: {promo['category']}, Brand: {promo['brand']}\\n"
                response += f"   Performance: {promo['kpi_lift']:.1%} lift, {promo['kpi_roi']:.1f}x ROI\\n\\n"
            
            if sku_suggestions:
                response += "Suggested SKUs for copy re-use:\\n"
                for sku in sku_suggestions[:3]:
                    response += f"- {sku['sku']}: {sku['product_name']}\\n"
            
            return response
        
        # Analysis requests
        elif 'analyze' in user_input:
            if '"' in user_input:
                headline = user_input.split('"')[1]
            else:
                headline = "BELI 2 GRATIS 1"
            
            # Simple analysis based on keywords
            if 'diskon' in headline.lower() or '%' in headline:
                emotion = "Excitement"
                promo_type = "Discount"
            elif 'gratis' in headline.lower():
                emotion = "Value"
                promo_type = "Free Gift"
            else:
                emotion = "Trust"
                promo_type = "Other"
            
            response = f"Analysis for '{headline}':\\n"
            response += f"- Emotion: {emotion}\\n"
            response += f"- Type: {promo_type}\\n"
            response += f"- Length: {len(headline)} characters"
            
            return response
        
        # Statistics requests
        elif 'stats' in user_input or 'statistics' in user_input:
            data = app.enhanced_promos
            
            response = "Enhanced Dataset Statistics:\\n\\n"
            response += f"- Total Promotions: {len(data)}\\n"
            response += f"- Categories: {data['category'].nunique()}\\n"
            response += f"- Brands: {data['brand'].nunique()}\\n"
            response += f"- Touchpoints: {data['touchpoint'].nunique()}\\n"
            response += f"- Indonesian Brands: {len([b for b in data['brand'].unique() if b in ['Wings', 'Mayora', 'Kalbe', 'Indofood', 'Wardah']])}\\n"
            response += f"- Average Performance: {data['kpi_lift'].mean():.1%} lift, {data['kpi_roi'].mean():.1f}x ROI\\n"
            response += f"- Average CTR: {data['ctr'].mean():.2%}\\n"
            response += f"- Average Conversion: {data['conversion_rate'].mean():.1%}"
            
            return response
        
        # Default response
        else:
            return """I can help you with:
- Search: "Search for promos like 'DISKON 50%'"
- Analysis: "Analyze 'BELI 2 GRATIS 1'"
- Statistics: "Show me statistics"

What would you like to do?"""
        
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"

def show_search_page():
    """Display the main search and analysis page."""
    st.subheader("🔍 Search Similar Promotions")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter promotional headline:",
            value="BELI 2 GRATIS 1 - Hemat Hingga Rp 25.000!",
            help="Type a promotional headline to find similar ones"
        )
    
    with col2:
        n_results = st.number_input("Results", min_value=1, max_value=20, value=10)
    
    # Filters
    st.subheader("🎛️ Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    app = get_app()
    if app and app.is_ready:
        with col1:
            categories = ['All'] + sorted(list(app.enhanced_promos['category'].unique()))
            category = st.selectbox("Category", categories)
        
        with col2:
            touchpoints = ['All'] + sorted(list(app.enhanced_promos['touchpoint'].unique()))
            touchpoint = st.selectbox("Touchpoint", touchpoints)
        
        with col3:
            min_lift = st.slider("Min Performance", 0.0, 1.0, 0.0, 0.01)
        
        with col4:
            min_roi = st.slider("Min ROI", 0.0, 10.0, 0.0, 0.1)
        
        # Search button
        if st.button("🔍 Search", type="primary"):
            with st.spinner("Searching..."):
                filters = {
                    'category': category,
                    'touchpoint': touchpoint,
                    'min_kpi_lift': min_lift,
                    'min_kpi_roi': min_roi,
                    'n_results': n_results
                }
                
                similar_promos, sku_suggestions = search_promotions(query, filters)
                
                if similar_promos:
                    st.subheader("📊 Similar Promotions")
                    
                    # Create simple display table
                    results_data = []
                    for promo in similar_promos:
                        results_data.append({
                            'Headline': promo['headline'],
                            'Category': promo['category'],
                            'Brand': promo['brand'],
                            'Touchpoint': promo['touchpoint'],
                            'KPI Lift': f"{promo['kpi_lift']:.1%}",
                            'ROI': f"{promo['kpi_roi']:.1f}x",
                            'Similarity': f"{promo['similarity_score']:.3f}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Show SKU suggestions
                    if sku_suggestions:
                        st.subheader("🎯 SKU Suggestions")
                        sku_df = pd.DataFrame(sku_suggestions)
                        st.dataframe(sku_df, use_container_width=True)
                else:
                    st.warning("No similar promotions found. Try different keywords or filters.")
    else:
        st.error("Application not ready. Please refresh the page.")

def show_agent_page():
    """Display the AI agent chat page."""
    st.subheader("🤖 AI Assistant")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about promotions..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_agent_request(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def show_analytics_page():
    """Display the analytics dashboard."""
    st.subheader("📊 Analytics Dashboard")
    
    app = get_app()
    if not app or not app.is_ready:
        st.error("Application not ready. Please refresh the page.")
        return
    
    data = app.enhanced_promos
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Promotions", len(data))
    
    with col2:
        avg_lift = data['kpi_lift'].mean()
        st.metric("Avg Performance", f"{avg_lift:.1%}")
    
    with col3:
        avg_roi = data['kpi_roi'].mean()
        st.metric("Avg ROI", f"{avg_roi:.1f}x")
    
    with col4:
        st.metric("Categories", data['category'].nunique())
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Category chart
        category_counts = data['category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Promotions by Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Touchpoint chart
        touchpoint_counts = data['touchpoint'].value_counts()
        fig = px.pie(
            values=touchpoint_counts.values,
            names=touchpoint_counts.index,
            title="Promotions by Touchpoint"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(data, x='kpi_lift', title="Performance Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(data, x='kpi_roi', title="ROI Distribution")
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function - simple and easy to follow."""
    
    # Page setup
    st.set_page_config(
        page_title="FMCG Promotion Recommender",
        page_icon="🎯",
        layout="wide"
    )
    
    # Header
    st.title("🎯 FMCG Promotion Copy Re-use Recommender")
    st.markdown("*Find similar high-performing promotional copies for your marketing campaigns*")
    
    # Initialize app
    if 'app_initialized' not in st.session_state:
        with st.spinner("Setting up application... This may take a moment."):
            app = get_app()
            if app and app.is_ready:
                st.session_state.app_initialized = True
                st.success("Application ready!")
            else:
                st.error("Failed to initialize application. Please refresh the page.")
                return
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Search & Analysis", "🤖 AI Assistant", "📊 Analytics"]
    )
    
    # Example queries
    st.sidebar.markdown("### 💡 Example Queries")
    examples = [
        "BELI 2 GRATIS 1 - Hemat Hingga Rp 25.000!",
        "FLASH SALE 50% OFF - Stok Terbatas!",
        "GRATIS SAMPLE SUSU - Coba Sekarang!",
        "WIN HADIAH JUTAAN - Upload & Menang!"
    ]
    
    for example in examples:
        st.sidebar.markdown(f"• `{example[:30]}...`")
    
    # Show selected page
    if page == "🔍 Search & Analysis":
        show_search_page()
    elif page == "🤖 AI Assistant":
        show_agent_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How to use this app:**
    
    1. **Search**: Enter promotional headlines to find similar high-performing copies
    2. **AI Assistant**: Chat with the AI to get recommendations and analysis
    3. **Analytics**: Explore data insights and performance metrics
    
    Built for FMCG marketers to find and re-use successful promotional copy! 🚀
    """)

if __name__ == "__main__":
    main()