
![preview of the app](copy-reuse-recommender.jpeg)

# 🎯 FMCG Promotion Copy Re-use Recommender

A semantic search and analytics tool for FMCG marketers and trade marketing teams. Given a past promotional headline, the app finds similar high-performing promo copies, buckets them by emotional tone, and suggests which current SKUs could re-use that copy.

## 🚀 Features

- **Semantic Search**: Find similar promotional headlines using AI embeddings
- **Emotion Analysis**: Detect emotional tone (excitement, urgency, trust, joy, etc.)
- **Zero-Shot Classification**: Classify promo types (Price-off, Bundle, Value-add, Sampling, Contest)
- **KPI Filtering**: Filter by sales lift and ROI performance
- **SKU Suggestions**: Get recommendations for copy re-use opportunities
- **Interactive Dashboard**: Beautiful Gradio interface for easy interaction
- **Agent Compatibility**: All modules can be called as autonomous agents

## 📋 Requirements

- Python 3.9+
- 8GB+ RAM (for model loading)
- Internet connection (for model downloads)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd FMCG-Promotion-Copy-Re-use-Recommender
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Setup the project**:
```bash
python setup_simple.py
```

## 🏃‍♂️ Quick Start

### Simple One-Command Launch
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

### Test Everything Works
```bash
python simple_test.py
```

### Use as Python Modules
```python
from app import PromotionApp

# Initialize the app
app = PromotionApp()
app = app.setup()

# Search for similar promos
similar_promos, sku_suggestions = search_promotions("BELI 2 GRATIS 1")
print(f"Found {len(similar_promos)} similar promos")
```

### Use Individual Modules
```python
# Data cleaning
from data_cleaning import DataCleaner
cleaner = DataCleaner()
promos_df, skus_df = cleaner.clean_all_data("data/promos.csv", "data/sku_master.csv")

# Vector search
from vector_db import VectorDatabase
vector_db = VectorDatabase()
vector_db.initialize()
similar = vector_db.search_similar_promos("DISKON 50%", n_results=5)

# NLP analysis
from nlp_modules import NLPAnalyzer
analyzer = NLPAnalyzer()
result = analyzer.analyze_headline("GRATIS SAMPLING - Coba Sekarang!")
```

## 📁 Project Structure

```
FMCG Promotion Copy Re-use Recommender/
├── data/
│   ├── promos.csv              # Promotional data (50 records)
│   ├── enhanced_promos.csv     # Enhanced data with derived features
│   └── sku_master.csv          # SKU master data (105 records)
├── app.py                      # Main Streamlit application
├── data_cleaning.py            # Data validation and cleaning
├── vector_db.py                # Vector database and semantic search
├── nlp_modules.py              # NLP analysis modules
├── eda_analysis.ipynb          # Exploratory data analysis
├── setup_simple.py             # Setup and installation script
├── simple_test.py              # Simple functionality test
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── PORTFOLIO_SUMMARY.md        # Portfolio project summary
└── .gitignore                  # Git ignore rules
```

## 📊 Sample Data

The project includes realistic sample data with 50 promotional records:

- **Categories**: Snacks, Beverages
- **Brands**: Indomie, Chitato, Lays, Oreo, Ultra Milk, Aqua, etc.
- **Touchpoints**: in-store, digital, flyer
- **KPIs**: Sales lift (0-35%), ROI (2.1x-4.3x)

### Data Structure

**promos.csv**:
- `promo_id`: Unique identifier
- `headline`: Promotional headline text
- `touchpoint`: Marketing channel
- `category`: Product category
- `brand`: Brand name
- `sku`: Product SKU
- `period`: Time period (e.g., "2023-Q1")
- `kpi_lift`: Sales lift percentage
- `kpi_roi`: ROI multiplier
- `description`: Optional description

**sku_master.csv**:
- `sku`: Product SKU
- `brand`: Brand name
- `category`: Product category
- `product_name`: Product name
- `pack_size`: Package size

## 🏗️ Project Structure

```
FMCG-Promotion-Copy-Re-use-Recommender/
├── data/
│   ├── promos.csv          # Sample promotional data
│   └── sku_master.csv      # Sample SKU master data
├── data_cleaning.py        # Data validation and preprocessing
├── vector_db.py           # Vector database and semantic search
├── nlp_modules.py         # Emotion analysis and classification
├── app.py                 # Main application and Gradio interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── PRD.md               # Product Requirements Document
```

## 🤖 Agent Mode Usage

All modules are designed to be called as autonomous agents. Here are some example agent prompts:

### Data Cleaning Agent
```
"Clean the promotional data from promos.csv and sku_master.csv, 
remove invalid records, and provide statistics on the cleaned dataset."
```

### Vector Search Agent
```
"Given the headline 'BELI 2 GRATIS 1 - Hemat Rp 15.000!' and a minimum 
KPI lift of 20%, find the top 5 most similar past promos."
```

### NLP Analysis Agent
```
"Analyze the emotional tone and promotional type of the headline 
'DISKON 50% - Stok Terbatas!'"
```

### SKU Recommendation Agent
```
"For the promotional record with headline 'GRATIS SAMPLING - Coba Sekarang!', 
suggest 3 SKUs from the same category that could re-use this copy."
```

## 🎨 Dashboard Features

### Search Interface
- **Query Input**: Enter promotional headlines to search
- **Filters**: Category, touchpoint, KPI lift, ROI, emotion
- **Results Count**: Adjustable number of results (1-20)

### Results Display
- **Similar Promos**: Table with headline, category, brand, SKU, KPIs, emotion, similarity score
- **SKU Suggestions**: Table with recommended SKUs for copy re-use

### Quick Examples
Pre-loaded example queries for easy testing:
- "BELI 2 GRATIS 1 - Hemat Rp 15.000!"
- "DISKON 50% - Stok Terbatas!"
- "GRATIS SAMPLING - Coba Sekarang!"
- "WIN PRIZE - Kirim Foto & Menang!"
- "GRATIS TUMBLER - Beli 3 Dapat 1!"

## 🔧 Configuration

### Model Settings
- **Embedding Model**: `all-MiniLM-L6-v2` (fast, accurate)
- **Emotion Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Zero-shot Model**: `facebook/bart-large-mnli`

### Performance Tuning
- **Vector DB**: ChromaDB with persistent storage
- **Batch Processing**: Optimized for large datasets
- **Caching**: Models cached after first load

## 📈 Usage Examples

### Example 1: Find High-Performing Bundle Promos
```python
# Search for bundle promotions with high ROI
similar_promos, sku_suggestions = app.search_and_analyze(
    query="BELI 2 GRATIS 1",
    category_filter="Snacks",
    min_kpi_roi=3.0,
    n_results=10
)
```

### Example 2: Analyze Emotional Tone
```python
# Analyze emotion and type of a headline
from nlp_modules import NLPAnalyzer
analyzer = NLPAnalyzer()
result = analyzer.analyze_headline("WIN PRIZE - Kirim Foto & Menang!")
print(f"Emotion: {result['summary']['primary_emotion']}")
print(f"Type: {result['summary']['promo_type']}")
```

### Example 3: Get SKU Recommendations
```python
# Get SKU suggestions for copy re-use
selected_promo = {
    'category': 'Beverages',
    'brand': 'Ultra Milk'
}
suggestions = app.get_sku_suggestions(selected_promo, n_suggestions=5)
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Errors**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   python -c "from nlp_modules import NLPAnalyzer; analyzer = NLPAnalyzer(); analyzer.initialize()"
   ```

2. **Memory Issues**
   ```bash
   # Use smaller models
   export TRANSFORMERS_CACHE="/tmp/transformers_cache"
   ```

3. **ChromaDB Errors**
   ```bash
   # Clear vector database
   rm -rf ./chroma_db/
   ```

### Performance Tips

- **First Run**: Models will download (~2GB), subsequent runs are faster
- **Memory**: Close other applications during first model load
- **Batch Processing**: Use batch functions for large datasets

## 🔄 Extending the Project

### Adding New Data Sources
1. Update `data_cleaning.py` to handle new data formats
2. Modify `vector_db.py` for new embedding strategies
3. Extend `nlp_modules.py` for additional analysis

### Customizing Models
```python
# Use different embedding model
vector_db = VectorDatabase(model_name="all-mpnet-base-v2")

# Use custom emotion model
emotion_analyzer = EmotionAnalyzer(model_name="your-custom-model")
```

### Adding New Features
1. **New Filter Types**: Extend filter logic in `app.py`
2. **Additional Analytics**: Add new analysis functions in `nlp_modules.py`
3. **Custom UI Components**: Modify Gradio interface in `app.py`

## 📝 API Reference

### Core Classes

#### `DataCleaner`
- `clean_all_data(promos_path, skus_path)` → Tuple[DataFrame, DataFrame]
- `validate_promos(promos_df)` → DataFrame
- `validate_skus(skus_df)` → DataFrame

#### `VectorDatabase`
- `search_similar_promos(query, n_results, filters)` → List[Dict]
- `add_promos_to_db(promos_df)` → Dict
- `get_collection_stats()` → Dict

#### `NLPAnalyzer`
- `analyze_headline(headline)` → Dict
- `analyze_batch_headlines(headlines)` → List[Dict]
- `analyze_promos_dataframe(promos_df)` → DataFrame

#### `PromotionRecommenderApp`
- `search_and_analyze(query, filters)` → Tuple[List[Dict], List[Dict]]
- `get_sku_suggestions(selected_promo)` → List[Dict]
- `get_available_filters()` → Dict

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **ChromaDB** for vector database
- **Gradio** for the web interface
- **Sentence Transformers** for embeddings
