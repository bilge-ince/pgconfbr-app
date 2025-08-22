# PGConf.br Product Recommendation Engine

A demonstration application showcasing PostgreSQL's vector similarity search capabilities for product recommendations. This application demonstrates the power of **pgvector**, **HNSW indexing**, and **iterative scan optimization** for high-performance vector search at scale.

## 🚀 Features

- **Semantic Text Search**: Natural language product discovery using sentence transformers
- **BM25 Full-Text Search**: Traditional PostgreSQL text search with ts_rank scoring
- **HNSW Iterative Scan**: Optimized vector search with relaxed_order mode demonstration  
- **Image Similarity Search**: CLIP-based visual product matching
- **Performance Comparison**: Side-by-side search mode comparisons with timing metrics
- **Real-time Web Interface**: Interactive Streamlit application with instant results
- **Production Scale**: 44,093 products with pre-computed embeddings

## 🏗️ Architecture Highlights

- **pgvector 0.8.0** with HNSW indexing (m=16, ef_construction=40)
- **Dual embedding tables**: Separate tables for regular and iterative scan demonstrations
- **Model caching**: 16x performance improvement with pre-warmed models
- **AWS S3 integration**: Cloud-hosted product images
- **Sub-second search**: Optimized for conference demonstrations

## Prerequisites

- PostgreSQL 12+ with **pgvector extension** installed
- Python 3.8+
- Database user with CREATE EXTENSION privileges
- AWS S3 bucket for product images (or local image directory)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pgconfbr-app
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file or export:
   ```bash
   DB_NAME=your_database_name
   DB_USER=your_username  
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

## 🚀 Quick Start

1. **Initialize database and create tables**
   ```bash
   python setup_db.py
   ```

2. **Set up pgvector embeddings** (⚠️ This takes ~10-15 minutes for 44K products)
   ```bash
   python setup_pgvector.py
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

4. **Visit** `http://localhost:8501` and start exploring!

## 🎯 Demo Usage

### Search Modes Comparison

1. **Semantic Search**: Enter "rock music outfits" in semantic search box
2. **BM25 Search**: Try "dress" in full-text search box  
3. **HNSW Iterative Scan**: Check "Enable Iterative Scan" → search for "summer outfits"
4. **Image Search**: Upload a product image to find similar items

### Performance Demonstrations

- **Model pre-warming**: Notice the one-time loading at app startup
- **Sub-second search**: All searches complete in <100ms after model loading
- **Metrics display**: Each search shows embedding time vs search time breakdown
- **Scale demonstration**: Search across 44,093 products with HNSW index

## Architecture

### Database Schema
- `products_pgconf`: Main product catalog with metadata
- `products_embeddings_pgvector`: Text and image embeddings for pgvector
- `products_embeddings_vchord`: Text and image embeddings for vchord

### Embedding Models
- **Text**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
- **Images**: openai/clip-vit-base-patch32 (512 dimensions)
- **Alternative**: GritLM/GritLM-7B, Ollama models

### Vector Search
- Text similarity uses cosine distance (`<=>`)
- Image similarity uses L2 distance (`<->`)
- HNSW indexing for efficient similarity search

## Usage

1. **Browse by Category**: Select product categories from the dropdown
2. **Text Search**: Enter search terms for semantic or full-text search
3. **Image Search**: Upload an image to find visually similar products
4. **Filter by Gender**: Combine searches with gender-specific filtering

## 🏗️ Technical Architecture

### Database Schema
```sql
-- Main product catalog
products_pgconf (44,093 records)
├── img_id (PRIMARY KEY)
├── productdisplayname
├── gender, mastercategory, subcategory
├── articletype, basecolour, season
└── usage, year

-- Regular semantic search embeddings  
products_embeddings_pgvector
├── id → products_pgconf.img_id
├── embeddings vector(384)  -- text embeddings
└── image_embedding vector(512)  -- CLIP image embeddings

-- Iterative scan demonstration embeddings
products_embeddings_iterative_scan  
├── id → products_pgconf.img_id
├── embeddings vector(384) 
└── image_embedding vector(512)
```

### HNSW Index Configuration
```sql
-- Regular search index
CREATE INDEX ON products_embeddings_pgvector 
USING hnsw (embeddings vector_l2_ops) 
WITH (m = 16, ef_construction = 100);

-- Iterative scan demo index  
CREATE INDEX ON products_embeddings_iterative_scan 
USING hnsw (embeddings vector_l2_ops) 
WITH (m = 16, ef_construction = 40);
```

### Embedding Models
- **Text Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384D)
- **Image Embeddings**: `openai/clip-vit-base-patch32` (512D)  
- **Distance Metrics**: Cosine distance (`<=>`) for text, L2 distance (`<->`) for images

### Search Performance
- **Model Loading**: 1.3s (one-time, cached via `@st.cache_resource`)
- **Embedding Generation**: ~40ms per query
- **HNSW Search**: ~48ms across 44K records
- **Total Search Time**: <100ms after model warm-up

## 📊 Dataset

The application includes a large dataset of product images and metadata:
- **44,093 products** from fashion/apparel domain
- **Product images** hosted on AWS S3
- **Rich metadata**: gender, category, color, season, usage type
- **Global coverage**: Multi-language product names
- **Categories**: Apparel, Accessories, Footwear, Personal Care

## 🎪 Conference Demo Script

### HNSW Iterative Scan Demo Flow
1. **Show regular semantic search** - "rock music outfits" 
2. **Enable iterative scan checkbox** - reveal the optimized search button
3. **Search with iterative scan** - same query, show performance metrics
4. **Highlight the differences**:
   - Different embedding tables (`products_embeddings_pgvector` vs `products_embeddings_iterative_scan`)
   - HNSW settings: `ef_search=40`, `iterative_scan=relaxed_order`
   - Performance breakdown: embedding time vs search time
   - Real-world scale: 44K products, 71MB HNSW index

### Key Talking Points
- **PostgreSQL as Vector Database**: Production-ready with pgvector
- **HNSW Performance**: Sub-second search across tens of thousands of records
- **Iterative Scan Optimization**: Fine-tuned search behavior for different use cases
- **Hybrid Search**: Combine vector similarity with traditional PostgreSQL filters

## Contributing

This is a demonstration application for PGConf.br showcasing PostgreSQL's vector capabilities.

## License

Apache License 2.0 - Built for the PostgreSQL community
