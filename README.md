# PGConf.br Product Recommendation Engine

A demonstration application showcasing PostgreSQL's vector similarity search capabilities for product recommendations. This application supports both text and image-based product search using embeddings and features integration with AWS S3 for image storage.

## Features

- **Semantic Text Search**: Find products using natural language queries
- **Image Similarity Search**: Upload images to find visually similar products  
- **Full-Text Search**: Traditional BM25-based text search
- **Hybrid Filtering**: Combine vector search with categorical filters (gender, category)
- **Dual Vector Backends**: Support for both pgvector and vchord extensions
- **Real-time Web Interface**: Built with Streamlit
- **Cloud Integration**: AWS S3 for image storage and display

## Prerequisites

- PostgreSQL database with pgvector extension installed
- Python 3.8+
- Database user with CREATE EXTENSION privileges
- For vchord backend: vchord extension must be installed

## Installation

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
   ```bash
   export DB_NAME=your_database_name
   export DB_USER=your_username
   export DB_PASSWORD=your_password
   export DB_HOST=localhost
   export DB_PORT=5432
   ```

## Setup

1. **Create base tables and populate data**
   ```bash
   python setup_db.py
   ```

2. **Set up vector embeddings** (choose one):
   
   For pgvector backend:
   ```bash
   python setup_pgvector.py
   ```
   
   For vchord backend:
   ```bash
   python setup_vchord.py
   ```

## Running the Application

**Main application (pgvector backend):**
```bash
streamlit run app.py
```

**Alternative application (vchord backend):**
```bash
streamlit run app_vchord.py
```

The application will be available at `http://localhost:8501`

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

## Dataset

The application includes a large dataset of product images and metadata:
- 44,000+ product records
- Product images stored in `dataset/images/`
- Categories include apparel, accessories, and more
- Metadata includes gender, category, color, season, etc.

## Contributing

This is a demonstration application for PGConf.br showcasing PostgreSQL's vector capabilities.

## License

[Add your license information here]
