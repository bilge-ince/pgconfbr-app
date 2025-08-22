import pandas as pd
import psycopg2

import time
import os
import numpy as np
from io import StringIO
from PIL import Image
from psycopg2.extras import execute_batch

import sys

# Add the parent directory of 'code' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Python search paths:", sys.path)  # Debugging line
from utils.db_connection import create_db_connection
from utils.generate_embeddings import generate_short_text_embeddings, initialize_model


def initialize_database(conn):
    """Initialize the database with required extensions and tables."""
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            print(f"Extension creation warning (may be expected): {e}")
        _create_tables(cur)


def _create_tables(cur, text_vector_size=384):
    """Create required tables."""
    # Main embeddings table
    query = """
        CREATE TABLE IF NOT EXISTS products_embeddings_pgvector(
            id INTEGER PRIMARY KEY REFERENCES products_pgconf(img_id) ON DELETE CASCADE,
            embeddings vector(%s),
            image_embedding vector(512));
    """
    cur.execute(query, (text_vector_size,))
    
    # Separate table for iterative scan demonstrations
    iterative_scan_query = """
        CREATE TABLE IF NOT EXISTS products_embeddings_iterative_scan(
            id INTEGER PRIMARY KEY REFERENCES products_pgconf(img_id) ON DELETE CASCADE,
            embeddings vector(%s),
            image_embedding vector(512));
    """
    cur.execute(iterative_scan_query, (text_vector_size,))


def create_pgvector_indexes(conn):
    cur = conn.cursor()
    # L2 index for main table
    cur.execute(
        """CREATE INDEX IF NOT EXISTS idx_products_embeddings_pgvector_hnsw 
        ON products_embeddings_pgvector USING hnsw (embeddings vector_l2_ops)
        WITH (m = 16, ef_construction = 100);"""
    )
    
    # L2 index for iterative scan table with relaxed_order setting
    cur.execute(
        """CREATE INDEX IF NOT EXISTS idx_products_embeddings_iterative_scan_hnsw 
        ON products_embeddings_iterative_scan USING hnsw (embeddings vector_l2_ops)
        WITH (m = 16, ef_construction = 100);"""
    )
    
    print("Created HNSW indexes for both embedding tables")


def generate_embeddings_single(conn):
    table_name = "products_embeddings_gritllm"
    function_start_time = time.time()
    # Run for S3 bucket
    # The idea is to create a retriever for the images bucket so the image search can run over it.
    # Load the model and processor with timing
    fetch_start = time.time()
    cursor = conn.cursor()
    cursor.execute("SELECT img_id, productdisplayname FROM products_pgconf;")
    result = cursor.fetchall()
    fetch_end = time.time()
    total_rows_inserted = 0
    for i in range(0, len(result)):
        batch_text = result[i][1]
        if batch_text:
            embedding_output = generate_short_text_embeddings(
                batch_text, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", trust_remote_code=True
            )
            cursor.execute(
                f"INSERT INTO {table_name} (id, embeddings) VALUES (%s, %s)",
                (result[i][0], embedding_output),
            )

    function_end_time = time.time()
    total_time = function_end_time - function_start_time
    print(f"Total Rows: {total_rows_inserted}")
    print(f"Total function execution time: {total_time} seconds")
    print(f"Fetching time: {fetch_end - fetch_start} seconds")


def generate_store_embeddings(conn, base_path, batch_size=1000):
    """
    This function is a specific implementation for pgvector semantic search capability
    """
    table_name = "products_embeddings_pgvector"

    function_start_time = time.time()
    total_rows_inserted = 0
    all_ids = []
    all_texts = []
    
    text_tokenizer = None
    text_model = None
    try:
        cursor = conn.cursor()

        fetch_start = time.time()
        # Fetch only products that don't have embeddings yet
        cursor.execute("""
            SELECT p.img_id, p.productdisplayname 
            FROM products_pgconf p 
            LEFT JOIN products_embeddings_pgvector e ON p.img_id = e.id 
            WHERE e.id IS NULL
            ORDER BY p.img_id;
        """)
        all_results = cursor.fetchall()
        fetch_end = time.time()
        print(
            f"Fetched {len(all_results)} rows without embeddings in {fetch_end - fetch_start:.2f} seconds."
        )

        data_to_insert = []
        text_model, text_tokenizer = initialize_model(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
            trust_remote_code=False
        )
        for i in range(0, len(all_results)):
            img_id, product_text = all_results[i]

            if product_text:  # Ensure there is text to process
                all_ids.append(img_id)
                all_texts.append(product_text)

            # When the batch is full or it's the last item
            if len(all_texts) >= batch_size or (
                i == len(all_results) - 1 and len(all_texts) > 0
            ):
                print(f"Processing batch of {len(all_texts)} texts...")
                batch_embedding_start_time = time.time()
                # Generate embeddings for the current batch of texts
                # Ensure you use the correct model name here if it's different from the default
                batch_embeddings = generate_short_text_embeddings(
                    all_texts, text_tokenizer, text_model
                )
                batch_embedding_end_time = time.time()
                print(
                    f"Generated embeddings for batch in {batch_embedding_end_time - batch_embedding_start_time:.2f} seconds."
                )

                # Prepare data for batch insertion
                for img_id_item, embedding_output in zip(all_ids, batch_embeddings):
                    # pgvector expects a NumPy array or a list for the vector type
                    data_to_insert.append((img_id_item, embedding_output))

                # Insert the batch into the database
                insert_query = f"INSERT INTO {table_name} (id, embeddings) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET embeddings = EXCLUDED.embeddings;"
                execute_batch(cursor, insert_query, data_to_insert)
                conn.commit()  # Commit after each batch insertion

                total_rows_inserted += len(data_to_insert)
                # Get current total count for progress tracking
                cursor.execute("SELECT COUNT(*) FROM products_embeddings_pgvector;")
                current_total = cursor.fetchone()[0]
                print(
                    f"Inserted batch of {len(data_to_insert)}. Current total embeddings: {current_total}/44093 ({current_total/44093*100:.1f}%)"
                )

                # Clear lists for the next batch
                all_ids = []
                all_texts = []
                data_to_insert = []

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()  # Rollback any pending transactions on error
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("Database connection closed.")

    function_end_time = time.time()
    total_time = function_end_time - function_start_time
    print(f"\n--- Summary ---")
    print(f"Total rows processed and attempted for insertion: {total_rows_inserted}")
    print(f"Total function execution time: {total_time:.2f} seconds")
    if (
        "fetch_end" in locals() and "fetch_start" in locals()
    ):  # Check if fetch timing variables exist
        print(f"Data fetching time: {fetch_end - fetch_start:.2f} seconds")
    print("Batch embedding process complete.")


def populate_iterative_scan_table(conn, batch_size=1000):
    """
    Populate the iterative scan table with embeddings from the main table
    """
    function_start_time = time.time()
    total_rows_inserted = 0
    
    try:
        cursor = conn.cursor()
        
        # Copy embeddings from main table to iterative scan table
        print("Copying embeddings from main table to iterative scan table...")
        cursor.execute("""
            INSERT INTO products_embeddings_iterative_scan (id, embeddings, image_embedding)
            SELECT id, embeddings, image_embedding 
            FROM products_embeddings_pgvector
            ON CONFLICT (id) DO UPDATE SET 
                embeddings = EXCLUDED.embeddings,
                image_embedding = EXCLUDED.image_embedding;
        """)
        
        # Get count of inserted rows
        cursor.execute("SELECT COUNT(*) FROM products_embeddings_iterative_scan;")
        total_rows_inserted = cursor.fetchone()[0]
        
        conn.commit()
        print(f"Successfully copied {total_rows_inserted} embeddings to iterative scan table")
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if cursor:
            cursor.close()
    
    function_end_time = time.time()
    total_time = function_end_time - function_start_time
    print(f"Iterative scan table population took {total_time:.2f} seconds")


def load_images_batch(batch_ids, base_path, processor):
    images, valid_paths = [], []
    for image_id in batch_ids:
        image_path = f"{base_path}/{image_id}.jpg"
        try:
            img = Image.open(image_path)
            img.verify()  # Verify the image integrity
            img = Image.open(image_path)  # Reopen to reset file pointer
            images.append(img)
            # valid_paths.append(image_path)
        except OSError as e:
            print(f"Failed to process image {image_path}: {e}")
            continue  # Skip problematic images
    if images:
        return processor(
            text=["dummy text"] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        )
    else:
        return None, []


def main():
    conn = None
    try:
        conn = create_db_connection()  # Connect to the database
        conn.autocommit = True  # Enable autocommit for creating the database
        start_time = time.time()
        initialize_database(conn)  # Initialize the db extensions and necessary tables
        create_pgvector_indexes(conn)  # Create the indexes for pgvector
        generate_store_embeddings(
            conn, "dataset/images", 100
        )  # Create and refresh the retriever for the products table and images bucket
        
        # Populate the iterative scan table with embeddings
        populate_iterative_scan_table(conn)
        vector_time = time.time() - start_time
        print(f"Total process time: {vector_time:.4f} seconds.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
