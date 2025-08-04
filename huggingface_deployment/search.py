"""
Vector Database Module for FMCG Promotion Copy Re-use Recommender

This module handles embeddings generation and semantic search for promotional headlines.
It can be called as an autonomous agent or task for orchestration.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Vector database for semantic search of promotional headlines.
    
    This class can be used as an autonomous agent to create and query
    vector embeddings for promotional data.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        """
        Initialize the VectorDatabase.
        
        Args:
            model_name: Name of the sentence transformer model to use
            persist_directory: Directory to persist ChromaDB data
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.is_initialized = False
        
    def initialize(self):
        """
        Initialize the embedding model and ChromaDB client.
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            
            logger.info(f"Initializing ChromaDB client at: {self.persist_directory}")
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Create or get collection
            try:
                self.collection = self.client.get_collection("promo_headlines")
                logger.info("Using existing collection: promo_headlines")
            except Exception as e:
                try:
                    self.collection = self.client.create_collection("promo_headlines")
                    logger.info("Created new collection: promo_headlines")
                except Exception as e2:
                    # If creation fails, try to get it again
                    self.collection = self.client.get_collection("promo_headlines")
                    logger.info("Retrieved collection after creation attempt")
            
            self.is_initialized = True
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise
    
    def create_embeddings(self, headlines: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of headlines.
        
        Args:
            headlines: List of promotional headlines
            
        Returns:
            Array of embeddings
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Creating embeddings for {len(headlines)} headlines")
        embeddings = self.embedding_model.encode(headlines, show_progress_bar=True)
        return embeddings
    
    def add_promos_to_db(self, promos_df: pd.DataFrame) -> Dict:
        """
        Add promotional data to the vector database.
        
        Args:
            promos_df: DataFrame with promotional data
            
        Returns:
            Dictionary with operation results
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Check if collection is empty
            collection_count = self.collection.count()
            if collection_count > 0:
                logger.info(f"Collection already contains {collection_count} documents")
                return {
                    'status': 'info',
                    'message': f'Collection already contains {collection_count} documents. Use clear_collection() to reset.',
                    'documents_count': collection_count
                }
            
            # Prepare data for ChromaDB
            headlines = promos_df['headline'].tolist()
            promo_ids = promos_df['promo_id'].tolist()
            
            # Create metadata with enhanced fields
            metadata_list = []
            for _, row in promos_df.iterrows():
                metadata = {
                    'promo_id': row['promo_id'],
                    'touchpoint': row['touchpoint'],
                    'category': row['category'],
                    'brand': row['brand'],
                    'brand_tier': row.get('brand_tier', 'mass'),
                    'sku': row['sku'],
                    'period': row['period'],
                    'kpi_lift': str(row['kpi_lift']),
                    'kpi_roi': str(row['kpi_roi']),
                    'ctr': str(row.get('ctr', 0.0)),
                    'conversion_rate': str(row.get('conversion_rate', 0.0)),
                    'engagement_rate': str(row.get('engagement_rate', 0.0)),
                    'roas': str(row.get('roas', 0.0)),
                    'promo_type': row.get('promo_type', 'other'),
                    'campaign_objective': row.get('campaign_objective', 'awareness'),
                    'target_age_group': row.get('target_age_group', '25-35'),
                    'target_region': row.get('target_region', 'Jakarta'),
                    'description': row.get('description', '')
                }
                metadata_list.append(metadata)
            
            # Create embeddings
            embeddings = self.create_embeddings(headlines)
            
            # Add to collection
            logger.info("Adding documents to ChromaDB collection...")
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=headlines,
                metadatas=metadata_list,
                ids=promo_ids
            )
            
            final_count = self.collection.count()
            logger.info(f"Successfully added {final_count} documents to vector database")
            
            return {
                'status': 'success',
                'message': f'Successfully added {final_count} promotional records to vector database',
                'documents_count': final_count
            }
            
        except Exception as e:
            logger.error(f"Error adding promos to database: {e}")
            return {
                'status': 'error',
                'message': f'Error adding promos to database: {str(e)}'
            }
    
    def search_similar_promos(self, query: str, n_results: int = 10, 
                            category_filter: Optional[str] = None,
                            touchpoint_filter: Optional[str] = None,
                            min_kpi_lift: Optional[float] = None,
                            min_kpi_roi: Optional[float] = None) -> List[Dict]:
        """
        Search for similar promotional headlines.
        
        Args:
            query: Search query (promotional headline)
            n_results: Number of results to return
            category_filter: Filter by category
            touchpoint_filter: Filter by touchpoint
            min_kpi_lift: Minimum KPI lift filter
            min_kpi_roi: Minimum KPI ROI filter
            
        Returns:
            List of similar promotional records with metadata
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Create where clause for filtering
            where_clause = {}
            if category_filter and category_filter != "All":
                where_clause['category'] = category_filter
            if touchpoint_filter and touchpoint_filter != "All":
                where_clause['touchpoint'] = touchpoint_filter
            
            # Search in vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Get more results for filtering
                where=where_clause if where_clause else None
            )
            
            # Process results with enhanced fields
            similar_promos = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                promo_data = {
                    'promo_id': metadata['promo_id'],
                    'headline': results['documents'][0][i],
                    'touchpoint': metadata['touchpoint'],
                    'category': metadata['category'],
                    'brand': metadata['brand'],
                    'brand_tier': metadata.get('brand_tier', 'mass'),
                    'sku': metadata['sku'],
                    'period': metadata['period'],
                    'kpi_lift': float(metadata['kpi_lift']),
                    'kpi_roi': float(metadata['kpi_roi']),
                    'ctr': float(metadata.get('ctr', 0.0)),
                    'conversion_rate': float(metadata.get('conversion_rate', 0.0)),
                    'engagement_rate': float(metadata.get('engagement_rate', 0.0)),
                    'roas': float(metadata.get('roas', 0.0)),
                    'promo_type': metadata.get('promo_type', 'other'),
                    'campaign_objective': metadata.get('campaign_objective', 'awareness'),
                    'target_age_group': metadata.get('target_age_group', '25-35'),
                    'target_region': metadata.get('target_region', 'Jakarta'),
                    'description': metadata['description'],
                    'similarity_score': results['distances'][0][i]
                }
                
                # Apply KPI filters
                if min_kpi_lift is not None and promo_data['kpi_lift'] < min_kpi_lift:
                    continue
                if min_kpi_roi is not None and promo_data['kpi_roi'] < min_kpi_roi:
                    continue
                
                similar_promos.append(promo_data)
                
                # Stop if we have enough results
                if len(similar_promos) >= n_results:
                    break
            
            logger.info(f"Found {len(similar_promos)} similar promos for query: {query}")
            return similar_promos
            
        except Exception as e:
            logger.error(f"Error searching similar promos: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector database collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            count = self.collection.count()
            
            # Get sample documents for metadata analysis
            sample_results = self.collection.query(
                query_texts=["sample"],
                n_results=min(count, 100)
            )
            
            # Analyze metadata
            categories = set()
            touchpoints = set()
            brands = set()
            
            for metadata in sample_results['metadatas'][0]:
                categories.add(metadata['category'])
                touchpoints.add(metadata['touchpoint'])
                brands.add(metadata['brand'])
            
            return {
                'total_documents': count,
                'unique_categories': len(categories),
                'unique_touchpoints': len(touchpoints),
                'unique_brands': len(brands),
                'categories': list(categories),
                'touchpoints': list(touchpoints),
                'brands': list(brands)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> Dict:
        """
        Clear all documents from the collection.
        
        Returns:
            Dictionary with operation results
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            self.client.delete_collection("promo_headlines")
            self.collection = self.client.create_collection("promo_headlines")
            
            logger.info("Collection cleared successfully")
            return {
                'status': 'success',
                'message': 'Collection cleared successfully'
            }
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return {
                'status': 'error',
                'message': f'Error clearing collection: {str(e)}'
            }

# Agent-compatible functions for orchestration
def initialize_vector_database(model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db") -> Dict:
    """
    Agent function to initialize the vector database.
    
    Args:
        model_name: Name of the sentence transformer model
        persist_directory: Directory to persist ChromaDB data
        
    Returns:
        Dictionary with initialization results
    """
    try:
        vector_db = VectorDatabase(model_name=model_name, persist_directory=persist_directory)
        vector_db.initialize()
        
        stats = vector_db.get_collection_stats()
        
        return {
            'status': 'success',
            'message': 'Vector database initialized successfully',
            'model_name': model_name,
            'collection_stats': stats
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error initializing vector database: {str(e)}'
        }

def add_promos_to_vector_db(promos_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> Dict:
    """
    Agent function to add promotional data to vector database.
    
    Args:
        promos_df: DataFrame with promotional data
        model_name: Name of the sentence transformer model
        
    Returns:
        Dictionary with operation results
    """
    vector_db = VectorDatabase(model_name=model_name)
    return vector_db.add_promos_to_db(promos_df)

def search_similar_promos_agent(query: str, n_results: int = 10, 
                              category_filter: str = None,
                              touchpoint_filter: str = None,
                              min_kpi_lift: float = None,
                              min_kpi_roi: float = None) -> Dict:
    """
    Agent function to search for similar promotional headlines.
    
    Args:
        query: Search query
        n_results: Number of results to return
        category_filter: Filter by category
        touchpoint_filter: Filter by touchpoint
        min_kpi_lift: Minimum KPI lift filter
        min_kpi_roi: Minimum KPI ROI filter
        
    Returns:
        Dictionary with search results
    """
    vector_db = VectorDatabase()
    similar_promos = vector_db.search_similar_promos(
        query=query,
        n_results=n_results,
        category_filter=category_filter,
        touchpoint_filter=touchpoint_filter,
        min_kpi_lift=min_kpi_lift,
        min_kpi_roi=min_kpi_roi
    )
    
    return {
        'status': 'success',
        'query': query,
        'results_count': len(similar_promos),
        'similar_promos': similar_promos
    }

def get_vector_db_stats() -> Dict:
    """
    Agent function to get vector database statistics.
    
    Returns:
        Dictionary with collection statistics
    """
    vector_db = VectorDatabase()
    stats = vector_db.get_collection_stats()
    
    return {
        'status': 'success',
        'collection_stats': stats
    }

if __name__ == "__main__":
    # Example usage
    vector_db = VectorDatabase()
    vector_db.initialize()
    
    # Load sample data
    from data import DataCleaner
    cleaner = DataCleaner()
    promos_df, skus_df = cleaner.clean_all_data("data/promos.csv", "data/sku_master.csv")
    
    # Add to vector database
    result = vector_db.add_promos_to_db(promos_df)
    print(result)
    
    # Search for similar promos
    similar = vector_db.search_similar_promos("BELI 2 GRATIS 1", n_results=5)
    print(f"\nFound {len(similar)} similar promos")
    
    for promo in similar[:3]:
        print(f"- {promo['headline']} (Score: {promo['similarity_score']:.3f})") 