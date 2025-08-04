"""
Data Cleaning Module for FMCG Promotion Copy Re-use Recommender

This module handles data validation, filtering, and preprocessing for promotional data.
It can be called as an autonomous agent or task for orchestration.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Data cleaning and preprocessing class for promotional data.
    
    This class can be used as an autonomous agent to clean and validate
    promotional data from CSV files.
    """
    
    def __init__(self, min_headline_length: int = 10):
        """
        Initialize the DataCleaner.
        
        Args:
            min_headline_length: Minimum length for valid headlines
        """
        self.min_headline_length = min_headline_length
        self.cleaned_promos = None
        self.cleaned_skus = None
        
    def load_data(self, promos_path: str, skus_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load promotional and SKU data from CSV files.
        
        Args:
            promos_path: Path to promos.csv file
            skus_path: Path to sku_master.csv file
            
        Returns:
            Tuple of (promos_df, skus_df)
        """
        try:
            promos_df = pd.read_csv(promos_path)
            skus_df = pd.read_csv(skus_path)
            logger.info(f"Loaded {len(promos_df)} promotional records and {len(skus_df)} SKU records")
            return promos_df, skus_df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_promos(self, promos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean promotional data.
        
        Args:
            promos_df: Raw promotional data
            
        Returns:
            Cleaned promotional data
        """
        logger.info("Starting promotional data validation...")
        
        # Create a copy to avoid modifying original data
        df = promos_df.copy()
        
        # Remove rows with missing headlines
        initial_count = len(df)
        df = df.dropna(subset=['headline'])
        logger.info(f"Removed {initial_count - len(df)} rows with missing headlines")
        
        # Remove headlines that are too short
        df = df[df['headline'].str.len() >= self.min_headline_length]
        logger.info(f"Removed {initial_count - len(df)} rows with short headlines")
        
        # Clean headline text
        df['headline'] = df['headline'].str.strip()
        df['headline'] = df['headline'].str.replace('\s+', ' ', regex=True)
        
        # Validate KPI values
        df['kpi_lift'] = pd.to_numeric(df['kpi_lift'], errors='coerce')
        df['kpi_roi'] = pd.to_numeric(df['kpi_roi'], errors='coerce')
        
        # Remove rows with invalid KPI values
        df = df.dropna(subset=['kpi_lift', 'kpi_roi'])
        logger.info(f"Removed {initial_count - len(df)} rows with invalid KPI values")
        
        # Ensure required columns exist
        required_columns = ['promo_id', 'headline', 'touchpoint', 'category', 'brand', 'sku', 'period']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Fill missing descriptions with empty string
        if 'description' in df.columns:
            df['description'] = df['description'].fillna('')
        else:
            df['description'] = ''
        
        logger.info(f"Validation complete. Final dataset has {len(df)} records")
        return df
    
    def validate_skus(self, skus_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean SKU data.
        
        Args:
            skus_df: Raw SKU data
            
        Returns:
            Cleaned SKU data
        """
        logger.info("Starting SKU data validation...")
        
        # Create a copy to avoid modifying original data
        df = skus_df.copy()
        
        # Remove rows with missing required fields
        initial_count = len(df)
        df = df.dropna(subset=['sku', 'brand', 'category', 'product_name'])
        logger.info(f"Removed {initial_count - len(df)} rows with missing required fields")
        
        # Clean text fields
        text_columns = ['sku', 'brand', 'category', 'product_name', 'pack_size']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['sku'])
        logger.info(f"Removed {initial_count - len(df)} duplicate SKU records")
        
        logger.info(f"SKU validation complete. Final dataset has {len(df)} records")
        return df
    
    def get_category_stats(self, promos_df: pd.DataFrame) -> Dict:
        """
        Get statistics by category for cleaned promotional data.
        
        Args:
            promos_df: Cleaned promotional data
            
        Returns:
            Dictionary with category statistics
        """
        stats = {}
        
        # Category distribution
        category_counts = promos_df['category'].value_counts()
        stats['category_distribution'] = category_counts.to_dict()
        
        # Average KPI by category
        avg_kpi_by_category = promos_df.groupby('category').agg({
            'kpi_lift': 'mean',
            'kpi_roi': 'mean'
        }).round(3)
        stats['avg_kpi_by_category'] = avg_kpi_by_category.to_dict()
        
        # Touchpoint distribution
        touchpoint_counts = promos_df['touchpoint'].value_counts()
        stats['touchpoint_distribution'] = touchpoint_counts.to_dict()
        
        return stats
    
    def clean_all_data(self, promos_path: str, skus_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data cleaning pipeline.
        
        Args:
            promos_path: Path to promos.csv file
            skus_path: Path to sku_master.csv file
            
        Returns:
            Tuple of (cleaned_promos_df, cleaned_skus_df)
        """
        logger.info("Starting complete data cleaning pipeline...")
        
        # Load data
        promos_df, skus_df = self.load_data(promos_path, skus_path)
        
        # Clean data
        self.cleaned_promos = self.validate_promos(promos_df)
        self.cleaned_skus = self.validate_skus(skus_df)
        
        # Log statistics
        stats = self.get_category_stats(self.cleaned_promos)
        logger.info(f"Category distribution: {stats['category_distribution']}")
        logger.info(f"Touchpoint distribution: {stats['touchpoint_distribution']}")
        
        return self.cleaned_promos, self.cleaned_skus
    
    def get_cleaned_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get the cleaned data if available.
        
        Returns:
            Tuple of (cleaned_promos_df, cleaned_skus_df) or (None, None)
        """
        return self.cleaned_promos, self.cleaned_skus

# Agent-compatible functions for orchestration
def clean_promotional_data(promos_path: str, skus_path: str, min_headline_length: int = 10) -> Dict:
    """
    Agent function to clean promotional data.
    
    Args:
        promos_path: Path to promos.csv file
        skus_path: Path to sku_master.csv file
        min_headline_length: Minimum length for valid headlines
        
    Returns:
        Dictionary with cleaning results and statistics
    """
    cleaner = DataCleaner(min_headline_length=min_headline_length)
    promos_df, skus_df = cleaner.clean_all_data(promos_path, skus_path)
    
    stats = cleaner.get_category_stats(promos_df)
    
    return {
        'status': 'success',
        'cleaned_promos_count': len(promos_df),
        'cleaned_skus_count': len(skus_df),
        'statistics': stats,
        'message': f"Successfully cleaned {len(promos_df)} promotional records and {len(skus_df)} SKU records"
    }

def validate_single_promo(headline: str, category: str, brand: str, kpi_lift: float, kpi_roi: float) -> Dict:
    """
    Agent function to validate a single promotional record.
    
    Args:
        headline: Promotional headline
        category: Product category
        brand: Brand name
        kpi_lift: Sales lift KPI
        kpi_roi: ROI KPI
        
    Returns:
        Dictionary with validation results
    """
    cleaner = DataCleaner()
    
    # Create a temporary DataFrame for validation
    temp_df = pd.DataFrame([{
        'promo_id': 'TEMP001',
        'headline': headline,
        'touchpoint': 'digital',
        'category': category,
        'brand': brand,
        'sku': 'TEMP_SKU',
        'period': '2024-Q1',
        'kpi_lift': kpi_lift,
        'kpi_roi': kpi_roi,
        'description': ''
    }])
    
    try:
        validated_df = cleaner.validate_promos(temp_df)
        is_valid = len(validated_df) > 0
        
        return {
            'status': 'success',
            'is_valid': is_valid,
            'message': 'Promotional record is valid' if is_valid else 'Promotional record is invalid',
            'validation_details': {
                'headline_length': len(headline),
                'min_required_length': cleaner.min_headline_length,
                'kpi_lift_valid': not pd.isna(kpi_lift),
                'kpi_roi_valid': not pd.isna(kpi_roi)
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'is_valid': False,
            'message': f'Validation error: {str(e)}'
        }

if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner()
    promos_df, skus_df = cleaner.clean_all_data("data/promos.csv", "data/sku_master.csv")
    
    print(f"Cleaned {len(promos_df)} promotional records")
    print(f"Cleaned {len(skus_df)} SKU records")
    
    # Print some statistics
    stats = cleaner.get_category_stats(promos_df)
    print("\nCategory Distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}") 