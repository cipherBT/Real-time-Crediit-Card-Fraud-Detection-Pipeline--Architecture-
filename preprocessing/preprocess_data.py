"""
Credit Card Fraud Dataset Preprocessing and EDA

This script performs data cleaning, feature engineering, and exploratory data analysis
on the credit card fraud dataset before it's used in the fraud detection pipeline.

Usage:
    python preprocess_data.py --input-dir /path/to/raw/data --output-dir /path/to/output
"""

import os
import argparse
import logging
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in km).
    
    Args:
        lat1, lon1: Latitude and longitude of point 1
        lat2, lon2: Latitude and longitude of point 2
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def calculate_age(dob: str, reference_date: datetime = None) -> int:
    """
    Calculate age from date of birth.
    
    Args:
        dob: Date of birth string (YYYY-MM-DD format)
        reference_date: Reference date for age calculation (default: 2020-01-01)
    
    Returns:
        Age in years
    """
    if reference_date is None:
        reference_date = datetime(2020, 1, 1)
    
    try:
        birth_date = pd.to_datetime(dob)
        age = (reference_date - birth_date).days // 365
        return max(0, min(age, 120))  # Clamp to reasonable range
    except Exception:
        return 0


class FraudDataPreprocessor:
    """
    Preprocessor for credit card fraud dataset.
    
    Handles:
    - Data loading and inspection
    - Cleaning and validation
    - Feature engineering
    - EDA visualization
    - Export of processed data
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize preprocessor with input and output directories.
        
        Args:
            input_dir: Directory containing raw CSV files
            output_dir: Directory for processed output files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train_df = None
        self.test_df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        logger.info(f"Preprocessor initialized. Input: {input_dir}, Output: {output_dir}")
    
    def load_data(self) -> None:
        """Load train and test CSV files."""
        logger.info("Loading datasets...")
        
        train_path = os.path.join(self.input_dir, 'fraudTrain.csv')
        test_path = os.path.join(self.input_dir, 'fraudTest.csv')
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        logger.info(f"Train shape: {self.train_df.shape}")
        logger.info(f"Test shape: {self.test_df.shape}")
    
    def inspect_data(self) -> dict:
        """
        Perform initial data inspection and return summary statistics.
        
        Returns:
            Dictionary with inspection results
        """
        logger.info("Inspecting data...")
        
        inspection = {
            'train_shape': self.train_df.shape,
            'test_shape': self.test_df.shape,
            'train_columns': list(self.train_df.columns),
            'train_dtypes': self.train_df.dtypes.to_dict(),
            'train_missing': self.train_df.isnull().sum().to_dict(),
            'train_duplicates': self.train_df.duplicated().sum(),
            'test_duplicates': self.test_df.duplicated().sum(),
            'train_fraud_rate': self.train_df['is_fraud'].mean() * 100,
            'test_fraud_rate': self.test_df['is_fraud'].mean() * 100
        }
        
        # Print summary
        print("\n" + "="*60)
        print("DATA INSPECTION SUMMARY")
        print("="*60)
        print(f"\nTrain Shape: {inspection['train_shape']}")
        print(f"Test Shape: {inspection['test_shape']}")
        print(f"\nTrain Columns ({len(inspection['train_columns'])}):")
        for col in inspection['train_columns']:
            print(f"  - {col}: {inspection['train_dtypes'].get(col, 'unknown')}")
        print(f"\nMissing Values (Train):")
        for col, count in inspection['train_missing'].items():
            if count > 0:
                print(f"  - {col}: {count}")
        if sum(inspection['train_missing'].values()) == 0:
            print("  None")
        print(f"\nDuplicates: Train={inspection['train_duplicates']}, Test={inspection['test_duplicates']}")
        print(f"\nFraud Rate: Train={inspection['train_fraud_rate']:.2f}%, Test={inspection['test_fraud_rate']:.2f}%")
        print("="*60 + "\n")
        
        return inspection
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the dataset.
        
        Steps:
        1. Remove index column (Unnamed: 0 or #)
        2. Handle missing values
        3. Convert datetime columns
        4. Remove duplicates
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        df = df.copy()
        
        # 1. Remove index column (could be 'Unnamed: 0' or '#')
        index_cols = [col for col in df.columns if 'unnamed' in col.lower() or col == '#']
        if index_cols:
            df = df.drop(columns=index_cols)
            logger.info(f"Removed index columns: {index_cols}")
        
        # 2. Convert transaction datetime
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        # 3. Convert date of birth
        df['dob'] = pd.to_datetime(df['dob'])
        
        # 4. Handle missing values (if any)
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            # Fill numeric with median, categorical with mode
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['float64', 'int64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
            logger.info(f"Filled {missing_before} missing values")
        
        # 5. Remove duplicates based on trans_num (unique transaction ID)
        duplicates = df.duplicated(subset=['trans_num']).sum()
        if duplicates > 0:
            df = df.drop_duplicates(subset=['trans_num'], keep='first')
            logger.info(f"Removed {duplicates} duplicate transactions")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features for fraud detection.
        
        Features:
        1. Distance between user and merchant (haversine)
        2. Age from date of birth
        3. Temporal features (hour, day_of_week, is_weekend, is_night)
        4. Amount-based features
        
        Args:
            df: Cleaned DataFrame
        
        Returns:
            DataFrame with new features
        """
        logger.info("Engineering features...")
        df = df.copy()
        
        # 1. Distance between user and merchant
        df['distance_km'] = df.apply(
            lambda row: haversine_distance(
                row['lat'], row['long'], 
                row['merch_lat'], row['merch_long']
            ), axis=1
        )
        
        # 2. Age calculation (using transaction date as reference)
        df['age'] = df.apply(
            lambda row: (row['trans_date_trans_time'] - row['dob']).days // 365,
            axis=1
        )
        df['age'] = df['age'].clip(lower=18, upper=100)  # Reasonable age range
        
        # 3. Temporal features
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df['is_weekend'] = (df['trans_day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['trans_hour'] >= 22) | (df['trans_hour'] < 6)).astype(int)
        
        # 4. Amount features
        df['amt_log'] = np.log1p(df['amt'])  # Log transform for skewed amounts
        
        # 5. Gender encoding (already 0/1 based on description, but verify)
        if df['gender'].dtype == 'object':
            df['gender'] = df['gender'].map({'F': 0, 'M': 1}).fillna(0).astype(int)
        
        logger.info(f"Engineered {6} new features")
        
        return df
    
    def generate_eda_plots(self, df: pd.DataFrame, prefix: str = 'train') -> None:
        """
        Generate EDA visualizations and save to output directory.
        
        Args:
            df: DataFrame to analyze
            prefix: Prefix for output filenames
        """
        logger.info(f"Generating EDA plots for {prefix}...")
        plots_dir = os.path.join(self.output_dir, 'plots')
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Fraud Distribution (Class Imbalance)
        fig, ax = plt.subplots(figsize=(8, 5))
        fraud_counts = df['is_fraud'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(['Legitimate', 'Fraud'], fraud_counts.values, color=colors)
        ax.set_title('Transaction Class Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        for bar, count in zip(bars, fraud_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{count:,}\n({count/len(df)*100:.2f}%)', 
                   ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{prefix}_fraud_distribution.png'), dpi=150)
        plt.close()
        
        # 2. Amount Distribution by Fraud Status
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df[df['is_fraud']==0]['amt'], bins=50, alpha=0.7, label='Legitimate', color='#2ecc71')
        axes[0].hist(df[df['is_fraud']==1]['amt'], bins=50, alpha=0.7, label='Fraud', color='#e74c3c')
        axes[0].set_title('Transaction Amount Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Amount ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].set_xlim(0, df['amt'].quantile(0.99))  # Exclude extreme outliers
        
        # Box plot
        df.boxplot(column='amt', by='is_fraud', ax=axes[1])
        axes[1].set_title('Amount by Fraud Status', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Is Fraud')
        axes[1].set_ylabel('Amount ($)')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{prefix}_amount_distribution.png'), dpi=150)
        plt.close()
        
        # 3. Fraud Rate by Category
        fig, ax = plt.subplots(figsize=(12, 6))
        category_fraud = df.groupby('category')['is_fraud'].agg(['sum', 'count'])
        category_fraud['rate'] = category_fraud['sum'] / category_fraud['count'] * 100
        category_fraud = category_fraud.sort_values('rate', ascending=True)
        
        colors = plt.cm.RdYlGn_r(category_fraud['rate'] / category_fraud['rate'].max())
        bars = ax.barh(category_fraud.index, category_fraud['rate'], color=colors)
        ax.set_title('Fraud Rate by Transaction Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fraud Rate (%)')
        for bar, rate in zip(bars, category_fraud['rate']):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{rate:.2f}%', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{prefix}_fraud_by_category.png'), dpi=150)
        plt.close()
        
        # 4. Fraud Rate by Hour
        fig, ax = plt.subplots(figsize=(12, 5))
        hourly_fraud = df.groupby('trans_hour')['is_fraud'].agg(['sum', 'count'])
        hourly_fraud['rate'] = hourly_fraud['sum'] / hourly_fraud['count'] * 100
        
        ax.bar(hourly_fraud.index, hourly_fraud['rate'], color='#3498db', alpha=0.8)
        ax.axhline(y=df['is_fraud'].mean()*100, color='red', linestyle='--', label=f'Average: {df["is_fraud"].mean()*100:.2f}%')
        ax.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Fraud Rate (%)')
        ax.set_xticks(range(24))
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{prefix}_fraud_by_hour.png'), dpi=150)
        plt.close()
        
        # 5. Distance Distribution
        if 'distance_km' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df[df['is_fraud']==0]['distance_km'], bins=50, alpha=0.7, 
                   label='Legitimate', color='#2ecc71', density=True)
            ax.hist(df[df['is_fraud']==1]['distance_km'], bins=50, alpha=0.7, 
                   label='Fraud', color='#e74c3c', density=True)
            ax.set_title('User-Merchant Distance Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.set_xlim(0, df['distance_km'].quantile(0.99))
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{prefix}_distance_distribution.png'), dpi=150)
            plt.close()
        
        # 6. Correlation Heatmap (numeric features only)
        fig, ax = plt.subplots(figsize=(12, 10))
        numeric_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 
                       'is_fraud', 'distance_km', 'age', 'trans_hour', 'is_weekend', 'is_night']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        corr_matrix = df[numeric_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.2f', square=True, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{prefix}_correlation_matrix.png'), dpi=150)
        plt.close()
        
        logger.info(f"Saved 6 plots to {plots_dir}")
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and order features for model training.
        
        Removes non-predictive columns (names, street, etc.)
        
        Args:
            df: Featured DataFrame
        
        Returns:
            DataFrame with selected features
        """
        logger.info("Selecting features for model...")
        
        # Columns to keep for modeling
        feature_cols = [
            # Transaction identifiers (keep for tracking)
            'trans_num',
            'cc_num',
            
            # Transaction details
            'amt',
            'amt_log',
            'merchant',
            'category',
            
            # Location features
            'lat',
            'long',
            'merch_lat',
            'merch_long',
            'distance_km',
            'city_pop',
            'state',
            
            # Temporal features
            'trans_date_trans_time',
            'unix_time',
            'trans_hour',
            'trans_day_of_week',
            'trans_month',
            'is_weekend',
            'is_night',
            
            # Demographics
            'gender',
            'age',
            
            # Target
            'is_fraud'
        ]
        
        # Keep only columns that exist
        available_cols = [c for c in feature_cols if c in df.columns]
        
        # Columns being dropped
        dropped = set(df.columns) - set(available_cols)
        logger.info(f"Dropping {len(dropped)} columns: {dropped}")
        
        return df[available_cols]
    
    def process(self) -> tuple:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        logger.info("="*60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # 1. Load data
        self.load_data()
        
        # 2. Inspect
        self.inspect_data()
        
        # 3. Clean
        self.train_df = self.clean_data(self.train_df)
        self.test_df = self.clean_data(self.test_df)
        
        # 4. Engineer features
        self.train_df = self.engineer_features(self.train_df)
        self.test_df = self.engineer_features(self.test_df)
        
        # 5. Generate EDA plots (only for train)
        self.generate_eda_plots(self.train_df, prefix='train')
        
        # 6. Select features
        train_processed = self.select_features(self.train_df)
        test_processed = self.select_features(self.test_df)
        
        # 7. Save processed data
        train_output = os.path.join(self.output_dir, 'processed_train.csv')
        test_output = os.path.join(self.output_dir, 'processed_test.csv')
        
        train_processed.to_csv(train_output, index=False)
        test_processed.to_csv(test_output, index=False)
        
        logger.info(f"Saved processed train to: {train_output}")
        logger.info(f"Saved processed test to: {test_output}")
        
        # Print final summary
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"\nProcessed Train Shape: {train_processed.shape}")
        print(f"Processed Test Shape: {test_processed.shape}")
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"  - processed_train.csv")
        print(f"  - processed_test.csv")
        print(f"  - plots/ (6 EDA visualizations)")
        print("="*60 + "\n")
        
        return train_processed, test_processed


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(
        description='Preprocess credit card fraud dataset'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../FraudDataset',
        help='Directory containing raw CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/processed',
        help='Directory for processed output files'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocessor = FraudDataPreprocessor(args.input_dir, args.output_dir)
    preprocessor.process()


if __name__ == '__main__':
    main()
