"""
Credit Card Fraud Detection Model Training & Evaluation Pipeline

This implementation is a production-grade ML training system for credit card fraud detection
optimized for the processed credit card fraud dataset with features including:
- Transaction amount and log-transformed amount
- Geographic distance between user and merchant
- Temporal features (hour, day of week, month, weekend, night)
- Demographics (age, gender)
- Category and merchant information

Key Components:
1. Environment Agnostic Configuration
2. MLflow Experiment Tracking
3. Class Imbalance Mitigation with SMOTE
4. Hyperparameter Tuning with XGBoost
5. Model Evaluation on Test Data
"""

import json
import logging
import os
from datetime import datetime

import boto3
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from kafka import KafkaConsumer
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (make_scorer, fbeta_score, precision_recall_curve, 
                            average_precision_score, precision_score,
                            recall_score, f1_score, confusion_matrix,
                            classification_report, roc_curve, auc, accuracy_score)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Configure dual logging to file and stdout with structured format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CreditCardFraudTraining:
    """
    Credit card fraud detection training system implementing MLOps best practices.

    Key Architecture Components:
    - Configuration Management: Centralized YAML config with environment overrides
    - Data Ingestion: Kafka consumer OR CSV file loading
    - Feature Engineering: Distance, temporal, demographic, and categorical features
    - Model Development: XGBoost with SMOTE for class imbalance
    - Hyperparameter Tuning: Randomized search with stratified cross-validation
    - Model Tracking: MLflow integration with metrics/artifact logging
    - Deployment Prep: Model serialization and registry
    - Evaluation: Comprehensive model evaluation on test data
    """

    def __init__(self, config_path='/app/config.yaml'):
        # Environment hardening for containerized deployments
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git'

        # Load environment variables before config to allow overrides
        load_dotenv(dotenv_path='/app/.env')

        # Configuration lifecycle management
        self.config = self._load_config(config_path)

        # Security-conscious credential handling
        os.environ.update({
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
        })

        # Pre-flight system checks
        self._validate_environment()

        # MLflow configuration for experiment tracking
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        # Feature configuration
        self.feature_cols = [
            'amt', 'amt_log', 'distance_km', 'city_pop',
            'trans_hour', 'trans_day_of_week', 'trans_month',
            'is_weekend', 'is_night', 'gender', 'age',
            'category', 'merchant', 'state'
        ]
        
        self.categorical_cols = ['category', 'merchant', 'state', 'gender']
        self.numeric_cols = [c for c in self.feature_cols if c not in self.categorical_cols]
        
        # Store trained model reference
        self.model = None

    def _load_config(self, config_path: str) -> dict:
        """Load and validate hierarchical configuration with fail-fast semantics."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully')
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise

    def _validate_environment(self):
        """System integrity verification with defense-in-depth checks."""
        required_vars = ['KAFKA_BOOTSTRAP_SERVERS', 'KAFKA_USERNAME', 'KAFKA_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.warning(f'Missing Kafka environment variables: {missing}')
            logger.info('Will attempt to load from CSV instead')

        self._check_minio_connection()

    def _check_minio_connection(self):
        """Validate object storage connectivity and bucket configuration."""
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=self.config['mlflow']['s3_endpoint_url'],
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            logger.info('Minio connection verified. Buckets: %s', bucket_names)

            mlflow_bucket = self.config['mlflow'].get('bucket', 'mlflow')

            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket=mlflow_bucket)
                logger.info('Created missing MLFlow bucket: %s', mlflow_bucket)
        except Exception as e:
            logger.error('Minio connection failed: %s', str(e))

    def read_from_csv(self, csv_path: str = None) -> pd.DataFrame:
        """
        Read processed credit card fraud data from CSV file.
        
        Args:
            csv_path: Path to processed CSV file
            
        Returns:
            DataFrame with transaction data
        """
        if csv_path is None:
            csv_path = self.config.get('data', {}).get('train_path', 
                '/app/data/processed/processed_train.csv')
        
        logger.info(f'Loading data from CSV: {csv_path}')
        
        df = pd.read_csv(csv_path)
        
        if 'is_fraud' not in df.columns:
            raise ValueError('Fraud label (is_fraud) missing from CSV data')
        
        # Data quality monitoring point
        fraud_rate = df['is_fraud'].mean() * 100
        logger.info('CSV data loaded successfully: %d rows, fraud rate: %.2f%%', 
                   len(df), fraud_rate)
        
        return df

    def read_from_kafka(self) -> pd.DataFrame:
        """
        Secure Kafka consumer implementation with enterprise features.
        
        Returns:
            DataFrame with transaction data from Kafka
        """
        try:
            topic = self.config['kafka']['topic']
            logger.info('Connecting to kafka topic %s', topic)

            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config['kafka']['bootstrap_servers'].split(','),
                security_protocol='SASL_SSL',
                sasl_mechanism='PLAIN',
                sasl_plain_username=self.config['kafka']['username'],
                sasl_plain_password=self.config['kafka']['password'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='earliest',
                consumer_timeout_ms=self.config['kafka'].get('timeout', 10000)
            )

            messages = [msg.value for msg in consumer]
            consumer.close()

            df = pd.DataFrame(messages)
            if df.empty:
                raise ValueError('No messages received from Kafka.')

            if 'is_fraud' not in df.columns:
                raise ValueError('Fraud label (is_fraud) missing from Kafka data')

            # Data quality monitoring point
            fraud_rate = df['is_fraud'].mean() * 100
            logger.info('Kafka data read successfully with fraud rate: %.2f%%', fraud_rate)

            return df
        except Exception as e:
            logger.error('Failed to read data from Kafka: %s', str(e), exc_info=True)
            raise

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training.
        
        The preprocessing script should have already created most features.
        This method ensures all required columns exist and handles any edge cases.
        
        Args:
            df: DataFrame from CSV or Kafka
            
        Returns:
            DataFrame with prepared features
        """
        logger.info('Preparing features for training...')
        df = df.copy()
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                logger.warning(f'Missing column {col}, filling with default value')
                if col in self.categorical_cols:
                    df[col] = 'unknown'
                else:
                    df[col] = 0
        
        # Handle missing values
        for col in self.numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        for col in self.categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna('unknown')
        
        # Schema validation guard
        if 'is_fraud' not in df.columns:
            raise ValueError('Missing target column "is_fraud"')
        
        logger.info(f'Features prepared. Shape: {df.shape}')
        
        return df[self.feature_cols + ['is_fraud']]

    def train_model(self, csv_path: str = None, use_kafka: bool = False):
        """
        End-to-end training pipeline implementing ML best practices.
        
        Args:
            csv_path: Path to processed CSV file (if not using Kafka)
            use_kafka: Whether to read from Kafka instead of CSV
        """
        try:
            logger.info('Starting credit card fraud model training process')

            # Data ingestion
            if use_kafka:
                df = self.read_from_kafka()
            else:
                df = self.read_from_csv(csv_path)
            
            # Feature preparation
            data = self.prepare_features(df)

            # Prevent Docker Out-Of-Memory (OOM) by downsampling intelligently
            # We keep ALL fraud cases and only downsample the majority class (legitimate)
            if len(data) > 150000:
                logger.info(f"Applying smart undersampling to protect memory while maximizing fraud examples")
                fraud_data = data[data['is_fraud'] == 1]
                legit_data = data[data['is_fraud'] == 0]
                
                # Sample the majority class down to 150k max
                n_legit = min(len(legit_data), 150000)
                legit_sampled = legit_data.sample(n=n_legit, random_state=self.config['model'].get('seed', 42))
                
                # Recombine and shuffle
                data = pd.concat([fraud_data, legit_sampled]).sample(frac=1, random_state=self.config['model'].get('seed', 42)).reset_index(drop=True)
                logger.info(f"Downsampled dataset size: {len(data)} rows (Fraud cases retained: {len(fraud_data)})")

            # Train/Test split with stratification
            X = data.drop(columns=['is_fraud'])
            y = data['is_fraud']

            # Class imbalance safeguards
            if y.sum() == 0:
                raise ValueError('No positive samples in training data')
            if y.sum() < 10:
                logger.warning('Low positive samples: %d. Consider additional data augmentation', y.sum())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['model'].get('test_size', 0.2),
                stratify=y,
                random_state=self.config['model'].get('seed', 42)
            )

            # MLflow experiment tracking context
            with mlflow.start_run():
                # Dataset metadata logging
                mlflow.log_metrics({
                    'train_samples': X_train.shape[0],
                    'positive_samples': int(y_train.sum()),
                    'class_ratio': float(y_train.mean()),
                    'test_samples': X_test.shape[0]
                })

                # Categorical feature preprocessing
                preprocessor = ColumnTransformer([
                    ('cat_encoder', OrdinalEncoder(
                        handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32
                    ), self.categorical_cols),
                    ('num_scaler', StandardScaler(), self.numeric_cols)
                ], remainder='passthrough')

                # XGBoost configuration with efficiency optimizations
                xgb = XGBClassifier(
                    eval_metric='aucpr',  # Optimizes for precision-recall area
                    random_state=self.config['model'].get('seed', 42),
                    reg_lambda=1.0,
                    n_estimators=self.config['model']['params']['n_estimators'],
                    n_jobs=1,
                    tree_method=self.config['model']['params'].get('tree_method', 'hist')
                )

                # Imbalanced learning pipeline
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(sampling_strategy=0.5, random_state=self.config['model'].get('seed', 42))),
                    ('classifier', xgb)
                ], memory='./cache')

                # Hyperparameter search space design
                param_dist = {
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.05, 0.1],
                    'classifier__subsample': [0.6, 0.8, 1.0],
                    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                    'classifier__gamma': [0, 0.1, 0.3],
                    'classifier__reg_alpha': [0, 0.1, 0.5]
                }

                # Optimizing for F-beta score (beta=2 emphasizes recall)
                searcher = RandomizedSearchCV(
                    pipeline,
                    param_dist,
                    n_iter=20,
                    scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
                    cv=StratifiedKFold(n_splits=3, shuffle=True),
                    n_jobs=1,
                    refit=True,
                    error_score='raise',
                    random_state=self.config['model'].get('seed', 42)
                )

                logger.info('Starting hyperparameter tuning...')
                searcher.fit(X_train, y_train)
                best_model = searcher.best_estimator_
                best_params = searcher.best_params_
                logger.info('Best hyperparameters: %s', best_params)

                # Threshold optimization using training data
                train_proba = best_model.predict_proba(X_train)[:, 1]
                precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_train, train_proba)
                f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in
                             zip(precision_arr[:-1], recall_arr[:-1])]
                best_threshold = thresholds_arr[np.argmax(f1_scores)]
                logger.info('Optimal threshold determined: %.4f', best_threshold)

                # Model evaluation
                X_test_processed = best_model.named_steps['preprocessor'].transform(X_test)
                test_proba = best_model.named_steps['classifier'].predict_proba(X_test_processed)[:, 1]
                y_pred = (test_proba >= best_threshold).astype(int)

                # Comprehensive metrics suite
                metrics = {
                    'auc_pr': float(average_precision_score(y_test, test_proba)),
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                    'threshold': float(best_threshold)
                }

                mlflow.log_metrics(metrics)
                mlflow.log_params(best_params)

                # Confusion matrix visualization
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Credit Card Fraud Detection - Confusion Matrix', fontsize=14, fontweight='bold')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Legitimate', 'Fraud'])
                plt.yticks(tick_marks, ['Legitimate', 'Fraud'])

                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, format(cm[i, j], ',d'), ha='center', va='center', 
                                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)

                plt.ylabel('Actual', fontsize=12)
                plt.xlabel('Predicted', fontsize=12)
                plt.tight_layout()
                cm_filename = 'confusion_matrix.png'
                plt.savefig(cm_filename, dpi=150)
                mlflow.log_artifact(cm_filename)
                plt.close()

                # Precision-Recall curve documentation
                plt.figure(figsize=(10, 6))
                plt.plot(recall_arr, precision_arr, marker='.', label='Precision-Recall Curve', linewidth=2)
                plt.axhline(y=metrics['precision'], color='r', linestyle='--', 
                           label=f'Precision @ threshold: {metrics["precision"]:.3f}')
                plt.axvline(x=metrics['recall'], color='g', linestyle='--',
                           label=f'Recall @ threshold: {metrics["recall"]:.3f}')
                plt.xlabel('Recall', fontsize=12)
                plt.ylabel('Precision', fontsize=12)
                plt.title('Credit Card Fraud Detection - Precision-Recall Curve', fontsize=14, fontweight='bold')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                pr_filename = 'precision_recall_curve.png'
                plt.savefig(pr_filename, dpi=150)
                mlflow.log_artifact(pr_filename)
                plt.close()

                # Model packaging and registry
                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path='model',
                    signature=signature,
                    registered_model_name='credit_card_fraud_detection_model'
                )

                # Model serialization for deployment
                os.makedirs('/app/models', exist_ok=True)
                joblib.dump(best_model, '/app/models/fraud_detection_model.pkl')
                
                # Store model reference for evaluation
                self.model = best_model

                logger.info('Training successfully completed with metrics: %s', metrics)

                return best_model, metrics

        except Exception as e:
            logger.error('Training failed: %s', str(e), exc_info=True)
            raise

    def evaluate_model(self, test_csv_path: str, model_path: str = None, 
                       output_dir: str = './evaluation_results', threshold: float = 0.5) -> dict:
        """
        Evaluate trained model on holdout test dataset (fraudTest.csv).
        
        Args:
            test_csv_path: Path to processed test CSV file
            model_path: Path to trained model file (if loading from disk)
            output_dir: Directory for evaluation outputs
            threshold: Classification threshold for fraud detection
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("="*60)
        logger.info("STARTING MODEL EVALUATION ON TEST SET")
        logger.info("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model if not already in memory
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
        elif self.model is None:
            raise ValueError("No model available. Either train a model or provide model_path")
        
        # Load and prepare test data
        logger.info(f"Loading test data from {test_csv_path}")
        test_df = pd.read_csv(test_csv_path)
        
        # Prepare features
        test_data = self.prepare_features(test_df)
        X_test = test_data[self.feature_cols]
        y_test = test_data['is_fraud']
        
        logger.info(f"Test data: {len(X_test)} transactions, {y_test.sum()} frauds ({y_test.mean()*100:.2f}%)")
        
        # Generate predictions
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'accuracy': float(accuracy_score(y_test, predictions)),
            'precision': float(precision_score(y_test, predictions, zero_division=0)),
            'recall': float(recall_score(y_test, predictions, zero_division=0)),
            'f1_score': float(f1_score(y_test, predictions, zero_division=0)),
            'auc_pr': float(average_precision_score(y_test, probabilities)),
            'total_transactions': len(y_test),
            'actual_frauds': int(y_test.sum()),
            'predicted_frauds': int(predictions.sum()),
            'true_positives': int(((predictions == 1) & (y_test == 1)).sum()),
            'false_positives': int(((predictions == 1) & (y_test == 0)).sum()),
            'true_negatives': int(((predictions == 0) & (y_test == 0)).sum()),
            'false_negatives': int(((predictions == 0) & (y_test == 1)).sum())
        }
        metrics['fraud_detection_rate'] = metrics['true_positives'] / max(metrics['actual_frauds'], 1)
        
        # Print classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, predictions, target_names=['Legitimate', 'Fraud']))
        
        # Generate visualizations
        self._plot_test_confusion_matrix(y_test, predictions, output_dir)
        self._plot_test_pr_curve(y_test, probabilities, predictions, output_dir)
        self._plot_test_roc_curve(y_test, probabilities, output_dir)
        self._plot_threshold_analysis(y_test, probabilities, output_dir)
        
        # Category analysis
        if 'category' in test_df.columns:
            self._analyze_by_category(test_df, predictions, probabilities, output_dir)
        
        # Generate text report
        self._generate_evaluation_report(metrics, output_dir)
        
        logger.info("="*60)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
        return metrics
    
    def _plot_test_confusion_matrix(self, y_test, predictions, output_dir):
        """Generate confusion matrix visualization for test data."""
        cm = confusion_matrix(y_test, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_normalized[i, j] * 100
                color = 'white' if count > cm.max()/2 else 'black'
                plt.text(j + 0.5, i + 0.5, f'{count:,}\n({pct:.1f}%)',
                        ha='center', va='center', color=color, fontsize=14)
        
        plt.title('Confusion Matrix - Test Set Evaluation', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.xticks([0.5, 1.5], ['Legitimate', 'Fraud'])
        plt.yticks([0.5, 1.5], ['Legitimate', 'Fraud'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_confusion_matrix.png'), dpi=150)
        plt.close()
        logger.info("Saved confusion matrix")
    
    def _plot_test_pr_curve(self, y_test, probabilities, predictions, output_dir):
        """Generate precision-recall curve for test data."""
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, probabilities)
        avg_precision = average_precision_score(y_test, probabilities)
        
        current_precision = precision_score(y_test, predictions, zero_division=0)
        current_recall = recall_score(y_test, predictions, zero_division=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall_arr, precision_arr, 'b-', linewidth=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.fill_between(recall_arr, precision_arr, alpha=0.2)
        plt.scatter([current_recall], [current_precision], color='red', s=100, 
                   zorder=5, label=f'Current (P={current_precision:.2f}, R={current_recall:.2f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_precision_recall_curve.png'), dpi=150)
        plt.close()
        logger.info("Saved PR curve")
    
    def _plot_test_roc_curve(self, y_test, probabilities, output_dir):
        """Generate ROC curve for test data."""
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.fill_between(fpr, tpr, alpha=0.2)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_roc_curve.png'), dpi=150)
        plt.close()
        logger.info("Saved ROC curve")
    
    def _plot_threshold_analysis(self, y_test, probabilities, output_dir):
        """Analyze model performance across different thresholds."""
        thresholds = np.arange(0.1, 1.0, 0.05)
        precisions, recalls, f1_scores_list = [], [], []
        
        for thresh in thresholds:
            preds = (probabilities >= thresh).astype(int)
            precisions.append(precision_score(y_test, preds, zero_division=0))
            recalls.append(recall_score(y_test, preds, zero_division=0))
            f1_scores_list.append(f1_score(y_test, preds, zero_division=0))
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision', marker='o')
        plt.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall', marker='s')
        plt.plot(thresholds, f1_scores_list, 'r-', linewidth=2, label='F1 Score', marker='^')
        
        best_idx = np.argmax(f1_scores_list)
        best_thresh = thresholds[best_idx]
        plt.axvline(x=best_thresh, color='purple', linestyle='--', 
                   label=f'Best F1 threshold: {best_thresh:.2f}')
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Threshold Analysis - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_threshold_analysis.png'), dpi=150)
        plt.close()
        logger.info("Saved threshold analysis")
    
    def _analyze_by_category(self, test_df, predictions, probabilities, output_dir):
        """Analyze fraud detection performance by transaction category."""
        results = test_df.copy()
        results['predicted'] = predictions
        results['probability'] = probabilities
        
        category_analysis = results.groupby('category').agg({
            'is_fraud': ['count', 'sum', 'mean'],
            'predicted': ['sum', 'mean'],
            'probability': 'mean'
        }).round(4)
        
        category_analysis.columns = [
            'total_txns', 'actual_frauds', 'actual_fraud_rate',
            'predicted_frauds', 'predicted_fraud_rate', 'avg_probability'
        ]
        
        category_analysis['detection_rate'] = (
            category_analysis['predicted_frauds'] / 
            category_analysis['actual_frauds'].clip(lower=1)
        ).round(4)
        
        category_analysis = category_analysis.sort_values('actual_fraud_rate', ascending=False)
        category_analysis.to_csv(os.path.join(output_dir, 'category_analysis.csv'))
        
        print("\nFraud Detection by Category (Top 10):")
        print(category_analysis.head(10).to_string())
        logger.info("Saved category analysis")
    
    def _generate_evaluation_report(self, metrics, output_dir):
        """Generate a text summary report."""
        report = [
            "="*60,
            "FRAUD DETECTION MODEL - TEST SET EVALUATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*60,
            "",
            "MODEL PERFORMANCE SUMMARY",
            "-"*40,
            f"Threshold:          {metrics['threshold']:.2f}",
            f"Accuracy:           {metrics['accuracy']:.4f}",
            f"Precision:          {metrics['precision']:.4f}",
            f"Recall:             {metrics['recall']:.4f}",
            f"F1 Score:           {metrics['f1_score']:.4f}",
            f"AUC-PR:             {metrics['auc_pr']:.4f}",
            "",
            "FRAUD DETECTION STATISTICS",
            "-"*40,
            f"Total Transactions: {metrics['total_transactions']:,}",
            f"Actual Frauds:      {metrics['actual_frauds']:,}",
            f"Predicted Frauds:   {metrics['predicted_frauds']:,}",
            f"Detection Rate:     {metrics['fraud_detection_rate']*100:.1f}%",
            "",
            "CONFUSION MATRIX BREAKDOWN",
            "-"*40,
            f"True Positives:     {metrics['true_positives']:,}",
            f"False Positives:    {metrics['false_positives']:,}",
            f"True Negatives:     {metrics['true_negatives']:,}",
            f"False Negatives:    {metrics['false_negatives']:,}",
            "="*60
        ]
        
        report_text = "\n".join(report)
        
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report_text)
        
        print(report_text)
        logger.info("Saved evaluation report")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Training & Evaluation')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], default='train',
                       help='Mode: train, evaluate, or both')
    parser.add_argument('--train-csv', type=str, default=None,
                       help='Path to processed training CSV file')
    parser.add_argument('--test-csv', type=str, default=None,
                       help='Path to processed test CSV file')
    parser.add_argument('--model-path', type=str, default='./models/fraud_detection_model.pkl',
                       help='Path to model file (for loading/saving)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Directory for evaluation outputs')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold for evaluation')
    parser.add_argument('--use-kafka', action='store_true',
                       help='Read training data from Kafka instead of CSV')
    
    args = parser.parse_args()
    
    trainer = CreditCardFraudTraining('/app/config.yaml')
    
    if args.mode in ['train', 'both']:
        trainer.train_model(csv_path=args.train_csv, use_kafka=args.use_kafka)
    
    if args.mode in ['evaluate', 'both']:
        if args.test_csv is None:
            print("Error: --test-csv is required for evaluation mode")
        else:
            model_path = args.model_path if args.mode == 'evaluate' else None
            trainer.evaluate_model(
                test_csv_path=args.test_csv,
                model_path=model_path,
                output_dir=args.output_dir,
                threshold=args.threshold
            )