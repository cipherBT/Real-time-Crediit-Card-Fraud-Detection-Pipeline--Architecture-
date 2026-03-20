"""
Credit Card Fraud Detection - Complete Workflow Runner

This script orchestrates the complete workflow:
1. Preprocess data (clean, engineer features, EDA)
2. Train model on processed_train.csv
3. Evaluate model on processed_test.csv
4. (Optional) Stream test data through producer for real-time simulation

Usage:
    python run_workflow.py --step all       # Run complete workflow
    python run_workflow.py --step preprocess
    python run_workflow.py --step train
    python run_workflow.py --step evaluate
    python run_workflow.py --step stream
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
DAGS_DIR = PROJECT_ROOT / "dags"
PRODUCER_DIR = PROJECT_ROOT / "producer"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = PROJECT_ROOT / "FraudDataset"


def run_preprocessing():
    """Step 1: Preprocess raw data and generate EDA."""
    logger.info("="*60)
    logger.info("STEP 1: PREPROCESSING & EDA")
    logger.info("="*60)
    
    script = PREPROCESSING_DIR / "preprocess_data.py"
    
    if not script.exists():
        logger.error(f"Preprocessing script not found: {script}")
        return False
    
    cmd = [
        sys.executable, str(script),
        "--input-dir", str(RAW_DATA_DIR),
        "--output-dir", str(DATA_DIR)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        logger.error("Preprocessing failed!")
        return False
    
    # Verify outputs
    train_file = DATA_DIR / "processed_train.csv"
    test_file = DATA_DIR / "processed_test.csv"
    
    if not train_file.exists() or not test_file.exists():
        logger.error("Processed files not generated!")
        return False
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"  - Train data: {train_file}")
    logger.info(f"  - Test data: {test_file}")
    logger.info(f"  - EDA plots: {DATA_DIR / 'plots'}")
    
    return True


def run_training():
    """Step 2: Train model on processed_train.csv."""
    logger.info("="*60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("="*60)
    
    # Check if processed data exists
    train_file = DATA_DIR / "processed_train.csv"
    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        logger.error("Run preprocessing first: python run_workflow.py --step preprocess")
        return False
    
    script = DAGS_DIR / "fraud_detection_training.py"
    
    if not script.exists():
        logger.error(f"Training script not found: {script}")
        return False
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, str(script),
        "--csv-path", str(train_file)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        logger.error("Training failed!")
        return False
    
    # Verify model output
    model_file = MODELS_DIR / "fraud_detection_model.pkl"
    if not model_file.exists():
        logger.warning(f"Model file not found at {model_file}")
        logger.info("Model may be saved to /app/models/ (Docker path)")
    else:
        logger.info(f"Model saved to: {model_file}")
    
    logger.info("Training completed successfully!")
    return True


def run_evaluation():
    """Step 3: Evaluate model on processed_test.csv."""
    logger.info("="*60)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("="*60)
    
    # Check if test data exists
    test_file = DATA_DIR / "processed_test.csv"
    if not test_file.exists():
        logger.error(f"Test data not found: {test_file}")
        logger.error("Run preprocessing first: python run_workflow.py --step preprocess")
        return False
    
    # Check if model exists
    model_file = MODELS_DIR / "fraud_detection_model.pkl"
    if not model_file.exists():
        logger.error(f"Model not found: {model_file}")
        logger.error("Run training first: python run_workflow.py --step train")
        return False
    
    script = DAGS_DIR / "fraud_detection_training.py"
    
    if not script.exists():
        logger.error(f"Training script not found: {script}")
        return False
    
    output_dir = PROJECT_ROOT / "evaluation_results"
    
    cmd = [
        sys.executable, str(script),
        "--mode", "evaluate",
        "--test-csv", str(test_file),
        "--model-path", str(model_file),
        "--output-dir", str(output_dir),
        "--threshold", "0.5"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        logger.error("Evaluation failed!")
        return False
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    
    return True


def run_streaming(max_messages: int = 1000, interval: float = 0.1):
    """Step 4: Stream test data through producer."""
    logger.info("="*60)
    logger.info("STEP 4: REAL-TIME STREAMING SIMULATION")
    logger.info("="*60)
    
    # Check if test data exists
    test_file = DATA_DIR / "processed_test.csv"
    if not test_file.exists():
        logger.error(f"Test data not found: {test_file}")
        logger.error("Run preprocessing first: python run_workflow.py --step preprocess")
        return False
    
    script = PRODUCER_DIR / "main.py"
    
    if not script.exists():
        logger.error(f"Producer script not found: {script}")
        return False
    
    cmd = [
        sys.executable, str(script),
        "--csv-path", str(test_file),
        "--interval", str(interval),
        "--max-messages", str(max_messages)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"Streaming {max_messages} transactions at {interval}s intervals...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            logger.error("Streaming failed!")
            return False
    except KeyboardInterrupt:
        logger.info("Streaming stopped by user")
    
    logger.info("Streaming completed!")
    return True


def run_all():
    """Run complete workflow."""
    logger.info("="*60)
    logger.info("RUNNING COMPLETE FRAUD DETECTION WORKFLOW")
    logger.info("="*60)
    
    steps = [
        ("Preprocessing", run_preprocessing),
        ("Training", run_training),
        ("Evaluation", run_evaluation),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n>>> Starting: {step_name}")
        success = step_func()
        if not success:
            logger.error(f"Workflow failed at step: {step_name}")
            return False
        logger.info(f">>> Completed: {step_name}\n")
    
    logger.info("="*60)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("  1. Review evaluation results in ./evaluation_results/")
    logger.info("  2. Run streaming simulation: python run_workflow.py --step stream")
    logger.info("  3. Start full pipeline with Docker: docker-compose up")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Credit Card Fraud Detection Workflow Runner'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'preprocess', 'train', 'evaluate', 'stream'],
        default='all',
        help='Which step to run'
    )
    parser.add_argument(
        '--max-messages',
        type=int,
        default=1000,
        help='Max messages for streaming step'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=0.1,
        help='Interval between messages for streaming'
    )
    
    args = parser.parse_args()
    
    step_handlers = {
        'all': run_all,
        'preprocess': run_preprocessing,
        'train': run_training,
        'evaluate': run_evaluation,
        'stream': lambda: run_streaming(args.max_messages, args.interval)
    }
    
    success = step_handlers[args.step]()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
