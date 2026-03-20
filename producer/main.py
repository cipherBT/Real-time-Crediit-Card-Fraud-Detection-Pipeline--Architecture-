"""
Credit Card Fraud Transaction Producer

This producer reads transactions from processed CSV files and streams them
to Kafka for real-time fraud detection processing.

Supports:
- Batch reading from CSV with configurable row limits
- Cycling through data for continuous streaming
- Credit card fraud dataset schema (22 columns)
"""

import json
import logging
import os
import time
import signal
from typing import Dict, Any, Optional, Iterator
from datetime import datetime

import pandas as pd
from confluent_kafka import Producer
from dotenv import load_dotenv
from jsonschema import validate, ValidationError, FormatChecker

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path="../.env")

# JSON Schema for credit card transaction validation
TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "trans_num": {"type": "string"},
        "cc_num": {"type": "number"},
        "amt": {"type": "number", "minimum": 0},
        "amt_log": {"type": "number"},
        "merchant": {"type": "string"},
        "category": {"type": "string"},
        "lat": {"type": "number"},
        "long": {"type": "number"},
        "merch_lat": {"type": "number"},
        "merch_long": {"type": "number"},
        "distance_km": {"type": "number"},
        "city_pop": {"type": "number"},
        "state": {"type": "string"},
        "trans_date_trans_time": {"type": "string"},
        "unix_time": {"type": "number"},
        "trans_hour": {"type": "integer"},
        "trans_day_of_week": {"type": "integer"},
        "trans_month": {"type": "integer"},
        "is_weekend": {"type": "integer"},
        "is_night": {"type": "integer"},
        "gender": {"type": "string"},
        "age": {"type": "integer"},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1}
    },
    "required": ["trans_num", "cc_num", "amt", "is_fraud"]
}


class CreditCardTransactionProducer:
    """
    Kafka producer that streams credit card transactions from CSV files dataset.
    
    Features:
    - Reads from processed CSV files
    - Validates transactions against schema
    - Supports cycling for continuous streaming
    - Configurable batch size and send interval
    """
    
    def __init__(self, csv_path: str = None):
        """
        Initialize producer with Kafka configuration and CSV data source.
        
        Args:
            csv_path: Path to processed CSV file. If None, reads from env/config.
        """
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_username = os.getenv("KAFKA_USERNAME")
        self.kafka_password = os.getenv("KAFKA_PASSWORD")
        self.topic = os.getenv("KAFKA_TOPIC", "transactions")
        self.running = False
        
        # CSV configuration
        self.csv_path = csv_path or os.getenv(
            "CSV_DATA_PATH", 
            "../data/processed/processed_train.csv"
        )
        self.cycle_data = os.getenv("CYCLE_DATA", "true").lower() == "true"
        self.current_index = 0
        self.data = None
        
        # Producer configuration for Confluent Kafka
        self.producer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": "credit-card-transaction-producer",
            "compression.type": "gzip",
            "linger.ms": 5,
            "batch.size": 16384,
        }

        if self.kafka_username and self.kafka_password:
            self.producer_config.update({
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "PLAIN",
                "sasl.username": self.kafka_username,
                "sasl.password": self.kafka_password,
            })
        else:
            self.producer_config["security.protocol"] = "PLAINTEXT"

        try:
            self.producer = Producer(self.producer_config)
            logger.info("Confluent Kafka Producer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Confluent Kafka Producer: {str(e)}")
            raise e
        
        # Load CSV data
        self._load_data()
        
        # Configure graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def _load_data(self) -> None:
        """Load transaction data from CSV file."""
        try:
            logger.info(f"Loading data from {self.csv_path}")
            self.data = pd.read_csv(self.csv_path)
            
            # Convert datetime columns to strings for JSON serialization
            if 'trans_date_trans_time' in self.data.columns:
                self.data['trans_date_trans_time'] = self.data['trans_date_trans_time'].astype(str)
            
            logger.info(f"Loaded {len(self.data)} transactions from CSV")
            logger.info(f"Fraud rate: {self.data['is_fraud'].mean()*100:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {str(e)}")
            raise e
    
    def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction against JSON schema."""
        try:
            validate(
                instance=transaction,
                schema=TRANSACTION_SCHEMA,
                format_checker=FormatChecker()
            )
            return True
        except ValidationError as e:
            logger.error(f"Invalid transaction: {e.message}")
            return False

    def get_next_transaction(self) -> Optional[Dict[str, Any]]:
        """
        Get the next transaction from CSV data.
        
        Returns:
            Transaction dict or None if data exhausted (and cycling disabled)
        """
        if self.data is None or len(self.data) == 0:
            logger.error("No data loaded")
            return None
        
        # Check if we've reached the end
        if self.current_index >= len(self.data):
            if self.cycle_data:
                logger.info("Reached end of data, cycling back to start")
                self.current_index = 0
            else:
                logger.info("Reached end of data, stopping")
                return None
        
        # Get current row as dictionary
        row = self.data.iloc[self.current_index].to_dict()
        self.current_index += 1
        
        # Convert numpy types to Python types for JSON serialization
        transaction = {}
        for key, value in row.items():
            if pd.isna(value):
                transaction[key] = None
            elif isinstance(value, (pd.Timestamp, datetime)):
                transaction[key] = value.isoformat()
            elif hasattr(value, 'item'):  # numpy types
                transaction[key] = value.item()
            else:
                transaction[key] = value
        
        # Validate before returning
        if self.validate_transaction(transaction):
            return transaction
        return None

    def delivery_report(self, err, msg):
        """Delivery callback for confirming message delivery."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def send_transaction(self) -> bool:
        """Send a single transaction to Kafka with error handling."""
        try:
            transaction = self.get_next_transaction()
            if not transaction:
                return False

            self.producer.produce(
                self.topic,
                key=str(transaction.get("trans_num", "")),
                value=json.dumps(transaction),
                callback=self.delivery_report
            )

            self.producer.poll(0)  # Trigger callbacks
            return True

        except Exception as e:
            logger.error(f"Error producing message: {str(e)}")
            return False

    def run_continuous_production(self, interval: float = 0.1, max_messages: int = None):
        """
        Run continuous message production with graceful shutdown.
        
        Args:
            interval: Time in seconds between messages
            max_messages: Maximum messages to send (None for unlimited)
        """
        self.running = True
        logger.info(f"Starting producer for topic {self.topic}...")
        logger.info(f"Interval: {interval}s, Cycling: {self.cycle_data}")
        
        messages_sent = 0
        fraud_count = 0

        try:
            while self.running:
                if self.send_transaction():
                    messages_sent += 1
                    
                    # Track fraud for stats
                    tx = self.data.iloc[self.current_index - 1]
                    if tx['is_fraud'] == 1:
                        fraud_count += 1
                    
                    # Log progress every 1000 messages
                    if messages_sent % 1000 == 0:
                        logger.info(
                            f"Sent {messages_sent} messages "
                            f"(fraud: {fraud_count}, rate: {fraud_count/messages_sent*100:.2f}%)"
                        )
                    
                    # Check max messages limit
                    if max_messages and messages_sent >= max_messages:
                        logger.info(f"Reached max messages limit: {max_messages}")
                        break
                    
                    time.sleep(interval)
                else:
                    if not self.cycle_data:
                        logger.info("No more transactions to send")
                        break
                        
        finally:
            logger.info(f"Total messages sent: {messages_sent}")
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown procedure."""
        if self.running:
            logger.info("Initiating shutdown...")
            self.running = False
            if self.producer:
                self.producer.flush(timeout=30)
            logger.info("Producer stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Credit Card Transaction Producer')
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Path to processed CSV file'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=0.1,
        help='Interval between messages in seconds'
    )
    parser.add_argument(
        '--max-messages',
        type=int,
        default=None,
        help='Maximum number of messages to send'
    )
    
    args = parser.parse_args()
    
    producer = CreditCardTransactionProducer(csv_path=args.csv_path)
    producer.run_continuous_production(
        interval=args.interval,
        max_messages=args.max_messages
    )