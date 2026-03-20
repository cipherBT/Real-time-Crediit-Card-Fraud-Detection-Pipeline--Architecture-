"""
Real-time Credit Card Fraud Detection Inference Pipeline

This script consumes credit card transaction data from Kafka, processes it using Spark Streaming,
applies a pre-trained machine learning model to detect fraudulent transactions,
and writes predictions back to Kafka.

Updated for credit card fraud dataset with 22 columns including:
- Transaction details (amount, merchant, category)
- Geolocation (user lat/long, merchant lat/long, distance)
- Temporal features (hour, day of week, is_weekend, is_night)
- Demographics (age, gender)
"""

# Standard library imports
import logging
import os

# Third-party imports
import joblib  # For loading serialized ML models
import yaml  # For parsing YAML configuration files
from dotenv import load_dotenv  # For loading environment variables from .env file

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (from_json, col, when, lit, coalesce)
from pyspark.sql.pandas.functions import pandas_udf  # For Pandas vectorized UDFs
from pyspark.sql.types import (StructType, StructField, StringType,
                              IntegerType, DoubleType, LongType)

# Configure logging to track pipeline operations and errors
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO for operational messages
    format="%(asctime)s [%(levelname)s] %(message)s"  # Structured log format
)
logger = logging.getLogger(__name__)  # Create logger instance for the module


class CreditCardFraudInference:
    """
    Credit card fraud detection inference pipeline class that handles:
    - Configuration loading
    - Spark session management
    - Kafka stream processing
    - Feature processing
    - Model inference
    - Results publishing

    Attributes:
        config (dict): Pipeline configuration parameters
        spark (SparkSession): Spark session instance
        model: Loaded ML model for fraud detection
        broadcast_model: Model broadcast to Spark workers for distributed inference
    """

    # Class variables for Kafka configuration
    bootstrap_servers = None
    topic = None
    security_protocol = None
    sasl_mechanism = None
    username = None
    password = None
    sasl_jaas_config = None

    def __init__(self, config_path="/app/config.yaml"):
        """Initialize pipeline with configuration and dependencies

        Args:
            config_path (str): Path to YAML configuration file
        """
        # Load environment variables from .env file
        load_dotenv(dotenv_path="/app/.env")

        # Load pipeline configuration from YAML file
        self.config = self._load_config(config_path)

        # Initialize Spark session with Kafka integration packages
        self.spark = self._init_spark_session()

        # Load and broadcast ML model to worker nodes for distributed inference
        self.model = self._load_model(self.config["model"]["path"])
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)

        # Debug: Log loaded environment variables (sensitive values should be masked in production)
        logger.debug("Environment variables loaded: %s", dict(os.environ))

    def _load_model(self, model_path):
        """Load pre-trained fraud detection model from disk

        Args:
            model_path (str): Path to serialized model file

        Returns:
            model: Loaded ML model

        Raises:
            Exception: If model loading fails
        """
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
            return model
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise

    @staticmethod
    def _load_config(config_path):
        """Load YAML configuration file

        Args:
            config_path (str): Path to configuration file

        Returns:
            dict: Parsed configuration parameters

        Raises:
            Exception: If file loading or parsing fails
        """
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _init_spark_session(self):
        """Initialize Spark session with Kafka dependencies

        Returns:
            SparkSession: Configured Spark session

        Raises:
            Exception: If Spark initialization fails
        """
        try:
            # Get required packages from config (typically Kafka integration packages)
            packages = self.config.get("spark", {}).get("packages", "")

            # Build Spark session with application name and packages
            builder = SparkSession.builder.appName("CreditCardFraudInferenceStreaming")

            # Add Maven packages if specified in config
            if packages:
                builder = builder.config("spark.jars.packages", packages)

            spark = builder.getOrCreate()
            logger.info("Spark Session initialized.")
            return spark
        except Exception as e:
            logger.error("Error initializing Spark Session: %s", str(e))
            raise

    def read_from_kafka(self):
        """Read streaming data from Kafka topic and parse JSON payload

        Returns:
            DataFrame: Spark DataFrame containing parsed transaction data
        """
        logger.info("Reading data from Kafka topic %s", self.config["kafka"]["topic"])

        # Load Kafka configuration parameters with fallback values
        kafka_config = self.config["kafka"]
        kafka_bootstrap_servers = kafka_config.get("bootstrap_servers", "localhost:9092")
        kafka_topic = kafka_config["topic"]
        kafka_security_protocol = kafka_config.get("security_protocol", "SASL_SSL")
        kafka_sasl_mechanism = kafka_config.get("sasl_mechanism", "PLAIN")
        kafka_username = kafka_config.get("username")
        kafka_password = kafka_config.get("password")

        # Configure Kafka SASL authentication string
        kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.plain.PlainLoginModule required '
            f'username="{kafka_username}" password="{kafka_password}";'
        )

        # Store Kafka configuration in instance variables for reuse
        self.bootstrap_servers = kafka_bootstrap_servers
        self.topic = kafka_topic
        self.security_protocol = kafka_security_protocol
        self.sasl_mechanism = kafka_sasl_mechanism
        self.username = kafka_username
        self.password = kafka_password
        self.sasl_jaas_config = kafka_sasl_jaas_config

        # Define schema for incoming JSON credit card transaction data
        json_schema = StructType([
            # Transaction identifiers
            StructField("trans_num", StringType(), True),
            StructField("cc_num", DoubleType(), True),
            
            # Transaction details
            StructField("amt", DoubleType(), True),
            StructField("amt_log", DoubleType(), True),
            StructField("merchant", StringType(), True),
            StructField("category", StringType(), True),
            
            # Location features
            StructField("lat", DoubleType(), True),
            StructField("long", DoubleType(), True),
            StructField("merch_lat", DoubleType(), True),
            StructField("merch_long", DoubleType(), True),
            StructField("distance_km", DoubleType(), True),
            StructField("city_pop", DoubleType(), True),
            StructField("state", StringType(), True),
            
            # Temporal features
            StructField("trans_date_trans_time", StringType(), True),
            StructField("unix_time", LongType(), True),
            StructField("trans_hour", IntegerType(), True),
            StructField("trans_day_of_week", IntegerType(), True),
            StructField("trans_month", IntegerType(), True),
            StructField("is_weekend", IntegerType(), True),
            StructField("is_night", IntegerType(), True),
            
            # Demographics
            StructField("gender", IntegerType(), True),
            StructField("age", IntegerType(), True),
            
            # Target (for validation/logging)
            StructField("is_fraud", IntegerType(), True),
        ])

        # Create streaming DataFrame from Kafka source
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
            .option("subscribe", kafka_topic) \
            .option("startingOffsets", "latest") \
            .option("kafka.security.protocol", kafka_security_protocol) \
            .option("kafka.sasl.mechanism", kafka_sasl_mechanism) \
            .option("kafka.sasl.jaas.config", kafka_sasl_jaas_config) \
            .load()

        # Parse JSON payload using defined schema
        parsed_df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), json_schema).alias("data")) \
            .select("data.*")

        return parsed_df

    def add_features(self, df):
        """Add or process features for model inference

        The preprocessing script already creates most features.
        This method handles any additional processing needed for inference.

        Args:
            df (DataFrame): Input DataFrame with transaction data

        Returns:
            DataFrame: DataFrame with processed feature columns
        """
        # Handle nulls in key columns with defaults
        df = df.withColumn("amt", coalesce(col("amt"), lit(0.0)))
        df = df.withColumn("distance_km", coalesce(col("distance_km"), lit(0.0)))
        df = df.withColumn("city_pop", coalesce(col("city_pop"), lit(0)))
        df = df.withColumn("age", coalesce(col("age"), lit(30)))
        
        # Handle temporal features
        df = df.withColumn("trans_hour", coalesce(col("trans_hour"), lit(12)))
        df = df.withColumn("trans_day_of_week", coalesce(col("trans_day_of_week"), lit(0)))
        df = df.withColumn("is_weekend", coalesce(col("is_weekend"), lit(0)))
        df = df.withColumn("is_night", coalesce(col("is_night"), lit(0)))
        
        # High-risk category indicator
        high_risk_categories = self.config.get('high_risk_categories', 
            ['gas_transport', 'misc_net', 'grocery_pos', 'shopping_net'])
        df = df.withColumn("category_risk", 
            when(col("category").isin(high_risk_categories), 1).otherwise(0))
        
        # Distance-based risk (transactions >100km away are higher risk)
        df = df.withColumn("distance_risk",
            when(col("distance_km") > 100, 1).otherwise(0))

        # Debug: Output schema of processed data for verification
        df.printSchema()
        return df

    def run_inference(self):
        """Main pipeline execution flow: process stream and run predictions"""
        # Local import for Spark executor compatibility
        import pandas as pd

        # Process streaming data from Kafka
        df = self.read_from_kafka()

        # Add engineered features to raw data
        feature_df = self.add_features(df)

        # Get broadcasted model reference for use in UDF
        broadcast_model = self.broadcast_model

        # Define prediction UDF using Pandas for vectorized operations
        @pandas_udf("int")
        def predict_udf(
                amt: pd.Series,
                amt_log: pd.Series,
                distance_km: pd.Series,
                city_pop: pd.Series,
                trans_hour: pd.Series,
                trans_day_of_week: pd.Series,
                trans_month: pd.Series,
                is_weekend: pd.Series,
                is_night: pd.Series,
                gender: pd.Series,
                age: pd.Series,
                category: pd.Series,
                merchant: pd.Series,
                state: pd.Series,
                category_risk: pd.Series,
                distance_risk: pd.Series
        ) -> pd.Series:
            """Vectorized UDF for batch prediction using pre-trained model

            Args:
                Various Pandas Series containing feature values

            Returns:
                pd.Series: Binary predictions (0=legitimate, 1=fraud)
            """
            # Create input DataFrame from feature columns
            input_df = pd.DataFrame({
                "amt": amt,
                "amt_log": amt_log,
                "distance_km": distance_km,
                "city_pop": city_pop,
                "trans_hour": trans_hour,
                "trans_day_of_week": trans_day_of_week,
                "trans_month": trans_month,
                "is_weekend": is_weekend,
                "is_night": is_night,
                "gender": gender,
                "age": age,
                "category": category,
                "merchant": merchant,
                "state": state,
                "category_risk": category_risk,
                "distance_risk": distance_risk
            })

            # Get fraud probabilities and apply classification threshold
            try:
                probabilities = broadcast_model.value.predict_proba(input_df)[:, 1]
                threshold = 0.48  # Tuned based on recent training precision/recall constraints
                predictions = (probabilities >= threshold).astype(int)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                predictions = pd.Series([0] * len(input_df))
            
            return pd.Series(predictions)

        # Apply predictions to streaming DataFrame
        prediction_df = feature_df.withColumn("prediction", predict_udf(
            col("amt"),
            col("amt_log"),
            col("distance_km"),
            col("city_pop"),
            col("trans_hour"),
            col("trans_day_of_week"),
            col("trans_month"),
            col("is_weekend"),
            col("is_night"),
            col("gender"),
            col("age"),
            col("category"),
            col("merchant"),
            col("state"),
            col("category_risk"),
            col("distance_risk")
        ))

        # Filter to only include high-confidence fraud predictions
        fraud_predictions = prediction_df.filter(col("prediction") == 1)

        # Write results back to Kafka topic
        (fraud_predictions.selectExpr(
            "CAST(trans_num AS STRING) AS key",
            "to_json(struct(*)) AS value"  # Serialize all fields as JSON
        )
         .writeStream
         .format("kafka")
         .option("kafka.bootstrap.servers", self.bootstrap_servers)
         .option("topic", 'fraud_predictions')  # Output topic for fraud alerts
         .option("kafka.security.protocol", self.security_protocol)
         .option("kafka.sasl.mechanism", self.sasl_mechanism)
         .option("kafka.sasl.jaas.config", self.sasl_jaas_config)
         .option("checkpointLocation", "checkpoints/checkpoint")  # For fault tolerance and recovery
         .outputMode("update")  # Only write updated records
         .start()
         .awaitTermination())  # Keep the streaming context alive


if __name__ == "__main__":
    """Main entry point for the inference pipeline"""
    # Initialize pipeline with configuration
    inference = CreditCardFraudInference("/app/config.yaml")

    # Start streaming processing and block until termination
    inference.run_inference()