# Operations & Incident Debugging History

### 1. Docker Daemon Out-of-Memory (OOM) resolution (`dags/fraud_detection_training.py`)
**Initial State**: The Airflow DAG container abruptly terminated during `RandomizedSearchCV` (Exit Code 137). 
**The Fixes**:
* **Smart Undersampling (Lines ~287-292)**: Deleted the blind `data.sample(n=150000)` code. Wrote targeted separation logic:
  ```python
  fraud_cases = data[data['is_fraud'] == 1]
  legit_cases = data[data['is_fraud'] == 0].sample(n=150000, random_state=42)
  data = pd.concat([fraud_cases, legit_cases])
  ```
  This single change preserved 100% of the minority incidents while preventing RAM exhaustion.
* **XGBoost Threading (Line ~352)**: Redefined `n_jobs=1` inside `RandomizedSearchCV` instead of `-1` to prevent CPU parallel threads from violently multiplying maximum memory allocation boundaries footprint.
* **Optimization Balance (Line ~335)**: `n_iter=20` and `SMOTE(sampling_strategy=0.5)` to violently bias the data for maximum precision.

### 2. Scikit-Learn Container Version Mismatch (`requirements.txt`)
**Initial State**: Airflow Model trained on Python 3.12 (Scikit 1.6+). Inference decoded on Python 3.10 (Scikit 1.5.0). 
**The Fix**:
* Modified `airflow/requirements.txt` (Line 11) directly from `scikit-learn` to `scikit-learn==1.5.0`.
* Modified `inference/requirements.txt` (Line 7) directly from `scikit-learn` to `scikit-learn==1.5.0`, enforcing absolute matching class signatures and fixing `imblearn` vs `imbalanced-learn` casing errors.

### 3. Kafka Producer Schema Rejection (`producer/main.py`)
**Initial State**: Exception thrown automatically: `Invalid transaction: 'M' is not of type 'integer'`. 
**The Fix**:
* Navigated to `TRANSACTION_SCHEMA` dictionary (Line ~60).
* Changed `"gender": {"type": "integer"}` directly to `"gender": {"type": "string"}` allowing `"M"`/`"F"` formats to cross the barrier safely.

### 4. Kafka Topic Dead-Ends & "UnknownTopicOrPartitionException"
**Initial State**: Confluent Cloud disables auto-topic generation. Both the producer and inference containers stalled due to interacting with null Kafka sockets. 
**The Fix**:
* Bypassed the Docker container to execute raw runtime injection. Ran the following Python script against the `src-producer-1` container twice:
  ```python
  admin = AdminClient(conf)
  admin.create_topics([NewTopic('transactions', num_partitions=1)])
  admin.create_topics([NewTopic('fraud_predictions', num_partitions=1)])
  ```

### 5. Docker Volume Binding Typo (`docker-compose.yml`)
**Initial State**: Inference threw SASL Auth failures. 
**The Fix**:
* Fixed line 343 in `docker-compose.yml` to securely mount the environment variable configuration:
  * Incorrect: `- ./env:/app/.env`
  * Corrected: `- ./.env:/app/.env`

### 6. Streaming Threshold Calibration (`inference/main.py`)
**Initial State**: Inference failed to output valid predictions due to a brutally aggressive probability ceiling. 
**The Fix**:
* Navigated to Line 345 inside the PySpark Pandas prediction UDF.
* Changed `threshold = 0.60` explicitly to `threshold = 0.48`. This perfectly offset the model's conservative bias.

### 7. Interactive Prediction Tooling (`ui/app.py` & `dags/predict_manual.py`)
**Initial State**: Headless environment required Cloud UI parsing. 
**The Fixes**:
* **CLI Wrapper**: Created `src/dags/predict_manual.py` that loads `/app/models/fraud_detection_model.pkl` to process a hardcoded Pandas Dictionary representing exactly 1 transaction in 15 seconds.
* **Custom Streamlit Web Dashboard**: Built a completely new `ui/app.py` interface.
  * Used `@st.cache_resource` for O(1) model loading. Bound input sliders to an interactive Dataframe triggering `predict_proba`.
  * Included a `requirements.txt` identifying a fatal internal module cache failure: Added `dill` explicitly (Line 6) because the Airflow DAG used `dill` context wrapping to initially pickle the SMOTE pipeline variables.
  * Resolved the final fatal Docker error (EOF memory dump during the UI compilation) by explicitly executing `docker builder prune -f` in the host CLI to reclaim block storage memory, securing the final container launch on port 8501.
