FROM apache/airflow:2.10.5

# Create models directory as root
USER root
RUN mkdir -p /app/models && chown airflow:root /app/models

# Switch to airflow user for pip install (required by Airflow image)
USER airflow

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt