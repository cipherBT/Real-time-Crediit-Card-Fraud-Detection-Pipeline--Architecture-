import pandas as pd
import joblib

# Load the optimized model directly from the volume
MODEL_PATH = "/app/models/fraud_detection_model.pkl"
print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)

# Define a highly suspicious manual transaction (Massive amount, far distance, risky category)
manual_transaction = {
    'amt': [850000.00],                    # Extremely high amount
    'amt_log': [9.05],                   # Log of the amount
    'distance_km': [2500.0],             # 2500 kilometers away from home
    'city_pop': [5000000],                 # Typical city population
    'trans_hour': [7],                   # 3 AM in the morning
    'trans_day_of_week': [6],            # Sunday
    'trans_month': [12],                 # December
    'is_weekend': [1],                   # Weekend
    'is_night': [1],                     # Nighttime
    'gender': ["M"],                     # "M" or "F"
    'age': [65],                         # 65 years old
    'category': ["shopping_net"],        # High risk internet shopping
    'merchant': ["fraud_store_123"],     # Arbitrary merchant
    'state': ["LG"],                     # State abbreviation
    'category_risk': [1],                # Flagged as high risk category
    'distance_risk': [1]                 # Flagged as unusual distance
}

# Convert to DataFrame
df = pd.DataFrame(manual_transaction)

# Get the exact probability score
probability = model.predict_proba(df)[0][1]

# Apply our optimized threshold
threshold = 0.48
is_fraud = probability >= threshold

print("\n" + "="*50)
print("🔍 TRANSACTION ANALYSIS RESULTS 🔍")
print("="*50)
print(f"Transaction Amount: ${manual_transaction['amt'][0]}")
print(f"Purchase Category: {manual_transaction['category'][0]}")
print(f"Time of Purchase: {manual_transaction['trans_hour'][0]}:00")
print("-"*50)
print(f"Fraud Probability Score: {probability:.4f} (Threshold: {threshold})")

if is_fraud:
    print(f"VERDICT: FRAUD DETECTED!")
else:
    print(f"VERDICT: LEGITIMATE TRANSACTION")
print("="*50 + "\n")
