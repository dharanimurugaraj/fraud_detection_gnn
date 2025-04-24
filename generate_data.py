import pandas as pd
import random
from datetime import datetime, timedelta

# Parameters
num_transactions = 1000
num_users = 100  # 100 unique users (A0 to A99)
num_devices = 20  # 20 unique devices (D0 to D19)
fraud_ratio = 0.2  # 20% fraud
start_time = datetime(2025, 4, 1, 10, 0, 0)

# Generate user and device IDs
users = [f"A{i}" for i in range(num_users)]
devices = [f"D{i}" for i in range(num_devices)]

# Create transactions
data = []
for i in range(num_transactions):
    sender = random.choice(users)
    receiver = random.choice(users)
    while receiver == sender:  # Avoid self-loops
        receiver = random.choice(users)
    
    amount = random.randint(50, 1000)  # Random amount between 50 and 1000
    timestamp = start_time + timedelta(minutes=i * 5)  # 5-minute intervals
    
    # Assign fraud label: 20% chance, or higher if in a fraud pattern
    is_fraud = 0
    device_id = random.choice(devices)
    
    # Simulate fraud patterns
    if random.random() < fraud_ratio or (sender in ["A10", "A11", "A12"] and receiver in ["A10", "A11", "A12"]):
        is_fraud = 1
        device_id = "D0"  # Fraud ring uses same device (e.g., A10-A11-A12 ring)
    elif sender == "A20" or receiver == "A20":  # A20 as a fraud hub
        is_fraud = random.choice([0, 1])
        device_id = "D1"
    
    data.append([sender, receiver, amount, timestamp.strftime("%Y-%m-%d %H:%M:%S"), is_fraud, device_id])

# Create DataFrame
df = pd.DataFrame(data, columns=["sender", "receiver", "amount", "timestamp", "is_fraud", "device_id"])

# Save to CSV
df.to_csv("data/transactions.csv", index=False)
print(f"Generated {len(df)} transactions in data/transactions.csv")