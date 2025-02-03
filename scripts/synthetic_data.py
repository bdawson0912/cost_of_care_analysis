import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of records
n = 1000

# Patient IDs
patient_ids = [f'P{str(i).zfill(4)}' for i in range(1, n+1)]

# Age: Random between 18 and 90
ages = np.random.randint(18, 90, size=n)

# Gender: Random choice
genders = np.random.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])

# Comorbidities Count: Skewed towards 0–2
comorbidities = np.random.poisson(1.5, size=n)

# Procedure code (same for all, as we're focusing on one)
procedure_code = ['KNEE123'] * n

# Random dates in the past two years
dates = pd.to_datetime('2022-01-01') + pd.to_timedelta(np.random.randint(0, 730, size=n), unit='D')

# Adjust probabilities to sum to 1
probabilities = [0.1]*5 + [0.05]*10  # 0.1*5 + 0.05*10 = 1.0

length_of_stay = np.random.choice(range(1, 16), size=n, p=probabilities)


# Base cost influenced by length of stay and comorbidities
base_cost = 10000 + (length_of_stay * 1500) + (comorbidities * 2000)
noise = np.random.normal(0, 3000, size=n)  # Add randomness

# Total cost with noise
total_cost = np.maximum(base_cost + noise, 5000).astype(int)

# Break down the cost
facility_fee = (total_cost * np.random.uniform(0.5, 0.7, size=n)).astype(int)
physician_fee = (total_cost * np.random.uniform(0.2, 0.3, size=n)).astype(int)
medication_cost = total_cost - (facility_fee + physician_fee)

# Hospitals and regions
hospital_ids = np.random.choice([f'HOSP{i}' for i in range(1, 11)], size=n)
regions = np.random.choice(['Northeast', 'South', 'Midwest', 'West'], size=n)
hospital_types = np.random.choice(['Teaching', 'Community'], size=n, p=[0.4, 0.6])

# Readmission influenced by comorbidities
readmission = np.where(comorbidities > 2, np.random.choice(['Yes', 'No'], size=n, p=[0.3, 0.7]), 'No')

# Complications influenced by age and comorbidities
complications = np.where((ages > 65) | (comorbidities > 3),
                          np.random.choice(['Yes', 'No'], size=n, p=[0.2, 0.8]),
                          'No')

# Create DataFrame
data = pd.DataFrame({
    'patient_id': patient_ids,
    'age': ages,
    'gender': genders,
    'comorbidities_count': comorbidities,
    'procedure_code': procedure_code,
    'date_of_service': dates,
    'length_of_stay': length_of_stay,
    'total_cost': total_cost,
    'facility_fee': facility_fee,
    'physician_fee': physician_fee,
    'medication_cost': medication_cost,
    'hospital_id': hospital_ids,
    'region': regions,
    'hospital_type': hospital_types,
    'readmission_within_30_days': readmission,
    'complications': complications
})

# Save to CSV
data.to_csv('synthetic_claims_data.csv', index=False)

# Preview
data.head()
