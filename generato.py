import pandas as pd
import random
import string

# Function to generate random strings
def random_string(length):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# Generate synthetic data
num_records = 1000  # Number of rows in the dataset
data = {
    "Bug_ID": [f"B{str(i).zfill(4)}" for i in range(1, num_records + 1)],
    "File_Changes": [random.randint(1, 20) for _ in range(num_records)],
    "Lines_Added": [random.randint(0, 500) for _ in range(num_records)],
    "Lines_Removed": [random.randint(0, 500) for _ in range(num_records)],
    "Code_Complexity": [round(random.uniform(1.0, 10.0), 2) for _ in range(num_records)],
    "Bug_Report_Length": [random.randint(20, 500) for _ in range(num_records)],
    "Reported_By": [random.choice(["Developer", "Tester", "User"]) for _ in range(num_records)],
    "Module_Affected": [random_string(5) for _ in range(num_records)],
    "Previous_Bugs_in_Module": [random.randint(0, 50) for _ in range(num_records)],
    "Time_to_Fix": [round(random.uniform(1.0, 100.0), 2) for _ in range(num_records)],
    "Severity": [random.choice(["High", "Medium", "Low"]) for _ in range(num_records)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_filename = "bug_severity_dataset.csv"
df.to_csv(csv_filename, index=False)

print(f"Dataset saved as {csv_filename}")
