import pandas as pd

df = pd.read_csv('dataset_01_companies_complete.csv')
print(f'✓ Dataset 01: {len(df)} rows, {len(df.columns)} columns')
print(f'  Columns: {", ".join(df.columns.tolist())}')

# Check all 10 datasets
from pathlib import Path
for i in range(1, 11):
    f = Path(f'dataset_{i:02d}_companies_complete.csv')
    if f.exists():
        df = pd.read_csv(f)
        print(f'✓ Dataset {i:02d}: {len(df)} rows')
