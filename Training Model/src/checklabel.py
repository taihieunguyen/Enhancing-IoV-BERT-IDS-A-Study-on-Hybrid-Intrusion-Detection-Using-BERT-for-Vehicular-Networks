import pandas as pd
import os

base_dir = '/home/user/Desktop/AI/iov_bert_project/data/finetune'
datasets = ['CICIDS-2017', 'BoT-IoT', 'Car-Hacking', 'IVN-IDS']

for dataset_name in datasets:
    dataset_dir = f'{base_dir}/{dataset_name}'
    print(f"\nDataset: {dataset_name}")
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dataset_dir, file_name)
            try:
                data = pd.read_csv(file_path)
                print(f"File: {file_name}")
                print(f"Columns: {list(data.columns)}")
                print(f"First few values in potential label columns:")
                for col in data.columns:
                    if any(keyword in col.lower() for keyword in ['label', 'class', 'attack', 'category', 'flag']):
                        print(f"  {col}: {data[col].unique()[:5]}")
                print()
            except Exception as e:
                print(f"Error reading {file_name}: {e}")