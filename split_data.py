import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_csv, output_train='train.csv', output_test='test.csv', test_size=0.2, random_state=42):
    """
    Reads a CSV, splits it into train/test sets, and saves them as separate files.
    """
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Perform the split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Save to disk
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    print(f"Success! Split complete.")
    print(f"  Train set saved to: {output_train} ({len(train_df)} samples)")
    print(f"  Test set saved to:  {output_test} ({len(test_df)} samples)")

if __name__ == "__main__":
    # Example: Generate dummy data if no file exists, then split it
    if not os.path.exists("dataset.csv"):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        df.to_csv("dataset.csv", index=False)
    
    split_dataset("dataset.csv")