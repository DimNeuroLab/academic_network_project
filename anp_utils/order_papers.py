import pandas as pd

# Load the CSV file
df = pd.read_csv('../anp_data/raw/papers.csv')

# Sort the DataFrame by the "citations" column
df_sorted = df.sort_values(by='citations', ascending=False)

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv('../anp_data/raw/sorted_papers.csv', index=False)

print("File 'sorted_papers.csv' created successfully.")

