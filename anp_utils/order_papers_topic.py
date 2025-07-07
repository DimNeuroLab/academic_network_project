import pandas as pd

df_papers = pd.read_csv('../anp_data/raw/papers.csv')
df_about = pd.read_csv('../anp_data/raw/about.csv')

# Perform inner join on id_paper
df_merged = pd.merge(df_papers, df_about, left_on='id', right_on='paper_id')

# Sort the merged DataFrame by the "citazioni" column
df_sorted = df_merged.sort_values(by='citations', ascending=False)

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv('../anp_data/raw/sorted_papers_about.csv', index=False)

print("File 'sorted_papers_about.csv' created successfully.")