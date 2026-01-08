import pandas as pd

df_writes = pd.read_csv('../anp_data/raw/writes.csv')
df_papers = pd.read_csv('../anp_data/raw/papers.csv')
df_about = pd.read_csv('../anp_data/raw/about.csv')

year = 2019

# Merge writes with papers to get paper topics
df_paper_topics = pd.merge(df_writes, df_papers, left_on='paper_id', right_on='id')

# Merge paper_topics with about to get author topics
df_author_topics = pd.merge(df_paper_topics, df_about, on='paper_id')

# Filter by year
df_author_topics_filtered = df_author_topics[df_author_topics['year'] <= year]

# Group by author_id and topic_id to count occurrences
df_author_topic_counts = df_author_topics_filtered.groupby(['author_id', 'topic_id']).size().reset_index(name='count')

# Sort by author_id and count to get the most common topics per author
df_sorted = df_author_topic_counts.sort_values(by=['author_id', 'count'], ascending=[True, False])

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv('../anp_data/raw/sorted_authors_topics_{}.csv'.format(year), index=False)

print("File 'sorted_authors_topics_{}.csv' created successfully.".format(year))
