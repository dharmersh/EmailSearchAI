import pandas as pd 
import re
from sentence_transformers import SentenceTransformer
import numpy as np
# Load the datasets
details_path = 'data\email_thread_details.csv'
summaries_path = 'data\email_thread_summaries.csv'

details_df = pd.read_csv(details_path)
summaries_df = pd.read_csv(summaries_path)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to clean email text
def clean_text(text):
    # Remove reply markers and normalize whitespace
    text = re.sub(r'-----Original Message-----', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

# Apply cleaning to the body and summary columns
details_df['cleaned_body'] = details_df['body'].apply(clean_text)
summaries_df['cleaned_summary'] = summaries_df['summary'].apply(clean_text)


# Generate embeddings for email bodies and summaries
details_df['body_embedding'] = details_df['cleaned_body'].apply(lambda x: model.encode(x).tolist())
summaries_df['summary_embedding'] = summaries_df['cleaned_summary'].apply(lambda x: model.encode(x).tolist())


# Merge datasets on thread_id to associate summaries with email details
merged_df = pd.merge(details_df, summaries_df, on='thread_id')

# Save preprocessed and embedded data for future use
output_path = 'data/preprocessed_email_data.csv'



# Ensure embeddings are saved as strings (JSON-like format)
merged_df['body_embedding'] = merged_df['body_embedding'].apply(lambda x: np.array(x).tolist())
merged_df['summary_embedding'] = merged_df['summary_embedding'].apply(lambda x: np.array(x).tolist())


# Save to a CSV file
merged_df.to_csv(output_path, index=False)

print(f"Preprocessed data saved to {output_path}")
