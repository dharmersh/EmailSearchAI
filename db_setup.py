import huggingface_Inference_api as api
from sentence_transformers import SentenceTransformer
import json

import faiss
import pandas as pd
import numpy as np


# Load the preprocessed data
preprocessed_path = 'data/preprocessed_email_data.csv'
data = pd.read_csv(preprocessed_path)

# Convert embeddings back to numpy arrays
data['body_embedding'] = data['body_embedding'].apply(lambda x: np.array(json.loads(x)))

# Create FAISS index
dimension = len(data['body_embedding'].iloc[0])
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
embeddings = np.vstack(data['body_embedding'].values)
index.add(embeddings)

# Create metadata for retrieval
metadata = data[['thread_id', 'subject', 'timestamp', 'cleaned_body']]

# Define a function to query FAISS
def search_faiss(query: str, top_k: int = 5):
    # Get embedding from Hugging Face model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings= model.encode(query).tolist()
    query_embedding = np.array(embeddings).reshape(1, -1)

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve metadata
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            result = metadata.iloc[idx].to_dict()
            result['distance'] = dist
            results.append(result)
    return results


# Test the integrated pipeline
user_query = "Details about the budget approval process"
search_results = search_faiss(user_query)
for result in search_results:
    print("Search Results:", result)
    
response = api.huggingface_api.query(search_results)
print("Search Results:", response)