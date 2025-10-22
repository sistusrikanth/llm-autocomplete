# Precompytes  a 512d embedding for each user based on the sentences they have spoken in the past. 
# TODO: apart from speech patterns, consider other factors like age, user interests, last-N, time of day, etc.  (TBD)


import torch
from sentence_transformers import SentenceTransformer
import random
import json

# Load sentence transformer model
embed_model = SentenceTransformer('all-MiniLM-L12-v2')  # 384-d by default

# read user_persona.json
with open('data/user_persona.json', 'r') as f:
    user_persona = json.load(f)

user_embeddings = {}
for user in user_persona:
    user_id = user['name']
    # Create a unique user_id
    user_id = f"{user_id}_{user['age']}"
    user_embeddings[user_id] = []
    sentences = user['sentences']
    embeddings = embed_model.encode(sentences, convert_to_tensor=True)
    avg_embedding = embeddings.mean(dim=0)
    
    # Move to CPU for consistency
    avg_embedding = avg_embedding.cpu()
    
    if avg_embedding.shape[0] != 512:
        projection = torch.nn.Linear(avg_embedding.shape[0], 512)
        avg_embedding = projection(avg_embedding)
    user_embeddings[user_id] = avg_embedding


# Print a random user embedding from the dict
random_user_id = random.choice(list(user_embeddings.keys()))
print("Example user embedding:", user_embeddings[random_user_id])
print("Example user embedding shape:", user_embeddings[random_user_id].shape)