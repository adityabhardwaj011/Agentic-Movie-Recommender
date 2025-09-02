import chromadb
import numpy as np
import pandas as pd


# 1. Initializing ChromaDB Client
# creating a persistent database in a folder 'chroma_db'
client = chromadb.PersistentClient(path="chroma_db")

# 2. Create or Get(if already exists) a Collection(like a table in traditional db)
collection_name = "movies"
collection = client.get_or_create_collection(name=collection_name)
print(f"Collection '{collection_name}' ready.")

# 3. Loading our Data 
try:
    movie_embeddings = np.load('data/movie_embeddings.npy')
    movie_titles_df = pd.read_csv('data/movie_titles.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("File not found")
    exit()

# 4. Preparing Data for ChromaDB (embeddings, metadatas, and ids(to add data))
embeddings_list = movie_embeddings.tolist() # Convert numpy array to a list of lists
ids_list = [str(i) for i in range(len(movie_titles_df))] # Create a unique ID for each movie

# Create metadata for each movie (just the title)
metadatas_list = [{'title': row['title']} for index, row in movie_titles_df.iterrows()]
print("Data ready for ingestion")

# 5. Ingest Data into ChromaDB
# We add the data to the collection. ChromaDB handles all the indexing automatically.
print("Ingesting data into ChromaDB")
collection.add(
    embeddings=embeddings_list,
    metadatas=metadatas_list,
    ids=ids_list
)

print("\n Ingestion complete! vector store is ready.")
print(f"Total items in collection: {collection.count()}")
