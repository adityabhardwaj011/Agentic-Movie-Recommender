import chromadb
from sentence_transformers import SentenceTransformer
import sys # to print debug msg to our terminal


# INITIALIZATING RECOMMENDER TOOL WITH CHROMADB
try:
    # Connecting to the ChromaDB client
    client = chromadb.PersistentClient(path="chroma_db")
    
    # loading the existing collection of movies
    collection = client.get_collection(name="movies")
    
    # Load the Sentence Transformer model (to encode user query and idntify which tool to use)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("ChromaDB client and Sentence Transformer model loaded successfully.")
    
except Exception as e:
    initialization_error = f"Failed to initialize ChromaDB or Sentence Transformer model: {e}"
    print(f"FATAL: {initialization_error}", file=sys.stderr)
    collection = None
    model = None

# RECOMMENDATION FUNCTIONS (TOOLS)

def get_recommendations_by_description(description: str, top_n: int = 10):
    if collection is None or model is None:
        return f"System initialization failed: {initialization_error}"
    
    try:
        # Encoding the user's input into a vector
        query_embedding = model.encode(description).tolist()
        
        # Query the ChromaDB collection
        # The database does all the similarity search and sorting for us!
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n
        )
        
        # Extracting the titles from the results
        recommended_movies = [meta['title'] for meta in results['metadatas'][0]]
        
        print(f"DEBUG: Found recommendations for description '{description}': {recommended_movies}")
        return recommended_movies
        
    except Exception as e:
        error_message = f"An unexpected error occurred in get_recommendations_by_description: {e}"
        print(error_message, file=sys.stderr)
        return "I'm sorry, I encountered an unexpected error while searching by description."

def get_recommendations_by_title(movie_title: str, top_n: int = 10):
    if collection is None:
        return f"System initialization failed: {initialization_error}"

    try:
        # finding the movie(filtering by metadata(title)) in the database to get its embedding
        movie = collection.get(
            where={'title': movie_title},
            limit=1,
            include=['embeddings']
        )
        
        if not movie['ids']:
            return f"I'm Sorry i can't help find releated movies to'{movie_title}' as i don't have it in my database "
            
        source_embedding = movie['embeddings'][0]

        # Quering the collection using the retrieved embedding
        results = collection.query(
            query_embeddings=[source_embedding],
            n_results=top_n + 1 # +1 to account for the movie itself being returned
        )

        # Extracting the titles(response)
        recommended_movies = []
        for meta in results['metadatas'][0]:
            if meta['title'] != movie_title:
                recommended_movies.append(meta['title'])
        
        print(f"DEBUG: Found recommendations for title '{movie_title}': {recommended_movies}")
        return recommended_movies[:top_n] # Ensure we only return top_n

    except Exception as e:
        error_message = f"An unexpected error occurred in get_recommendations_by_title: {e}"
        print(error_message, file=sys.stderr)
        return "I'm sorry, I encountered an unexpected error while searching by title."
