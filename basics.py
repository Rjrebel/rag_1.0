import os
import faiss

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "faiss_index")

# Check if the Chroma vector store already exists
index_path = os.path.join(persistent_directory, "index.faiss")
if not os.path.exists(index_path):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(r'K:\coding world\LangChain\IceBreaker\RAG\books\odyssey.txt'):
        raise FileNotFoundError(
            r"The file 'RAG\books\odyssey.txt' does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(r'RAG\books\odyssey.txt', encoding='utf-8')
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")


    # Load a Sentence-Transformer model for generating embeddings
    print("\n--- Creating embeddings using Sentence-Transformers ---")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedded_docs = model.encode([doc.page_content for doc in docs], convert_to_tensor=True).cpu().numpy()
    print("\n--- Finished creating embeddings ---")


   
 

     # Create the FAISS index and store the vectors
    print("\n--- Creating FAISS index ---")
    dimension = len(embedded_docs[0])  # Dimensionality of your embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric
    index.add(embedded_docs)  # Add embeddings to the index
    print("\n--- Finished creating FAISS index ---")

    vector_store = FAISS(
    embedding_function=model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    )

    # # Persist the FAISS index to disk
    # if not os.path.exists(persistent_directory):
    #     os.makedirs(persistent_directory)
    # # faiss.write_index(index, index_path)

    # Step 2: Save the FAISS Index to Disk
    folder_path = "faiss_index"
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path)
    print(f"FAISS index saved at: {folder_path}")

else:
    print("FAISS index already exists. Loading from disk...")
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from: {index_path}")

    # Define the user's question
    query = "Who is Odysseus' wife?"

    # Encode the query using the same Sentence-Transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

    # Search for similar document chunks in the FAISS index
    distances, indices = index.search(query_embedding, k=5)  # Retrieve top 5 similar chunks

    loader = TextLoader(r'RAG\books\odyssey.txt', encoding='utf-8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Print the similar document chunks
    print("\n--- Similar Document Chunks ---")
    for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
        print(f"Rank {i+1}:")
        print(f"Distance: {distance:.4f}")
        print(f"Chunk: {docs[index].page_content}\n")