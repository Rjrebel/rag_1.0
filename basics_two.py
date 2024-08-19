# import os
# import faiss
# from sentence_transformers import SentenceTransformer
# # from langchain.vectorstores.faiss import FAISS
# # from langchain.docstore import InMemoryDocstore
# # from langchain.schema import Document
# # from langchain.vectorstores.base import IndexToDocstoreId
# import numpy as np

# # Define the persistent directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# persistent_directory = os.path.join(current_dir, "db", "faiss_index")

# # Define the embedding model using Sentence-Transformers
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Check if the FAISS index already exists
# index_path = os.path.join(persistent_directory, "faiss.index")
# if os.path.exists(index_path):
#     print("Loading existing FAISS index...")
#     index = faiss.read_index(index_path)
# else:
#     raise ValueError("FAISS index not found. Ensure it is built and saved.")




# # Load or generate a query vector
# query_vector = model.encode([query])

# k = 5  # Number of nearest neighbors to retrieve
# distances, indices = index.search(np.array([query_vector], dtype=np.float32), k)

# # Print the most similar documents
# # for i, index in enumerate(indices[0]):
# #     distance = distances[0][i]
# #     print(f"Nearest neighbor {i+1}: {documents[index]}, Distance {distance}")

# # # Initialize the docstore and index_to_docstore_id
# # docstore = InMemoryDocstore()  # Stores the documents
# # index_to_docstore_id = IndexToDocstoreId()  # Maps FAISS index to document IDs

# # # Load the vector store with FAISS, docstore, and index_to_docstore_id
# # db = FAISS(embedding_function=model.encode, index=index, 
# #            docstore=docstore, index_to_docstore_id=index_to_docstore_id)



# # Convert the query to an embedding
# # query_embedding = model.encode(query).reshape(1, -1)

# # Retrieve relevant documents based on the query
# # # Using the FAISS index to perform similarity search
# # k = 3  # Number of top results to return
# # distances, indices = db.index.search(query_embedding, k)

# # Display the relevant results with metadata
# # print("\n--- Relevant Documents ---")
# # for i, idx in enumerate(indices[0], 1):
# #     if idx != -1:  # -1 indicates no more results
# #         doc = db.docstore.get_document(idx)
# #         print(f"Document {i}:\n{doc.page_content}\n")
# #         if doc.metadata:
# #             print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
