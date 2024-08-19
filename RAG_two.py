import os
import faiss

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


index_path = r"RAG\db\faiss_index"
loader = TextLoader(r'RAG\books\elon.txt', encoding='utf-8')
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

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

faiss.write_index(index, index_path)


# Define the user's question
query = "What is Elon Musk's ambitious goal?"
query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

# Search for similar document chunks in the FAISS index
distances, indices = index.search(query_embedding, k=3)


relevant_docs = ""
# Print the similar document chunks
print("\n--- Similar Document Chunks ---")
for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
    print(f"Rank {i+1}:")
    print(f"Distance: {distance:.4f}")
    print(f"Chunk: {docs[index].page_content}\n")
    relevant_docs += "\n"  + docs[index].page_content + "\n"



# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n" + relevant_docs
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

print("-----------------Combined Input------------------")
print(combined_input)

# Create a ChatOpenAI model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key="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")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)