import os
import faiss

# from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
# load_dotenv()


# Define the embedding model
embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embedded_docs = model.encode([doc.page_content for doc in docs], convert_to_tensor=True).cpu().numpy()

# Load the existing vector store with the embedding function
db = FAISS.load_local(
    folder_path=r"K:\coding world\LangChain\IceBreaker\faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True
)


# Define the user's question
query = "Who is Odysseus' wife?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

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