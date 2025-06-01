from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
import torch

#Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Paragraph chunks
chunks = [
    "Holi, also known as the Festival of Colors, is a vibrant and joyous Hindu festival celebrated in India and other parts of the world.",
    "It marks the arrival of spring and signifies the triumph of good over evil.",
    "People celebrate by throwing colored powder and water at each other, singing, dancing, and enjoying festive treats.",
    "Holi is a time for fun, forgiveness, and unity, as people of all backgrounds come together to celebrate the spirit of love and togetherness."
]

#Encode paragraph chunks
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

#question
question = "Why do people celebrate Holi?"
question_embedding = model.encode(question, convert_to_tensor=True)

#most relevant chunk
cosine_scores = util.cos_sim(question_embedding, chunk_embeddings)
top_result = torch.argmax(cosine_scores)
best_chunk = chunks[top_result]

print(f"Question: {question}")
print(f"Most Relevant Context: {best_chunk}")

#########end########




###########testing faiss now######
import faiss
import numpy as np

# Create some test vectors
d = 5  # dimension
nb = 10  # number of vectors
np.random.seed(0)
vectors = np.random.random((nb, d)).astype('float32')

# Create index and add vectors
index = faiss.IndexFlatL2(d)
index.add(vectors)

# Search for nearest neighbor of a new vector
query = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query, 3)  # top 3

print("Indices of similar vectors:", indices)



######FINAL CODE FOR FAISS######

#libraries
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

#document
loader = TextLoader("Kite_runner.txt")
documents = loader.load()

#Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

#SBERT Embeddings
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#FAISS vector database
db = FAISS.from_documents(docs, embedding_model)


#OpenAI Chat Model (GPT)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="API KEY to be added here")

#Retrieval-based QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

#question 
query = "Describe the house"
results = db.similarity_search(query, k=1)

print("\nAnswer:")
print(results[0].page_content)
response = qa_chain.run(query)

#Print the answer
print(response)




