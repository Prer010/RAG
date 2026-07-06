from xml.parsers.expat import model
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from ingestion_pipeline import EmbeddingFunction
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

persistent_directory = "db/chroma_db"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = EmbeddingFunction(embedding_model)


db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_function,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Create a ChatGroq model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone and searchable
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = llm.invoke(messages)
        search_question = result.content.strip() #removes leading and trailing whitespaces
        # print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve relevant document chunks in the vector database
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(search_question)

    # Display results
    # print("--- Context ---")
    # for i, doc in enumerate(relevant_docs, 1):
    #     print(f"Document {i}:\n{doc.page_content}\n") // displaying retrieved chunks for debugging purposes


    # Combine the query and the relevant document contents
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = llm.invoke(messages)
    answer = result.content

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print("\n--- Generated Response ---")
    print(answer)

    return answer


# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()