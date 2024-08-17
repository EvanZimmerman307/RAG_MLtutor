import argparse
import json
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
CHAT_HISTORY_FILE = "chat_history.json"

PROMPT_TEMPLATE = """
You are an AI assistant specializing in financial planning. Given a question, which might reference context in the chat history, 
formulate an answer to it. If the query does not relate to financial planning, state that it does not relate to financial planning, and do not answer the question, instead you must politely refuse to respond.
You are forbidden from reccommending actions. You can only explain financial planning concepts.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Current conversation:
{chat_history}
Human: {question}
AI Assistant: Let's approach this step-by-step:
"""

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(chat_history, f)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def print_retrieved_documents(docs):
    formatted_docs = format_docs(docs)
    print("\nRetrieved Documents:\n")
    print(formatted_docs)
    print("\n" + "-" * 40 + "\n")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = OpenAIEmbeddings()
    db = Chroma(collection_name="cfp_rag_store", persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    chat_history = load_chat_history()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for message in chat_history:
        if message['type'] == 'human':
            memory.chat_memory.add_user_message(message['content'])
        elif message['type'] == 'ai':
            memory.chat_memory.add_ai_message(message['content'])
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 7})
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    model = ChatOpenAI(model='gpt-4o')

    def format_chat_history(chat_history):
        return "\n".join(f"{msg['type'].capitalize()}: {msg['content']}" for msg in chat_history)

    def combine_documents(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    docs = retriever.get_relevant_documents(query_text)
    print_retrieved_documents(docs)

    # Create the context by combining the retrieved documents
    context = combine_documents(docs)

    # Prepare the input dictionary
    inputs = {
        "context": context,
        "chat_history": format_chat_history(chat_history),
        "question": query_text,
    }

    # Run the pipeline
    chain = prompt | model | StrOutputParser()
    result = chain.invoke(inputs)

    print(f"Question: {query_text}")
    print(f"Answer: {result}")
    
    # Update memory and chat history
    memory.chat_memory.add_user_message(query_text)
    memory.chat_memory.add_ai_message(result)
    
    chat_history.append({"type": "human", "content": query_text})
    chat_history.append({"type": "ai", "content": result})
    
    save_chat_history(chat_history)

    # Print chat history
    print("\nChat History:")
    for message in chat_history:
        print(f"{message['type'].capitalize()}: {message['content']}")


if __name__ == "__main__":
    main()
