# LLM aren't capable of get current information from the web, so we need to use a retrieval chain to get the information we need.
# We can load information from the web, pdfs, databases, etc. and pass it to the LLM.

from re import search
from dotenv import load_dotenv

load_dotenv()   

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

# nos permite criar uma chain com Prompt, model e output parser, mas passar uma lista de docs
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# used to create a history aware retriever
# a history aware retriever will use the chat history to improve the retrieval
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

#VectorDB
from langchain_community.vectorstores.faiss import FAISS 

#history schema
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

# docA = Document(
#     page_content="LangChain Expression Language, or LCEL, is a declarative way to define the structure of a document.\
#           It is a language that allows you to define the structure of a document in a way that is easy to read and write. \
#             LCEL is a language that is designed to be human-readable and easy to understand"
# )

def get_document_from_web(url):
    """
    WebBaseLoader will scrape the URL and return the content of the page.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()

    """
        Usado para limitar o texto retornado
        Este código limitará o texto que é "screpado" para 200 caracteres, com um overlap de 20 caracteres.
        E assim, economizar tokens.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_db(docs):
    """
    OpenAIEmbeddings will convert the text into a vector.
    Injecting the docs into the VectorDB. It wiill allow us to search for the most similar document.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    return vector_store

def create_chain(vector_store_db):
    #1 create the model
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.2, # the closer to zero we get, the more factual the response will be
    )

    #2 create the prompt dinamicall with ChatPromptTemplate.from_messages
    # prompt = ChatPromptTemplate.from_template("""
    #     Answer the user's question.
    #     Context: {context}
    #     Question: {input}
    # """)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"), # placeholder for the chat history
        ("human", "{input}")
    ])


    #3 create the chain and return it as a composition of the model and the prompt
    chain =create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    #4 returning docs from the vector store as a retriever to pass it to the retrieval chain
    """"
        k=1 will return the most similar document
        by default, search_kwargs it set to k=5 which are the 5 most similar documents
    """

    #this prompot is used to combine the chat history with the lastest user input
    # and will be used to generate the query for the vector store
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"), # placeholder for the chat history
         ("human", "{input}")
         ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # retriever = vector_store_db.as_retriever(search_kwargs={"k": 1}) 
    retriever = vector_store_db.as_retriever(search_kwargs={"k": 3})
    history_aware_retriever = create_history_aware_retriever(
        llm=model, # llm is used to generate the query based on the chat history
        # and why is that? Because the vector DB only has data about the documents, not the chat history
        retriever=retriever,
        prompt=retriever_prompt)

    #5 this chain will first search for the most relevant document and then pass it to the LLM
    retriavel_chain = create_retrieval_chain(
        # retriever,
        history_aware_retriever,
        chain
    )

    return retriavel_chain
# res = chain.invoke({
#     "context": "The following is a list of the top 10 most populous countries in the world.",
#     "input": "What is the most populous country in the world?"
# })

# res = chain.invoke({
#     "context": [docA],
#     "input": "What is LCEL?"
# })

def process_chat(chain, question, chat_history):
    # Se quisermos usar um retrievel chain, use o create_stuff_documents_chain
    res = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    # print(res)
    return res['answer']

if __name__ == '__main__':
    # fetch the content from the web    
    docs = get_document_from_web('https://python.langchain.com/docs/expression_language/')
    vector_store_db = create_vector_db(docs)
    chain = create_chain(vector_store_db)
    chat_history = [
    ]

    while True:
        user_input = input("You: ") 
        if user_input.lower() == 'exit':
            break
        res = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=res))
        print("Assistent: ", res)
