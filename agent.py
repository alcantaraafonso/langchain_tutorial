
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import create_openai_functions_agent, AgentExecutor

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

#VectorDB
from langchain_community.vectorstores.faiss import FAISS 

#Tavily Search is used to search for information on the web.
# to get real-time information
from langchain_community.tools.tavily_search import TavilySearchResults

# Used to convert the retrieved documents into a function that can be used by the agent
from langchain.tools.retriever import create_retriever_tool

#memory
# the memory module is used to store the conversation history
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

load_dotenv()

# Create a retriever
loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/')
docs = loader.load()

"""
    Usado para limitar o texto retornado
    Este código limitará o texto que é "screpado" para 200 caracteres, com um overlap de 20 caracteres.
    E assim, economizar tokens.
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

splitDocs = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 


model = ChatOpenAI(
    model = "gpt-3.5-turbo-1106",
    temperature = 0.7,
)

history = UpstashRedisChatMessageHistory(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
    session_id="my_session_id", # this need to be unique for each user
    ttl=3600 # time to live in seconds
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly Assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

#Attaching the memory to the chain
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    chat_memory=history #integrating Redis to the chat
)

search = TavilySearchResults()
retriever_tool = create_retriever_tool(
    retriever,
    "lcel_search",
    "Use this tool when seaching for information about LCEL or Langchain Expression Language")

tools = [search, retriever_tool]

agent = create_openai_functions_agent(
    llm = model,
    prompt = prompt,
    tools = tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory # usar o código do memory.py
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({	
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]

if __name__ == "__main__":
    chat_history = [
    ]

    while True:
        user_input = input("You: ") 
        if user_input.lower() == 'exit':
            break
        res = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=res))
        print("Assistent: ", res)
