
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain

#memory
# the memory module is used to store the conversation history
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory


history = UpstashRedisChatMessageHistory(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
    session_id="my_session_id", # this need to be unique for each user
    ttl=3600 # time to live in seconds
)

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly Assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

#Attaching the memory to the chain
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    chat_memory=history #integrating Redis to the chat
)

# chain = prompt | model
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def process_chat(chain, question):
    # Se quisermos usar um retrievel chain, use o create_stuff_documents_chain
    res = chain.invoke({
        "input": question
    })

    # print(res)
    return res["text"]

if __name__ == '__main__':

    while True:
        user_input = input("You: ") 
        if user_input.lower() == 'exit':
            break
        res = process_chat(chain, user_input)
        print("Assistent: ", res)
