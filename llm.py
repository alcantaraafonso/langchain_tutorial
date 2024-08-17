from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

if __name__ == '__main__':
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2, # quanto menor, mais conservador
        max_tokens=1000, # tamanho do texto gerado
        verbose=True # mostra o texto gerado
    )

    res = llm.invoke("Hello, how are you?")
    
    print(res)

    # batch
    # in this case, the model will generate a response for each input in parallel
    # res = llm.batch([
    #     "Hello, how are you?",
    #     "What is the meaning of life?"
    # ])

    # stream
    # in this case, the model will generate a response for each input in sequence in chunk
    # res = llm.stream("Hello, how are you?")
    # for chunk in res:
    #     print(chunk, end="", flush=True)