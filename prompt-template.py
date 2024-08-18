# Prompt template allows you to take control of the conversation 
# and we can reformat the user`s prompt in a way that we want.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if __name__ == '__main__':

    # instantiate the model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.2, # quanto menor, mais conservador
    )

    # Instatiating the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
            ("human", "{input}")
        ]
    )

    # create a LLM chain
    chain = prompt | llm
    
    # invoke the chain
    #subject will be replaced by "dog" and will be passed to the model via prompt object
    res = chain.invoke({"input": "tomatoes"})

    print(res)