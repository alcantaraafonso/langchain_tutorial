# LLM aren't capable of get current information from the web, so we need to use a retrieval chain to get the information we need.
# We can load information from the web, pdfs, databases, etc. and pass it to the LLM.

from pyexpat import model
from dotenv import load_dotenv

import llm

load_dotenv()   

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

# nos permite criar uma chain com Prompt, model e output parser, mas passar uma lista de docs
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
if __name__ == '__main__':

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
            chunk_size=200,
            chunk_overlap=20
        )

        splitDocs = text_splitter.split_documents(docs)

        return splitDocs

    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.2, # the closer to zero we get, the more factual the response will be
    )

    prompt = ChatPromptTemplate.from_template("""
        Answer the user's question.
        Context: {context}
        Question: {input}
    """)

    chain = prompt | model
    # res = chain.invoke({
    #     "context": "The following is a list of the top 10 most populous countries in the world.",
    #     "input": "What is the most populous country in the world?"
    # })

    # res = chain.invoke({
    #     "context": [docA],
    #     "input": "What is LCEL?"
    # })


    # fetch the content from the web    
    docs = get_document_from_web('https://python.langchain.com/docs/expression_language/')


    # Se quisermos usar um retrievel chain, use o create_stuff_documents_chain
    res = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    ).invoke({ "context": docs,
         "input": "What is LCEL?"
    })

    print(res)