# Langchain Tutorial
Tutorial baseado na série LangChain Tutorial (Python), disponível em: https://youtu.be/ekpnVh-l3YA?si=yxcA98jQwIS3QSPu

# Fluxo (Documents, Retrieval e Vector DB)
![alt text](image.png)

## Document Loader

### Prompt
```
    prompt = ChatPromptTemplate.from_template("""
        Answer the user's question.
        Context: {context}
        Question: {input}
    """)
```

### Context
É usado para dar um contexto para o LLM se basear
Ex.:
```
    docA = Document(
        page_content="LangChain Expression Language, or LCEL, is a declarative way to define the structure of a document.\
              It is a language that allows you to define the structure of a document in a way that is easy to read and write. \
                LCEL is a language that is designed to be human-readable and easy to understand"
    )
```

### Input
É a pergunta do usuário
```
    res = chain.invoke({
        "context": [docA],
        "input": "What is LCEL?"
    })
```

## Vector DB
Ainda que usemos o document pra fazer scraping, via DocumentLoader, de uma site e/ou doc, teremos um problema para resolver, pois o DocumentLoader busca o dado, mas
para darmos a resposta correta é necessário determinar a relevância da informação e é aqui que entra o Vector DB.
Um Vector DB usa uma técnica chamada semantic ou similarity search. O Vector DB vai retornar apenas os documentos mais relevantes ao contexto que foi passado na query

Exemplos de Vector DB são FAISS (para rodar In memory) e o Pinecone, que é um Serverless Vector DB


### Embed
Uma função de Embedding converte os documentos do DocumentLoader em um formato apropriado que um Vector DB aceita

