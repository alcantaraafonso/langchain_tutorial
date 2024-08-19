#Output parsers are used to parse the output of the model into a structured format legible to the user.
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
#Pydantic is a data validation and settings management using python type annotations
from langchain_core.pydantic_v1 import BaseModel, Field



if __name__ == '__main__':
    def call_string_otuput_parser():
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Tell me a joke about the following subject"),
                ("human", "{input}")
            ]
        )

        parser = StrOutputParser()

        chain = prompt | model | parser

        return chain.invoke({"input": "dog"})
    
    # returns each word in the response from LLM as a list
    def call_list_output_parser():
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
            ("human", "{input}")
        ])

        parser = CommaSeparatedListOutputParser()
        
        chain = prompt | model | parser

        return chain.invoke({
            "input": "happy"
        })   

    # returns the response from LLM as a json object
    def call_json_output_parser():
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract information from the following phrase.\nFormatting Instructions: {format_instructions}"),
            ("human", "{phrase}")
        ])

        class Person(BaseModel):
            recipe: str = Field(description="the name of the recipe")
            ingredients: list = Field(description="ingredients")
            

        # the output parser will follow the instructions provided by the class called Person
        parser = JsonOutputParser(pydantic_object=Person)

        chain = prompt | model | parser
        
        return chain.invoke({
            "phrase": "The ingredients for a Margherita pizza are tomatoes, onions, cheese, basil",
            "format_instructions": parser.get_format_instructions() # uses the pydantic object to get the format instructions
        }) 


    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.2, # quanto menor, mais conservador
    )

    # print(call_string_otuput_parser())
    # print(call_list_output_parser())
    print(call_json_output_parser())