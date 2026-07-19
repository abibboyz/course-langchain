from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from course-langchain!")
    #llm=ChatOpenAI(model_name="gpt-3.5-turbo")
    information ="""
    Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire in terms of US dollars in June 2026; as of July 10, 2026, Forbes estimates his net worth to be US$917 billion.
    """

    summary_template = """ Given the {information} about a person I want you to create:
    1. A short summary
    2. two interesting fact about them
    """

    summary_prompt_template=PromptTemplate(input_variables=["information"],template=summary_template)

    #llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information":information})
    print(response.content )

    
if __name__ == "__main__":
    main()
