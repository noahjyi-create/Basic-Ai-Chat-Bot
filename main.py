import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_openai_tools_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
import json

# Load environment variables
load_dotenv()

# Define the output schema
class MessageSchema(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Output parser
parser = PydanticOutputParser(pydantic_object=MessageSchema)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Create agent
tools = [search_tool, wiki_tool, save_tool]   
agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


while True:
    query = input("What can I help you research? (type 'exit' to quit) ")
    if query.lower() == "exit":
        break
    raw_response = agent_executor.invoke({"query": query})
    output_str = raw_response["output"]
    # Remove code block markers if present
    output_str = output_str.replace("```json", "").replace("```", "").strip()
    try:
        output_json = json.loads(output_str)
        print("\n--- Agent Response ---")
        print(f"Topic: {output_json.get('topic', '')}")
        print(f"Summary: {output_json.get('summary', '')}")
        print("Sources:")
        for src in output_json.get('sources', []):
            print(f"  - {src}")
        print("Tools Used:")
        for tool in output_json.get('tools_used', []):
            print(f"  - {tool}")
    except Exception as e:
        print("Could not parse response. Raw output:")
        print(raw_response["output"])
