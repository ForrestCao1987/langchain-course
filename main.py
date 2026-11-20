import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

llm = ChatOpenAI(
    model="qwen3.7-max",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    temperature=0,
    timeout=3000,
)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    human = [HumanMessage(content="Dalian YingBo 最新引援信息")]
    result = agent.invoke({"messages": human})
    print(result)


if __name__ == "__main__":
    main()
