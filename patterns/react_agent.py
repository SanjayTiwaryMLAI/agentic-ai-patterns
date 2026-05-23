"""ReAct Agent — Interleaved Reasoning + Acting"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults


class ReActAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm    = ChatOpenAI(model=model, temperature=0)
        self.tools  = [TavilySearchResults(max_results=3)]
        self.prompt = hub.pull("hwchase17/react")

    def run(self, question: str) -> str:
        agent    = create_react_agent(self.llm, self.tools, self.prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=8)
        return executor.invoke({"input": question})["output"]
