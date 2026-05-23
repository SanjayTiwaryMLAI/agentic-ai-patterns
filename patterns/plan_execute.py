"""Plan-and-Execute Agent — Plan upfront, execute each step"""
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


class PlanExecuteAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm    = ChatOpenAI(model=model, temperature=0)
        self.search = TavilySearchResults(max_results=3)

    def _plan(self, question: str) -> list[str]:
        resp = self.llm.invoke(f"Break into 3-5 numbered steps to answer: {question}").content
        return [l.strip() for l in resp.split("\n") if l.strip() and l[0].isdigit()]

    def _execute(self, step: str, context: str) -> str:
        results = self.search.invoke(step)
        txt     = "\n".join([r["content"] for r in results])
        return self.llm.invoke(f"Context: {context}\nStep: {step}\nSearch: {txt}\nAnswer:").content

    def run(self, question: str) -> str:
        context = ""
        for i, step in enumerate(self._plan(question), 1):
            print(f"  Step {i}: {step}")
            context += f"\nStep {i}: {self._execute(step, context)}"
        return self.llm.invoke(f"Q: {question}\nResearch:{context}\nFinal answer:").content
