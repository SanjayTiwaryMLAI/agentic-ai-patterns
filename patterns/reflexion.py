"""Reflexion Agent — Self-reflection and retry on failure"""
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


class ReflexionAgent:
    def __init__(self, model: str = "gpt-4o-mini", max_reflections: int = 3):
        self.llm             = ChatOpenAI(model=model, temperature=0)
        self.search          = TavilySearchResults(max_results=3)
        self.max_reflections = max_reflections

    def _attempt(self, question: str, memory: str) -> str:
        results = self.search.invoke(question)
        ctx     = "\n".join([r["content"] for r in results])
        return self.llm.invoke(f"{memory}\nContext: {ctx}\nAnswer: {question}").content

    def _reflect(self, question: str, answer: str) -> tuple[bool, str]:
        r = self.llm.invoke(
            f"Q: {question}\nA: {answer}\nIs this SUFFICIENT or INSUFFICIENT? Explain."
        ).content
        return r.upper().startswith("SUFFICIENT"), r

    def run(self, question: str) -> str:
        memory = ""
        for i in range(1, self.max_reflections + 1):
            print(f"  Attempt {i}/{self.max_reflections}")
            answer = self._attempt(question, memory)
            ok, reflection = self._reflect(question, answer)
            if ok:
                return answer
            memory += f"\nAttempt {i} INSUFFICIENT: {reflection}\n"
        return answer
