"""Side-by-side comparison of all agentic patterns"""
import time
from patterns.react_agent   import ReActAgent
from patterns.plan_execute  import PlanExecuteAgent
from patterns.reflexion     import ReflexionAgent

QUESTIONS = [
    "What are the top 3 LLM research breakthroughs in 2024?",
    "Compare LangGraph vs AutoGen for multi-agent systems.",
]


def benchmark():
    agents  = {"ReAct": ReActAgent(), "Plan-Execute": PlanExecuteAgent(), "Reflexion": ReflexionAgent()}
    for q in QUESTIONS:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        for name, agent in agents.items():
            t   = time.time()
            ans = agent.run(q)
            print(f"\n[{name}] ({time.time()-t:.1f}s):\n{ans[:400]}...")


if __name__ == "__main__":
    benchmark()
