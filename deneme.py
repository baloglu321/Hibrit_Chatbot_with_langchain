from agent import *
from langchain_classic.memory import ConversationBufferWindowMemory

chat_history = ConversationBufferWindowMemory(k=10)


def append_chat_history(input, response):
    chat_history.save_context({"input": input}, {"output": response})
    return chat_history


class BasicAgent:

    def __init__(self):
        print("BasicAgent initialized.")
        self.agent = build_agent()

    def __call__(self, question: str) -> str:
        msg = {
            "input": question,
            "chat_history": chat_history.load_memory_variables({}),
        }
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = self.agent.invoke(msg)

        append_chat_history(fixed_answer["input"], fixed_answer["output"])
        fixed_answer = fixed_answer.get("output")
        if not isinstance(fixed_answer, (str, int, float)):
            fixed_answer = str(fixed_answer)
        print(f"Agent returning fixed answer: {fixed_answer}")

        return fixed_answer


try:
    agent = BasicAgent()
except Exception as e:
    print(f"Error instantiating agent: {e}")

agent("Hi My name is Ramesh")
agent("What is my name ?")
