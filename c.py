from langchain.llms import Groq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import Basemodel
import os
# Step 1: Define the state
class AgentState():
    pass

# Step 2: Initialize Groq LLM
llm = Groq(
    model="mixtral-8x7b-32768",  # or "llama3-70b-8192"
    api_key="your-groq-api-key"
)

# Step 3: Define a prompt template
prompt = PromptTemplate.from_template("Answer this question: {question}")

# Step 4: Define a LangGraph node
def answer_question(state: AgentState) -> AgentState:
    question = state["question"]
    formatted_prompt = prompt.format(question=question)
    response = llm.invoke(formatted_prompt)
    return {"question": question, "answer": response}

# Step 5: Build the LangGraph
graph = StateGraph(AgentState)
graph.add_node("answer", answer_question)
graph.set_entry_point("answer")
graph.set_finish_point(END)

# Step 6: Compile and run
app = graph.compile()
result = app.invoke({"question": "What is LangGraph?"})
print(result)
