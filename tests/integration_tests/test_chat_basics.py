"""Model-agnostic chat completion smoke tests -- general mechanics only
(streaming/non-streaming completion works, tool calling round-trips), no
assumptions about which model is actually loaded on the test port."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .conftest import stream_and_merge


def test_streaming_chat_produces_content_and_model_name(model):
    net = stream_and_merge(model.stream("what is your name?"))
    assert net.content
    assert net.response_metadata.get("model_name")


def test_non_streaming_chat_produces_content_and_model_name(model):
    ai_message = model.invoke("what is your name?")
    assert ai_message.content
    assert ai_message.response_metadata.get("model_name")


def test_agent_with_python_tool_function(model):
    def calculator(expression: str):
        """compute the value of an expression"""
        return eval(expression)

    agent = create_agent(model, tools=[calculator])
    result = agent.invoke({"messages": [
        HumanMessage("Use the calculator to find the value of 12.42111*124.33434344"),
    ]})

    message_one = result["messages"][1]
    assert isinstance(message_one, AIMessage)
    assert len(message_one.tool_calls) == 1
    tool_call = message_one.tool_calls[0]
    assert tool_call["name"] == "calculator"
    assert isinstance(result["messages"][2], ToolMessage)
