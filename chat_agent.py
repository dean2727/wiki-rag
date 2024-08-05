# Allows us to initialize a query engine tool integrated with the Wiki search engine
# ReAct utilizes this tool to address user queries related to a Wikipage
from llama_index.tools import QueryEngineTool
# Lets us attribute metadata to the tool, mainly for tracking purposes
from llama_index.tools import ToolMetadata
# Library thats instrumental in constructing chat agent's user interface and interactions
import chainlit as cl
# Facilitates the initialization of a settings menu where users can make selections
from chainlit.input_widget import Select, TextInput
import openai
# Used for creating a ReAct agent
from llama_index.agent import ReActAgent
# Serves as a wrapper for ChatGPT API, making interactions smoother
from llama_index.llms import OpenAI
# Handy for overseeing events within the Llama Index - used to showcase agent interactions on chainlit UI
from llama_index.callbacks.base import CallbackManager

from index_wikipages import create_index
from utils import get_apikey

index = None


@cl.on_chat_start
async def on_chat_start():
    global index
    # Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="MODEL",
                label="Settings panel",
                values=["gpt-3.5-turbo", "gpt-4"],
                initial_index=0,
            ),            
            TextInput(
                id="WIKIPAGES",
                label="Request Wikipage",
                placeholder="Please index: Wikipedia page, Wikipedia page2, etc."
            )
        ]
    ).send()


def wikisearch_engine(index):
    query_engine = index.as_query_engine(
        response_model="compact",
        verbose=True,
        similarity_top_k=10
    )
    return query_engine


def create_react_agent(MODEL):
    """ 
    The agent determines if a tool is needed for a response, decides on the tool’s use,
    and evaluates its output to ensure the query is addressed. This loop continues until
    the query is resolved or a set iteration limit is reached.
    It consists of its tools it uses and the LLM that empowers it
    """
    query_engine_tools = [
        QueryEngineTool(
            query_engine=wikisearch_engine(index),
            metadata=ToolMetadata(
                name="Wikipedia Search",
                description="Useful for performing searches on the Wikipedia knowledgebase."
            )
        ),
    ]

    openai.api_key = get_apikey()
    llm = OpenAI(MODEL)
    agent = ReActAgent(
        tools=query_engine_tools,
        llm=llm,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        verbose=True
    )
    return agent


@cl.on_settings_update
async def setup_agent(settings):
    global agent
    global index
    query = settings.get("WIKIPAGES")
    index = create_index(query) # Vector DB now in memory

    print("on_settings_update", settings)
    MODEL = settings.get("MODEL")
    agent = create_react_agent(MODEL)
    await cl.Message(
        author="Agent", content=f"""Wikipage(s) "{query}" successfully indexed"""
    ).send()


@cl.on_message
async def main(message: str):
    if agent:
        response = await cl.make_async(agent.chat)(message)
        await cl.Message(author="Agent", content=response).send()
