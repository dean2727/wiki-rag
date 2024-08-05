# A library that provides a framework for data validation.
import pydantic
# A library that allows you to use the large language models from OpenAI.
import openai
# A class to create an in-memory vector store (by default, it's bge-small-en)
from llama_index import VectorStoreIndex
# Utility container that helps configure indexing and querying tasks
from llama_index import ServiceContext
# Lets us decide how to preprocess the Wikipedia page(s) before indexing.
from llama_index.node_parser import SimpleNodeParser
# Used to set our text preprocessing arguments.
from llama_index.text_splitter import get_default_text_splitter
# Lets us define data structures
from pydantic import BaseModel
# Lets us get a structured output from an OpenAI model call
from llama_index.program import OpenAIPydanticProgram
from llama_index import download_loader
from llama_index.llms.openai import OpenAI

from utils import get_apikey
from typing import List


# define the data model in pydantic
class WikiPageList(BaseModel):
    "Data model for WikiPageList"
    pages: List[str]

def wikipage_list(query):
    openai.api_key = get_apikey()

    prompt_template_str = """
    Given the query of what to index, generate the list of desired pages to index.\n\
    Query: {query}
    """
    program = OpenAIPydanticProgram.from_defaults(
        output_cls=WikiPageList,
        llm=OpenAI(model="gpt-3.5-turbo"),
        prompt_template_str=prompt_template_str,
        verbose=True
    )

    wikipage_requests = program(query=query)

    return wikipage_requests


def create_wikidocs(wikipage_requests):
    WikipediaReader = download_loader(
        loader_class="WikipediaReader",
        loader_hub_url='https://raw.githubusercontent.com/run-llama/llama-hub/main/llama_hub'
    )

    loader = WikipediaReader()

    documents = loader.load_data(wikipage_requests)
    
    return documents


def create_index(query):
    global index

    wikipage_requests = wikipage_list(query)
    documents = create_wikidocs(wikipage_requests)
    
    text_splitter = get_default_text_splitter(chunk_size=150, chunk_overlap=45)

    parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

    service_context = ServiceContext.from_defaults(node_parser=parser)

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    return index


if __name__ == "__main__":
    query = "/get wikipages: paris, lagos, lao"
    index = create_index(query)
    print("INDEX CREATED", index)
