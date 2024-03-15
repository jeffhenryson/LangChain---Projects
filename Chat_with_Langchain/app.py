import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_vectorstores_from_urls(urls):
    
    all_document_chunks = []
    for url in urls:
        loader = WebBaseLoader(url)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        all_document_chunks.extend(document_chunks)

    combined_vector_store = Chroma.from_documents(all_document_chunks, OpenAIEmbeddings())

    return combined_vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0.8)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Olá! Eu sou seu assistente de LangChain, um professor especializado em projetos de LangChain. Estou aqui para ajudá-lo a montar sistemas, aprimorar prompts e agentes, corrigir códigos e dar ideias criativas para seus projetos. Além disso, estou sempre atualizado com as melhores práticas da indústria. Como posso ajudá-lo hoje? Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        return "No vector store available."

    vector_store = st.session_state.vector_store
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

website_urls_input = """
https://python.langchain.com/docs/get_started/introduction,
https://python.langchain.com/docs/get_started/installation,
https://python.langchain.com/docs/expression_language/get_started,
https://python.langchain.com/docs/expression_language/why,
https://python.langchain.com/docs/expression_language/interface,
https://python.langchain.com/docs/expression_language/streaming,
https://python.langchain.com/docs/expression_language/how_to/map,
https://python.langchain.com/docs/expression_language/how_to/passthrough,
https://python.langchain.com/docs/expression_language/how_to/functions,
https://python.langchain.com/docs/expression_language/how_to/routing,
https://python.langchain.com/docs/expression_language/how_to/binding,
https://python.langchain.com/docs/expression_language/how_to/configure,
https://python.langchain.com/docs/expression_language/how_to/decorator,
https://python.langchain.com/docs/expression_language/how_to/fallbacks,
https://python.langchain.com/docs/expression_language/how_to/generators,
https://python.langchain.com/docs/expression_language/how_to/inspect,
https://python.langchain.com/docs/expression_language/how_to/message_history,
https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser,
https://python.langchain.com/docs/expression_language/cookbook/retrieval,
https://python.langchain.com/docs/expression_language/cookbook/multiple_chains,
https://python.langchain.com/docs/expression_language/cookbook/sql_db,
https://python.langchain.com/docs/expression_language/cookbook/agent,
https://python.langchain.com/docs/expression_language/cookbook/code_writing,
https://python.langchain.com/docs/expression_language/cookbook/embedding_router,
https://python.langchain.com/docs/expression_language/cookbook/memory,
https://python.langchain.com/docs/expression_language/cookbook/moderation,
https://python.langchain.com/docs/expression_language/cookbook/prompt_size,
https://python.langchain.com/docs/expression_language/cookbook/tools,
https://python.langchain.com/docs/modules/,
https://python.langchain.com/docs/modules/model_io/,
https://python.langchain.com/docs/modules/model_io/concepts,
https://python.langchain.com/docs/modules/model_io/prompts/composition,
https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples,
https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples_chat,
https://python.langchain.com/docs/modules/model_io/prompts/message_prompts,
https://python.langchain.com/docs/modules/model_io/prompts/partial,
https://python.langchain.com/docs/modules/model_io/prompts/pipeline,
https://python.langchain.com/docs/modules/model_io/chat/quick_start,
https://python.langchain.com/docs/modules/model_io/chat/function_calling,
https://python.langchain.com/docs/modules/model_io/chat/chat_model_caching,
https://python.langchain.com/docs/modules/model_io/chat/custom_chat_model,
https://python.langchain.com/docs/modules/model_io/chat/logprobs,
https://python.langchain.com/docs/modules/model_io/chat/streaming,
https://python.langchain.com/docs/modules/model_io/llms/custom_llm,
https://python.langchain.com/docs/modules/model_io/llms/llm_caching,
https://python.langchain.com/docs/modules/model_io/llms/streaming_llm,
https://python.langchain.com/docs/modules/model_io/llms/token_usage_tracking,
https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start,
https://python.langchain.com/docs/modules/model_io/output_parsers/types/csv,
https://python.langchain.com/docs/modules/model_io/output_parsers/types/json,
https://python.langchain.com/docs/modules/model_io/output_parsers/types/openai_functions,
https://python.langchain.com/docs/modules/model_io/output_parsers/types/openai_tools,
https://python.langchain.com/docs/modules/model_io/output_parsers/types/pandas_dataframe,
https://python.langchain.com/docs/modules/model_io/output_parsers/types/pydantic,
https://python.langchain.com/docs/modules/model_io/output_parsers/types/retry,
https://python.langchain.com/docs/modules/model_io/output_parsers/types/structured,
https://python.langchain.com/docs/langgraph,
https://python.langchain.com/docs/modules/data_connection/document_loaders/,
https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory,
https://python.langchain.com/docs/modules/data_connection/document_loaders/html,
https://python.langchain.com/docs/modules/data_connection/document_loaders/json,
https://python.langchain.com/docs/modules/data_connection/document_loaders/markdown,
https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf,
https://python.langchain.com/docs/modules/data_connection/document_transformers/,
https://python.langchain.com/docs/modules/data_connection/document_transformers/character_text_splitter,
https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker,
https://python.langchain.com/docs/modules/data_connection/text_embedding/,
https://python.langchain.com/docs/modules/data_connection/vectorstores/,
https://python.langchain.com/docs/modules/data_connection/retrievers/,
https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore,
https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever,
https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression,
https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble,
https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder,
https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector,
https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever,
https://python.langchain.com/docs/modules/data_connection/indexing,
https://python.langchain.com/docs/modules/agents/agent_types/,
https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent,
https://python.langchain.com/docs/modules/agents/agent_types/openai_tools,
https://python.langchain.com/docs/modules/agents/agent_types/json_agent,
https://python.langchain.com/docs/modules/agents/agent_types/structured_chat,
https://python.langchain.com/docs/modules/agents/agent_types/react,
https://python.langchain.com/docs/modules/agents/agent_types/self_ask_with_search,
https://python.langchain.com/docs/modules/agents/how_to/custom_agent,
https://python.langchain.com/docs/modules/agents/how_to/streaming,
https://python.langchain.com/docs/modules/agents/how_to/structured_tools,
https://python.langchain.com/docs/modules/agents/how_to/agent_iter,
https://python.langchain.com/docs/modules/agents/how_to/agent_structured,
https://python.langchain.com/docs/modules/agents/how_to/handle_parsing_errors,
https://python.langchain.com/docs/modules/agents/how_to/intermediate_steps,
https://python.langchain.com/docs/modules/agents/tools/,
https://python.langchain.com/docs/modules/agents/tools/toolkits,
https://python.langchain.com/docs/modules/agents/tools/custom_tools,
https://python.langchain.com/docs/modules/agents/tools/tools_as_openai_functions,
https://python.langchain.com/docs/modules/chains,
https://python.langchain.com/docs/modules/memory/chat_messages/,
https://python.langchain.com/docs/modules/memory/adding_memory,
https://python.langchain.com/docs/modules/memory/adding_memory_chain_multiple_inputs,
https://python.langchain.com/docs/modules/memory/agent_with_memory,
https://python.langchain.com/docs/modules/memory/agent_with_memory_in_db,
https://python.langchain.com/docs/modules/memory/conversational_customization,
https://python.langchain.com/docs/modules/memory/custom_memory,
https://python.langchain.com/docs/modules/memory/multiple_memory,

"""
website_urls = [url.strip() for url in website_urls_input.split(",") if url.strip()]

if website_urls is None or website_urls == "":
    st.info("Please enter a website URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstores_from_urls(website_urls)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)