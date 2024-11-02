import os
from indexer import CustomEmbeddingModel
from api_keys import GROQ_API_KEY, TAVILY_API_KEY
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_groq import ChatGroq
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START


class RagGraph:

    def __init__(self):
        self.import_api_keys()
        self.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
        self.embedding_model = CustomEmbeddingModel()
        self.initialise_retriever()
        self.initialise_retrieval_grader()
        self.initialise_query_writer()
        self.initialise_web_search_tool()
        self.initialise_rag_chain()
        self.initialise_rag_workflow()

    def import_api_keys(self):
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    def initialise_retriever(self):
        doc_content_info = "Weather and climate of the city"

        metadata_field_info = [
            AttributeInfo(name="city", description="The name of the city", type="string"),
            AttributeInfo(name="temperature", description="The temperature of the city", type="integer"),
            AttributeInfo(name="weather", description="The weather of the city", type="string"),
            AttributeInfo(name="climate", description="The climate of the city", type="string")
        ]

        vectorstore = ElasticsearchStore.from_documents([], self.embedding_model,
                                                        index_name="weather_rag", es_url="http://localhost:9200")

        self.retriever = SelfQueryRetriever.from_llm(self.llm, vectorstore, doc_content_info,
                                                metadata_field_info, verbose=True)

    def format_docs(self, docs):
        formatted_docs = []
        for i, doc in enumerate(docs):
            content = doc.page_content
            if doc.metadata.get("temperature"):
                doc.metadata.update({"temperature": f"{doc.metadata["temperature"]}°C"})
            metadata = "\n".join([f"{key}: {value}" for key, value in doc.metadata.items()])
            formatted_docs.append(f"Context {i + 1}:-\nContent: {content}\nMetadata: {metadata}")
            res = "\n\n".join(doc for doc in formatted_docs)
            return res

    def initialise_retrieval_grader(self):

        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""
            binary_score: str = Field(description="Documents are relevant to the question - 'yes' or 'no'")

        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        grader_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [("system", grader_system_prompt),
             ("human", "Retrieved document: \n\n {document} \n\n User question: {input}")])

        self.retrieval_grader = grade_prompt | structured_llm_grader

    def initialise_query_writer(self):
        rewrite_system_prompt = """You are a question re-writer that converts an input question to a better version that\
         is optimized for web search. Look at the input and try to reason about the underlying \
         semantic intent / meaning. While outputting, just output the improved query"""

        rewrite_prompt = ChatPromptTemplate.from_messages(
            [("system", rewrite_system_prompt),
             ("human", "Here is the initial question: \n\n {input} \n Formulate an improved question.")])

        self.query_rewriter = rewrite_prompt | self.llm | StrOutputParser()

    def initialise_web_search_tool(self, k=3):
        self.web_search_tool = TavilySearchResults(k=k)

    def initialise_rag_chain(self):
        template = """Answer the question based only on the following context. Don't try to make up an answer.
        {context}
        
        Question: {input}
        """

        prompt = ChatPromptTemplate(messages=[template])
        self.rag_chain = (RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
                          | prompt | self.llm | StrOutputParser())


    def retrieve(self, state):
        print("---RETRIEVE---")
        inp = state["input"]

        documents = self.retriever.get_relevant_documents(inp)
        # documents = history_aware_retriever.get_relevant_documents(question)
        return {"documents": documents, "input": inp}

    def generate(self, state):
        print("---GENERATE---")
        inp = state["input"]
        documents = state["documents"]

        generation = self.rag_chain.invoke({"context": documents, "input": inp})
        return {"documents": documents, "input": inp, "generation": generation}

    def grade_documents(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        inp = state["input"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for doc in documents:
            score = self.retrieval_grader.invoke({"input": inp, "document": self.format_docs([doc])})
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        web_search = "No" if filtered_docs else "Yes"
        return {"documents": filtered_docs, "input": inp, "web_search": web_search}

    def rewrite_query(self, state):
        print("---REWRITE QUERY---")
        inp = state["input"]
        documents = state["documents"]

        query_rewritten = self.query_rewriter.invoke({"input": inp})
        print(f"---modified_query: {query_rewritten}---")
        return {"documents": documents, "input": query_rewritten}

    def web_search(self, state):
        print("---WEB SEARCH---")
        inp = state["input"]
        documents = state["documents"]

        docs = self.web_search_tool.invoke({"query": inp})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "input": inp}

    def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]

        if web_search == "Yes":
            print("---DECISION: NONE OF THE DOCUMENTS ARE RELEVANT TO QUERY, REWRITE QUERY---")
            return "rewrite_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def create_graph_nodes(self):
        # Define the nodes
        self.workflow.add_node("retrieve", self.retrieve)  # retrieve
        self.workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        self.workflow.add_node("generate", self.generate)  # generatae
        self.workflow.add_node("rewrite_query", self.rewrite_query)  # transform_query
        self.workflow.add_node("web_search_node", self.web_search)  # web search

    def create_graph_edges(self):
        # Build graph
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents", self.decide_to_generate,
            {"rewrite_query": "rewrite_query", "generate": "generate"})
        self.workflow.add_edge("rewrite_query", "web_search_node")
        self.workflow.add_edge("web_search_node", "generate")
        self.workflow.add_edge("generate", END)

    def initialise_rag_workflow(self):
        self.workflow = StateGraph(GraphState)
        self.create_graph_nodes()
        self.create_graph_edges()
        self.app = self.workflow.compile()


class GraphState(TypedDict):
    """ Represents the state of our graph. """
    input: str
    generation: str
    web_search: str
    documents: List[str]


if __name__ == "__main__":
    rag_graph = RagGraph()
    rag_app = rag_graph.app
    # question = "What's the temperature in London?"
    question = "What's the temperature in Fort Worth?"
    res = rag_app.invoke({"input": question})
    print(res)
    print('Human:', question)
    print('AI:', res['generation'])
