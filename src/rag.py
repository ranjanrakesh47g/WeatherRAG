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
from langchain_community.tools import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START


class RagGraph:

    def __init__(self):
        self.chat_history = []
        self.import_api_keys()
        # self.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
        # self.llm = ChatGroq(model="llama-3.2-90b-text-preview", temperature=0)
        self.llm = ChatGroq(model="llama-3.2-90b-vision-preview", temperature=0)
        self.embedding_model = CustomEmbeddingModel()
        self.initialise_retriever()
        self.initialise_retrieval_grader()
        self.initialise_web_search_tool()
        self.initialise_rag_chain()
        self.initialise_answer_grader()
        self.initialise_query_rewriter()
        self.initialise_rag_workflow()
        self.initialise_query_decomposer()
        self.initialise_history_aware_query_reformulator()
        self.initialise_conversation_aligner()
        self.initialise_rag_chain_main()
        self.initialise_rag_workflow_main()

    def invoke(self, query, recursion_limit=15):
        result = self.app_main.invoke({"input": query, "chat_history": self.chat_history},
                                      config={"recursion_limit": recursion_limit})
        self.chat_history += [query, result['answer']]
        return result

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
            Be slightly generous. Give a binary score 'yes' or 'no' score to indicate whether the document is \
            relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [("system", grader_system_prompt),
             ("human", "Retrieved document: \n\n {document} \n\n User question: {input}")])

        self.retrieval_grader = grade_prompt | structured_llm_grader

    def initialise_web_search_tool(self, k=5):
        rewrite_system_prompt = """You are a question re-writer that converts an input question to a better version \ 
        that is optimized for web search. Look at the input and try to reason about the underlying \ 
        semantic intent / meaning. While outputting, just output the improved query"""

        rewrite_prompt = ChatPromptTemplate.from_messages(
            [("system", rewrite_system_prompt),
             ("human", "Here is the initial question: \n\n {input} \n Formulate an improved question.")])

        self.query_rewriter_for_search = rewrite_prompt | self.llm | StrOutputParser()
        self.web_search_tool = TavilySearchResults(max_results=k, include_answer=True)

    def initialise_rag_chain(self):
        template = """Answer the question based only on the following context. Don't try to make up an answer.
        {context}
        
        Question: {input}
        """

        prompt = ChatPromptTemplate(messages=[template])
        self.rag_chain = (RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
                          | prompt | self.llm | StrOutputParser())

    def initialise_answer_grader(self):
        class GradeAnswer(BaseModel):
            """Binary score to assess answer addresses question."""
            binary_score: str = Field(description="Answer addresses the question - 'yes' or 'no'")

        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        answer_grader_system_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
             Be a generous grader. Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

        answer_prompt = ChatPromptTemplate.from_messages(
            [("system", answer_grader_system_prompt),
             ("human", "User question: \n\n {input} \n\n LLM generation: {generation}")])

        self.answer_grader = answer_prompt | structured_llm_grader

    def initialise_query_rewriter(self):
        rewrite_system_prompt = """You are a question re-writer that converts an input question to a better version \
         that is optimized for vectorstore retrieval and highly context-rich. Look at the input and try to reason \
         about the underlying semantic intent / meaning. While outputting, just output the improved query"""

        rewrite_prompt = ChatPromptTemplate.from_messages(
            [("system", rewrite_system_prompt),
             ("human", "Here is the initial question: \n\n {input} \n Formulate an improved question.")])

        self.query_rewriter = rewrite_prompt | self.llm | StrOutputParser()

    def initialise_history_aware_query_reformulator(self):
        contextualize_q_prompt = """You are a query reformulator who, given a chat history and \
        the latest user query, first thinks and checks \
        if the user query doen't contain any references to past history and hence can be answered as an independent query. \
        If YES, just return it as it is. \
        If NO, reformulate it into a standalone query with the coreferences \
        resolved, such that it can be understood without the chat history. \
        DO NOT answer the query, just return it as is or reformulate it into standalone query if needed. \
        While outputting, just output the actual/reformulated query.

        Chat History: {chat_history}
        User Query: {input}
        Reformulated Query:
        """

        contextualize_q_prompt = ChatPromptTemplate.from_template(contextualize_q_prompt)
        self.history_aware_query_reformulator = contextualize_q_prompt | self.llm | StrOutputParser()

    def initialise_query_decomposer(self):
        decomposition_system_prompt = """
        You are a helpful assistant that prepares queries that will be sent to a search component.
        Sometimes, these queries are very complex.
        Your job is to simplify complex queries into multiple queries that can be answered
        in isolation to each other. While outputting, just output the decomposed queries separated by '\n'.

        If the query is simple, then keep it as it is.
        Examples
        1. Query: Did Microsoft or Google make more money last year?
           Decomposed Queries: How much profit did Microsoft make last year?\nHow much profit did Google make last year?
        2. Query: What is the capital of France?
           Decomposed Queries: What is the capital of France?
        3. Query: What are the names of the top 10 richest people?
           Decomposed Queries: What are the names of the top 10 richest people?
        4. Query: Which cities have sunny weather?
           Decomposed Queries: Which cities have sunny weather?
        5. Query: Where is it sunny?
           Decomposed Queries: Where is it sunny?
        """

        decomposition_prompt = ChatPromptTemplate.from_messages(
            [("system", decomposition_system_prompt),
             ("human", "Here is the initial query: \n\n {input} \n Formulate the simpler decomposed queries.")])

        self.query_decomposer = (decomposition_prompt | self.llm | StrOutputParser() | (lambda x: x.split("\n"))
                                 | (lambda x: [x_i.strip() for x_i in x]))

    def initialise_conversation_aligner(self):
        conversational_align_prompt = """You are an AI coversational assistant. Given chat_history, \
        user query and candidate answer, check if the candidate answer is able to maintain the conversational flow. \
        If YES, return the answer as it is. \
        If NO, rephrase the candidate answer to more suit the query and the chat_history such that the conversation \
        seems more fluid and natural. Also, ensure that the rephrased answer is more independent \
        and does not include references to the past history. \
        DO NOT add anything new in the answer, just return it as it is or rephrase it to align \
        more with the conversation if needed. While outputting, just output the actual/rephrased answer.

        Chat History: {chat_history}
        User Query: {input}
        Candidate Answer: {candidate_answer}
        Answer:
        """

        conversational_align_prompt = ChatPromptTemplate.from_template(conversational_align_prompt)
        self.conversation_aligner = conversational_align_prompt | self.llm | StrOutputParser()

    def initialise_rag_chain_main(self):
        template = """Answer the question based only on the following context. Don't try to make up an answer.
        {context}

        Question: {input}
        """

        prompt = ChatPromptTemplate(messages=[template])
        self.rag_chain_main = prompt | self.llm | StrOutputParser()

    def retrieve(self, state):
        print("---RETRIEVE---")
        inp = state["input"]

        try:
            documents = self.retriever.get_relevant_documents(inp)
            return {"documents": documents, "input": inp, "retriever_exception": False}
        except:
            return {"documents": [], "input": inp, "retriever_exception": True}

    def decide_to_rewrite(self, state):
        print("---CHECK RETRIEVER FOR EXCEPTION---")
        retriever_exception = state["retriever_exception"]

        if retriever_exception:
            print("---DECISION: EXCEPTION OCCURRED! REWRITE QUERY---")
            return "exception"
        else:
            print("---DECISION: NO EXCEPTION! GRADE DOCUMENTS---")
            return "grade_documents"

    def rewrite_query(self, state):
        print("---REWRITE QUERY---")
        inp = state["input"]
        documents = state["documents"]

        query_rewritten = self.query_rewriter.invoke({"input": inp})
        print(f"---modified_query: {query_rewritten}---")
        return {"documents": documents, "input": query_rewritten}

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

    def web_search(self, state):
        print("---WEB SEARCH---")
        inp = state["input"]
        documents = state["documents"]

        # Re-write query
        query_rewritten = self.query_rewriter_for_search.invoke({"input": inp})
        print(f"---modified_query: {query_rewritten}---")

        docs = self.web_search_tool.invoke({"query": query_rewritten})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "input": query_rewritten}

    def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]

        if web_search == "Yes":
            print("---DECISION: NONE OF THE DOCUMENTS ARE RELEVANT TO QUERY, WEB SEARCH---")
            return "web_search"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_if_answers_question(self, state):
        print("---CHECK WHETHER GENERATION ANSWERS QUESTION---")
        inp = state["input"]
        generation = state["generation"]

        score = self.answer_grader.invoke({"input": inp, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ANSWERS QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ANSWERS QUESTION---")
            return "not_useful"

    def create_graph_nodes(self):
        # Define the nodes
        self.workflow.add_node("retrieve", self.retrieve)  # retrieve
        self.workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        self.workflow.add_node("web_search_node", self.web_search)  # web search
        self.workflow.add_node("generate", self.generate)  # generate
        self.workflow.add_node("rewrite_query", self.rewrite_query)  # rewrite_query

    def create_graph_edges(self):
        # Build graph
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_conditional_edges(
            "retrieve", self.decide_to_rewrite,
            {"exception": "rewrite_query", "grade_documents": "grade_documents"})
        self.workflow.add_conditional_edges(
            "grade_documents", self.decide_to_generate,
            {"web_search": "web_search_node", "generate": "generate"})
        self.workflow.add_edge("web_search_node", "generate")
        self.workflow.add_conditional_edges(
            "generate", self.grade_generation_if_answers_question,
            {"useful": END, "not_useful": "rewrite_query"})
        self.workflow.add_edge("rewrite_query", "retrieve")

    def initialise_rag_workflow(self):
        self.workflow = StateGraph(GraphState)
        self.create_graph_nodes()
        self.create_graph_edges()
        self.app = self.workflow.compile()

    def reformulate_query_using_history(self, state):
        print("---REFORMULATE QUERY USING HISTORY---")
        inp = state["input"]
        chat_history = state["chat_history"]

        if chat_history:
            reformulated_query = self.history_aware_query_reformulator.invoke({"input": inp, "chat_history": chat_history})
        else:
            reformulated_query = inp

        print(f"---reformulated_query: {reformulated_query}--")
        return {"input": inp, "chat_history": chat_history, "reformulated_query": reformulated_query}

    def decompose_query(self, state):
        print("---DECOMPOSE QUERY---")
        reformulated_query = state["reformulated_query"]

        sub_queries = self.query_decomposer.invoke({"input": reformulated_query})
        print(f"---decomposed_queries: {sub_queries}--")
        return {"reformulated_query": reformulated_query, "sub_queries": sub_queries}

    def answer_sub_queries(self, state):
        print("---ANSWER SUB-QUERIES---")
        sub_queries = state["sub_queries"]

        sub_answers = []
        for sub_query in sub_queries:
            sub_answer = self.app.invoke({"input": sub_query})
            sub_answer = sub_answer['generation']
            sub_answers.append(sub_answer)

        return {"sub_queries": sub_queries, "sub_answers": sub_answers}

    def answer_query(self, state):
        print("---ANSWER QUERY---")
        reformulated_query = state["reformulated_query"]
        sub_answers = state["sub_answers"]

        answer = self.rag_chain_main.invoke({"context": sub_answers, "input": reformulated_query})
        return {"reformulated_query": reformulated_query, "sub_answers": sub_answers, "candidate_answer": answer}

    def maintain_conversational_flow(self, state):
        print("---MAINTAIN CONVERSATIONAL FLOW---")
        chat_history = state["chat_history"]
        inp = state["input"]
        candidate_answer = state["candidate_answer"]

        answer = self.conversation_aligner.invoke(
            {"chat_history": chat_history, "input": inp, "candidate_answer": candidate_answer})
        return {"input": inp, "chat_history": chat_history, "candidate_answer": candidate_answer, "answer": answer}

    def create_graph_nodes_main(self):
        # Define the nodes
        self.workflow_main.add_node("reformulate_query_using_history",
                                    self.reformulate_query_using_history)  # reformulate query using history
        self.workflow_main.add_node("decompose_query", self.decompose_query)  # decompose query
        self.workflow_main.add_node("answer_sub_queries", self.answer_sub_queries)  # answer sub-queries
        self.workflow_main.add_node("answer_query", self.answer_query)  # answer query
        self.workflow_main.add_node("maintain_conversational_flow",
                                    self.maintain_conversational_flow)  # maintain conversational flow

    def create_graph_edges_main(self):
        # Build graph
        self.workflow_main.add_edge(START, "reformulate_query_using_history")
        self.workflow_main.add_edge("reformulate_query_using_history", "decompose_query")
        self.workflow_main.add_edge("decompose_query", "answer_sub_queries")
        self.workflow_main.add_edge("answer_sub_queries", "answer_query")
        self.workflow_main.add_edge("answer_query", "maintain_conversational_flow")
        self.workflow_main.add_edge("maintain_conversational_flow", END)

    def initialise_rag_workflow_main(self):
        self.workflow_main = StateGraph(GraphStateMain)
        self.create_graph_nodes_main()
        self.create_graph_edges_main()
        self.app_main = self.workflow_main.compile()


class GraphState(TypedDict):
    """ Represents the state of the util graph. """
    input: str
    retriever_exception: bool
    generation: str
    web_search: str
    documents: List[str]


class GraphStateMain(TypedDict):
    """ Represents the state of the main graph. """
    input: str
    chat_history: List[str]
    reformulated_query: str
    sub_queries: List[str]
    sub_answers: List[str]
    candidate_answer: str
    answer: str


if __name__ == "__main__":
    rag_graph = RagGraph()
    questions = ["What's the temperature in London?",
                 "What's the temperature in Fort Worth?",
                 "Which cities have temperate climate?",
                 "Where is it hottest?",
                 "Where is it raining?"]

    for question in questions:
        res = rag_graph.invoke(question)
        print('-'*50)
        print(res)
        print('Human:', question)
        print('AI:', res['answer'])
        print('Chat History:', rag_graph.chat_history)
