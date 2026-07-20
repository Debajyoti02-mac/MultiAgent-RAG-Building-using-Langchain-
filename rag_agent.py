import os
import re
import ast
import operator
import hashlib
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize settings
DB_PATH = "./Database_VD"
COLLECTION_NAME = "clean_vectorDB"

def get_chroma_client():
    """Initializes and returns the persistent Chroma client."""
    return chromadb.PersistentClient(path=DB_PATH)

def get_collection(client):
    """Gets or creates the Chroma document collection."""
    embedding_function = DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=embedding_function
    )

def safe_calculator(expression: str) -> str:
    """Safely evaluates basic mathematical expressions without using eval()."""
    allowed_nodes = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd
    }
    
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: lambda x: x
    }
    
    def _eval(node):
        if not type(node) in allowed_nodes:
            raise TypeError(f"Unsupported syntax node: {type(node).__name__}")
        
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in operators:
                raise TypeError(f"Unsupported binary operator: {op_type.__name__}")
            if op_type == ast.Div and right == 0:
                return "Error: Division by zero"
            if op_type == ast.Mod and right == 0:
                return "Error: Modulo by zero"
            return operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in operators:
                raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
            return operators[op_type](operand)
        else:
            raise TypeError(f"Unsupported AST node: {type(node).__name__}")

    try:
        # Convert caret power representation to standard python power representation
        clean_expr = expression.replace('^', '**').strip()
        # Remove common arithmetic noise like '=' or trailing space
        clean_expr = clean_expr.rstrip('=')
        if not re.fullmatch(r'[\d\s\+\-\*\/\.\(\)\%\*\*]+', clean_expr):
            return "Error: Invalid characters in mathematical expression"
            
        tree = ast.parse(clean_expr, mode='eval')
        result = _eval(tree)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def ingest_pdf(pdf_path: str, collection) -> int:
    """Ingests a PDF file, splits it into chunks, and saves to Chroma DB."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=140)
    texts = text_splitter.split_documents(pages)
    
    chunks = [doc.page_content for doc in texts]
    metadatas = [doc.metadata for doc in texts]
    
    # Generate content-hash IDs to prevent duplicates and make it idempotent
    ids = []
    for chunk in chunks:
        hasher = hashlib.md5()
        hasher.update(chunk.encode('utf-8'))
        ids.append(hasher.hexdigest())
        
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)

def query_local_retrieval(query: str, collection, threshold: float = 1.5) -> list:
    """Queries Chroma DB and returns chunks that satisfy the similarity threshold."""
    if collection.count() == 0:
        return []
        
    result = collection.query(query_texts=[query], n_results=5)
    distances = result['distances'][0]
    documents = result['documents'][0]
    
    relevant_chunks = []
    for dist, doc in zip(distances, documents):
        if dist < threshold:
            relevant_chunks.append(doc)
            
    return relevant_chunks

def execute_web_search(query: str) -> str:
    """Runs a DuckDuckGo web search as fallback."""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search tool failed: {str(e)}"

def grade_retrieved_chunks(query: str, chunks: list, chat_model) -> bool:
    """
    Uses the LLM to grade if the retrieved document chunks are relevant to the query.
    Returns True if at least one chunk is relevant, False otherwise.
    """
    if not chunks:
        return False
        
    context = "\n\n".join(chunks)
    prompt = (
        "You are an expert evaluator. Your task is to determine if the retrieved document sections "
        "contain relevant information to answer the user query. "
        "You must answer with either 'YES' or 'NO'. Do not explain or write anything else.\n\n"
        f"Retrieved Document Sections:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Are these sections relevant and contain information to help answer the user query? (YES/NO):"
    )
    
    try:
        response = chat_model.invoke(prompt)
        decision = response.content.strip().upper()
        return "YES" in decision
    except Exception:
        # Fallback to True if LLM call fails
        return True

def route_query(query: str, chat_model=None) -> str:
    """Determines route: 'cal' for math expression, 'retrive' otherwise."""
    # Regex checks for pure math expressions (numbers, whitespace, mathematical operators, power ^, modulo %)
    clean_query = query.strip().rstrip('=').strip()
    if re.fullmatch(r'[\d\s\+\-\*\/\.\(\)\%\^\*\*]+', clean_query):
        return "cal"
        
    # If not a pure math expression, use LLM as a fallback routing classifier
    if chat_model is not None:
        prompt = (
            "You are an expert query router. Classify the user query into one of two routes:\n"
            "1. 'cal' (if the query is asking to perform a mathematical calculation, equation solving, or arithmetic, "
            "even if written in words, e.g. 'calculate 5 times 6', 'what is 2 + 2?', 'sum of 15 and 20')\n"
            "2. 'retrive' (for any other question, search query, or informational request)\n\n"
            "Respond with ONLY the string 'cal' or 'retrive'. Do not include any other text, explanation, or punctuation.\n\n"
            f"User Query: {query}"
        )
        try:
            response = chat_model.invoke(prompt)
            decision = response.content.strip().lower()
            if 'cal' in decision:
                return "cal"
        except Exception:
            pass
            
    return "retrive"

def run_agent(query: str, collection, chat_model, state: dict = None) -> tuple:
    """
    Executes the Multi-Agent routing flow.
    Returns: (answer, routing_metadata, updated_state)
    """
    state = state or {"history": []}
    routing_metadata = {
        "route": "unknown",
        "details": {},
        "fallback_taken": False
    }
    
    label = route_query(query, chat_model)
    
    if label == "cal":
        # Calculate
        routing_metadata["route"] = "Calculator"
        
        # If expression contains text (natural language calculation query), translate it using LLM
        clean_expr = query.strip().rstrip('=').strip()
        if re.search(r'[a-zA-Z]', clean_expr):
            try:
                translation_prompt = (
                    "Convert the following mathematical/arithmetic query into a clean Python mathematical expression "
                    "that can be evaluated using standard operators (+, -, *, /, **, (, )). "
                    "Convert power notation like '^' to '**'. Convert words like 'times', 'multiplied by', 'divided by', "
                    "'plus', 'minus', 'sum of x and y' into their standard operators. "
                    "Do not include any words, letters, variables, or functions. Only return the final expression.\n\n"
                    f"Query: {query}\n"
                    "Expression:"
                )
                response = chat_model.invoke(translation_prompt)
                expr_to_eval = response.content.strip()
                expr_to_eval = expr_to_eval.replace("Expression:", "").strip()
                expr_to_eval = expr_to_eval.replace("`", "").strip()
            except Exception:
                expr_to_eval = clean_expr
        else:
            expr_to_eval = clean_expr
            
        answer = safe_calculator(expr_to_eval)
        routing_metadata["details"] = {"expression": expr_to_eval}
        
    else:
        # Document retrieval route
        routing_metadata["route"] = "Local PDF Retrieval"
        chunks = query_local_retrieval(query, collection, threshold=1.5)
        
        # Grade the retrieved chunks using LLM relevance grader
        is_relevant = grade_retrieved_chunks(query, chunks, chat_model)
        
        if not is_relevant:
            # Fallback to web search
            routing_metadata["fallback_taken"] = True
            routing_metadata["route"] = "Web Search Fallback"
            
            web_results = execute_web_search(query)
            routing_metadata["details"] = {"web_search_results": web_results}
            
            # Synthesize answer using web search results
            prompt = (
                f"You are a helpful assistant. Use the following DuckDuckGo web search results "
                f"to answer the user's query as accurately and directly as possible. "
                f"If you cannot find the answer in the results, summarize what was found or answer generally.\n\n"
                f"Search Results:\n{web_results}\n\n"
                f"User Query: {query}"
            )
            try:
                response = chat_model.invoke(prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error calling ChatGroq: {str(e)}\n\nRaw Search Results:\n{web_results}"
        else:
            # Found relevant chunks
            routing_metadata["details"] = {"retrieved_chunks": chunks}
            context_text = "\n\n".join(chunks)
            
            # Synthesize answer using local PDF chunks
            prompt = (
                f"You are a helpful assistant. Use the following retrieved document sections "
                f"to answer the user's query as accurately and directly as possible.\n\n"
                f"Document Sections:\n{context_text}\n\n"
                f"User Query: {query}"
            )
            try:
                response = chat_model.invoke(prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error calling ChatGroq: {str(e)}\n\nRaw Retrived Content:\n{context_text}"
                
    # Update history
    state["history"].append({
        "query": query,
        "answer": answer,
        "route": routing_metadata["route"],
        "fallback_taken": routing_metadata["fallback_taken"]
    })
    
    return answer, routing_metadata, state
