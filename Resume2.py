# %%
from langchain_openai import ChatOpenAI

# %%
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List,Optional
class User_Input(BaseModel):
    cv:str
    jd:str
class Output(BaseModel):
    match_score:int
    matching_skills:list[str]
    missing_skills:list[str]
    seniority_mismatch: Optional[str] = None
    recommendation:str

# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
DATA_FOLDER = Path('./Data')

def load_query_pdf(path):
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    return " ".join([d.page_content for d in docs])


def createDataDict():
    data={}
    print(f"Checking DATA_FOLDER: {DATA_FOLDER}")
    print(f"Does it exist? {DATA_FOLDER.exists()}")
    
    if not DATA_FOLDER.exists():
        print(f"ERROR: Data folder not found at {DATA_FOLDER.absolute()}")
        print("Current working directory:", Path.cwd())
    pair=[ p for p in DATA_FOLDER.iterdir()]
    for i , d in enumerate(pair):
        files = list(d.iterdir())
        data[d.name]={
            'cv':files[0],
            'jd':files[1]
        }
    return data

def load_Data_and_create_Chunks(data:dict):
    chunks={}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
        )
    for key,value in data.items():
        cv_path = value["cv"]
        jd_path = value["jd"]
        cv_loader = PyPDFLoader(str(cv_path))
        jd_loader = PyPDFLoader(str(jd_path))
        cv_docs = cv_loader.load()
        jd_docs = jd_loader.load()
        # for doc in cv_docs:
        #     print(doc.page_content)
        cv_chunks= splitter.split_documents(cv_docs)
        jd_chunks= splitter.split_documents(jd_docs)
        # print(cv_chunks)

        chunks[key] = {
            "cv_chunks": [{"content": chunk.page_content} for chunk in cv_chunks],
            "jd_chunks": [{"content": chunk.page_content} for chunk in jd_chunks]
        }
    return chunks


# %%
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import json
def create_vectorDB(chunks,api_key):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
                )

    all_texts = []
    all_metadatas = []

    for candidate_id, value in chunks.items():

        # CV chunks
        for chunk in value["cv_chunks"]:
            all_texts.append(chunk["content"])
            all_metadatas.append({
                "candidate": candidate_id,
                "source": "cv"
            })

        # JD chunks
        for chunk in value["jd_chunks"]:
            all_texts.append(chunk["content"])
            all_metadatas.append({
                "candidate": candidate_id,
                "source": "jd"
            })

    vectorDB = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas
    )

    return vectorDB

def get_Analysis(cv,job_description,api_key):
   
    data = createDataDict()
    print("Got out of create Data Dict: ", data)
    chunks = load_Data_and_create_Chunks(data)
    print("Got out of load_Data_and_create_Chunks: ", chunks)

    vectorDB = create_vectorDB(chunks,api_key)
    retriever = vectorDB.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
    )

    cv_query=load_query_pdf(cv)
    query = f"""
    Retrieve relevant information to compare candidate skills with job requirements

    Focus on:
    - skills match
    - missing skills
    - experience level
    - technologies

    Candidate CV:
    {cv_query}

    Job Description:
    {job_description}
    """
    context=retriever.invoke(query)
    print("context from retiever",context)
    prompt = f"""
You are an expert Resume Analyzer and Job Matcher.

    Candidate CV:
    {cv_query}

    Job Description:
    {job_description}

    Retrieved context:
    {context}

    Tasks:
    - Calculate match score (0–100)
    - List matching skills
    - List missing skills
    - Highlight seniority mismatch
    - Provide final recommendation

    Output MUST be structured.
"Return the output as **valid JSON only**, no markdown, no extra text."
    {{
        "match_score": 0,
        "matching_skills": [],
        "missing_skills": [],
        "seniority_mismatch": "",
        "recommendation": ""
    }}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response)
    analysis_dict = json.loads(response.content)
    print(analysis_dict)
    analysis = Output(**analysis_dict) 
    return analysis

# %%
# if __name__ == "__main__":

#     testCV=Path("./TestCV.pdf")
#     job_description="""
# Frontend Developer (React / MERN) 

# We are seeking a skilled Frontend Developer to build modern, responsive, and user-friendly web applications. The ideal candidate is passionate about creating high-quality UI experiences and collaborating with cross-functional teams to deliver scalable products.

# Responsibilities

# Develop responsive web applications using React and modern JavaScript (ES6+).

# Translate UI/UX designs into clean, maintainable code.

# Integrate REST APIs and backend services.

# Optimize applications for performance, accessibility, and cross-browser compatibility.

# Collaborate with designers and backend developers to deliver end-to-end features.

# Maintain code quality using Git and best frontend practices.

# Participate in code reviews and continuous improvement.

# Requirements

# Strong knowledge of HTML5, CSS3, JavaScript, TypeScript.

# Experience with React (required) and familiarity with Next.js or Vue.js.

# Experience using Tailwind CSS or Material UI.

# Understanding of REST APIs and asynchronous programming.

# Familiarity with Git workflows.

# Basic backend knowledge (Node.js / Express) is a plus — MERN stack preferred.

# Preferred Qualifications

# Experience building dashboards or SaaS products.

# Knowledge of performance optimization and accessibility.

# Experience with Firebase or similar BaaS platforms.

# Understanding of state management (Redux, Context API).

# What You’ll Work On

# Interactive dashboards and web platforms.

# Responsive product interfaces.

# Feature development across the MERN stack.

# Performance improvements and UI enhancements.
# """
# result=get_Analysis(testCV,job_description)
# print(result)


# %%
# print(type(result.matching_skills))


