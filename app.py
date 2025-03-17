import streamlit as st
import httpx
import asyncio
import torch
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check for GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_CASE_TEXT_LENGTH = 9500
MAX_CONCURRENT_REQUESTS = 100

executor = ThreadPoolExecutor(max_workers=10)  # Increased workers for speed
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Load model and tokenizer (download once and use locally)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    return tokenizer, model

tokenizer, model = load_model()

async def fetch_url(url, client):
    """Fetches a URL using a shared async client with controlled concurrency."""
    async with semaphore:
        response = await client.get(url, headers=HEADERS)
        return response

async def search_indiankanoon(query, client):
    """Search Indian Kanoon for cases based on a query using persistent client."""
    search_url = f"https://indiankanoon.org/search/?formInput={query}"
    response = await fetch_url(search_url, client)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "lxml")  # Using faster lxml parser
    return [
        {"title": link.text.strip(), "url": "https://indiankanoon.org" + link["href"]}
        for link in soup.select(".result_title a")[:10]
    ]

async def scrape_case(url, client):
    """Scrapes case details from Indian Kanoon using persistent client."""
    response = await fetch_url(url, client)

    if response.status_code != 200:
        return {"title": "Unknown", "text": "Failed to fetch case details.", "url": url}

    soup = BeautifulSoup(response.text, "lxml")
    paragraphs = [
        p.get_text(separator=" ", strip=True)
        for fragment in soup.select(".expanded_headline .fragment")
        for p in fragment.find_all("p")
    ]

    case_text = " ".join(paragraphs)[:MAX_CASE_TEXT_LENGTH] if paragraphs else "No case text found."
    return {"title": "Unknown", "text": case_text, "url": url}

def summarize_text(text):
    """Summarizes case text using DeepSeek model locally."""
    prompt = f"""
    You are an Indian legal AI assistant. Summarize the following court case with high accuracy, using only the provided text.
    Do NOT add assumptions or external knowledge on your own.

    Include:
    - Case Title (if available)
    - Key Dates (case, judgement, arrested, seen, call, evidence, person, action time and date.)
    - Laws, Acts, or Articles cited (verbatim)
    - Main Legal Issue (in brief)
    - Court's Decision & Reasoning (without opinion)
    - Precedent or Impact (if mentioned in the case text)

    Ensure the summary remains neutral, concise, and fact-based.

    Case Text: {text}...
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_length=500, temperature=0.7)

    return tokenizer.decode(output[0], skip_special_tokens=True)

async def process_case(case, client):
    """Processes a single case asynchronously without redundant fetching."""
    case_data = await scrape_case(case["url"], client)

    if not case_data["text"] or "Failed to fetch" in case_data["text"]:
        return {"title": case["title"], "summary": "Summary unavailable due to missing case text."}

    loop = asyncio.get_running_loop()
    summary = await loop.run_in_executor(executor, summarize_text, case_data["text"])
    return {"title": case["title"], "summary": summary}

async def fetch_and_process_cases(query):
    """Fetches case references and summarizes them asynchronously using a shared HTTP client."""
    async with httpx.AsyncClient(timeout=10) as client:
        cases = await search_indiankanoon(query, client)
        if not cases:
            return None

        tasks = [process_case(case, client) for case in cases]
        return await asyncio.gather(*tasks)

def run_async_task(query):
    """Runs an async task inside a synchronous function for Streamlit."""
    return asyncio.run(fetch_and_process_cases(query))

# Streamlit UI
st.title("Indian Legal AI - Case Reference & Summarization")

query = st.text_input("Enter your legal query:")
if st.button("Search"):
    with st.spinner("Fetching results..."):
        results = run_async_task(query)  # Run async function synchronously

    if not results:
        st.error("No results found.")
    else:
        summaries = [f"### {case['title']}\n\n{case['summary']}\n\n" for case in results]
        st.write("### Overall Insights for Lawyers:")
        st.info("\n".join(summaries))
