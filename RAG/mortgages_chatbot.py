
import os
import pdfplumber
import requests
import time
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_core.documents import Document
from langchain.schema import AIMessage, HumanMessage

# Load environment variables
load_dotenv()

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# User-Agent Header to Avoid Bot Detection
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive"
}

# Website URLs
urls = [
    "https://askaboutmortgages.co.uk/",
    "https://www.money.co.uk/mortgages/a-complete-guide-to-mortgages",
    "https://moneysavingguru.co.uk/info/what-will-a-mortgage-advisor-ask-me/",
    "https://www.citizensadvice.org.uk/debt-and-money/mortgage-problems/",
    "https://www.propertymark.co.uk/professional-standards/consumer-guides/buying-selling-houses/mortgage-guide.html",
    "https://www.experian.co.uk/consumer/mortgages/guides/what-is-a-mortgage.html",
    "https://www.which.co.uk/money/mortgages-and-property/mortgages/types-of-mortgage/mortgage-types-explained-aIGHA3F2WqyQ",
    "https://getmymortgage.co.uk/pre/2/remortgage-calculator?campaign=686517173&adgroup=1328211653800542&keyword=remortgage%20calculator%20uk&matchtype=b&network=o&device=c&creative=83013500763381&target=&adposition=&placement=",
    "https://www.which.co.uk/money/mortgages-and-property/mortgages",
    "https://moneytothemasses.com/owning-a-home/mortgages/the-10-most-popular-mortgage-questions"
]

# Function to Extract Text from a PDF
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        return "No PDF file found."
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except Exception:
        return "No text extracted from PDF."
    return text.strip() if text else "No text extracted from PDF."

# Set PDF Path
pdf_path = r"C:\Users\STA\Desktop\KFV\AI Mortgage Advisor Project-12345.pdf"
pdf_docs_text = extract_text_from_pdf(pdf_path)

# Function to Scrape and Clean Website Data
def scrape_website(url):
    try:
        time.sleep(3)  # Avoid bot detection
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove footer, navigation, and legal text
            for tag in soup.find_all(["footer", "nav", "script", "style", "header", "aside"]):
                tag.extract()
            text_content = soup.get_text(separator=" ", strip=True)
            return text_content
        return None
    except Exception:
        return None

# Scrape and Clean Website Data
docs = []
for url in urls:
    scraped_data = scrape_website(url)
    if scraped_data:  # Only add non-None data
        docs.append(scraped_data)

# Combine Scraped Web Content and PDF Content
all_docs = [Document(page_content=doc) for doc in docs if doc] + [Document(page_content=pdf_docs_text)]

# Split Documents into Smaller Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(all_docs)

# Path to save/load FAISS database
faiss_db_path = "faiss_db"

# Check if FAISS database exists
if os.path.exists(faiss_db_path):
    # Load FAISS database from disk
    print("Loading FAISS database from disk...")
    vectordb = FAISS.load_local(faiss_db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    # Create FAISS Vector Store for Retrieval
    print("Creating new FAISS database...")
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    # Save FAISS database to disk
    vectordb.save_local(faiss_db_path)
    print("FAISS database saved to disk.")

# Initialize Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # Get top 3 most relevant chunks

# Initialize ChatOpenAI Model
llm = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=1200, top_p=0.9)

# Mortgage-related keywords to detect mortgage questions
mortgage_keywords = [
    "mortgage", "home loan", "property", "house", "buy", "purchase", "interest rate",
    "down payment", "deposit", "first-time buyer", "fixed rate", "variable rate",
    "remortgage", "equity", "repayment", "loan-to-value", "ltv", "term", "lender",
    "borrowing", "credit score", "affordability", "amortization", "foreclosure",
    "capital", "conveyancing", "stamp duty", "equity release", "tracker", "offset"
]

# Predefined Questions for detailed mortgage application
predefined_questions = [
    "Is this a single person application or are you a partnership? i.e married, civil partnership or living together.",
    "Are you a first-time buyer?",
    "Please can you give me your date of birth and your partner's, if applicable.",
    "Are you in employment?",
    "If yes, please specify if you are employed or self-employed?",
    "How much do you earn per year, gross pay before tax and NI, also include any commission, bonus, overtime and any other income sources.",
    "Please can you tell me if you have any credit commitments, if yes please list them, examples are credit cards, loans or even other mortgage payments and pensions.",
    "How many dependants do you have, if any?",
    "If you have adverse credit, please tell me what it is?",
    "Do you have an existing property that you plan to sell to support your new mortgage? If yes â€“ what is it worth, how much do you have outstanding on the mortgage and what do you hope to sell it for?",
    "Finally, what is your committed expenditure. Examples are maintenance, school fees & nursery costs."
]

# Trigger Phrases for full mortgage application
trigger_phrases = [
    "I want to buy a house",
    "I need a home loan",
    "I want to apply for a mortgage",
    "Can you help me get a mortgage",
    "I need a mortgage for my new property",
    "help me apply for a mortgage"
]

# Function to Check if User Input Contains Mortgage Keywords
def is_mortgage_related(input_text):
    input_text = input_text.lower()
    # Check for trigger phrases (full application)
    for phrase in trigger_phrases:
        if phrase.lower() in input_text:
            return "application"
    # Check for mortgage keywords (informational)
    for keyword in mortgage_keywords:
        if keyword.lower() in input_text:
            return "information"
    return "general"

# Function to Retrieve Relevant Information from Vector Database
def get_relevant_info(query):
    try:
        # Get relevant documents from vector database
        relevant_docs = retriever.invoke(query)
        # Extract and format the content
        retrieved_info = "\n".join([doc.page_content for doc in relevant_docs])
        return retrieved_info
    except Exception as e:
        print(f"Error retrieving information: {e}")
        return ""

# Function to Ask Predefined Questions
def ask_predefined_questions():
    user_responses = {}
    for question in predefined_questions:
        print(f"\n🤖 AI: {question}")
        user_response = input("Your answer: ").strip()
        user_responses[question] = user_response
    return user_responses

# Function to Generate Response Using Retrieval-Augmented Generation
def generate_rag_response(user_input):
    # Get relevant information from vector database
    relevant_info = get_relevant_info(user_input)
    # Create a prompt that includes the retrieved information
    rag_prompt = f"""
    You are a knowledgeable mortgage advisor with expertise in UK mortgages, home loans, and property financing.
    Relevant information from our knowledge base:
    {relevant_info}
    User's question: {user_input}
    Based on the relevant information provided above, please give a comprehensive, detailed, and informative answer to the user's question.
    Include specific details like typical interest rates, down payment percentages, loan terms, or requirements when relevant.
    Avoid using markdown symbols like ** or *. Provide the answer in plain text.
    If the information is not available in the knowledge base, provide general mortgage advice but be honest about limitations.
    """
    # Generate response using the model with RAG
    response = llm.invoke([HumanMessage(content=rag_prompt)])
    return response.content

# Function to Generate Summary Using the Model
def generate_summary(user_responses):
    # Combine questions and answers into a prompt for the model
    qa_pairs = "\n".join([f"Q: {q}\nA: {a}" for q, a in user_responses.items()])
    # Prompt for the model to generate a summary
    summary_prompt = f"""
    You are a helpful mortgage advisor. Below are the questions and answers provided by the user. 
    Please summarize the information in a clear, concise, and conversational way. Avoid repeating the same information multiple times.
    Provide any relevant advice or next steps.
    Questions and Answers:
    {qa_pairs}
    Summary:
    """
    # Generate summary using the model
    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    return summary_response.content

# Function to Save User Data to a JSON File
def save_user_data(user_responses, summary, contact_info):
    user_data = {
        "user_responses": user_responses,
        "summary": summary,
        "contact_info": contact_info
    }
    # Save the data to a JSON file
    with open("user_data.json", "w") as file:
        json.dump(user_data, file, indent=4)
    print("✅ Your information has been saved successfully!")

# Chat Memory Storage (Max 10 messages: 5 user + 5 AI)
chat_memory = []

# Function to Update Chat Memory
def update_chat_memory(user_message, ai_message):
    global chat_memory
    # Add new messages
    chat_memory.append({"user": user_message, "ai": ai_message})
    # If memory exceeds 10 messages, remove the oldest one
    if len(chat_memory) > 10:
        chat_memory.pop(0)

# Chatbot Interaction
print("\n🚀 Welcome! Your AI Mortgage Advisor is ready. Ask me anything!")
chat_active = True
user_data = {}
last_message = ""
while chat_active:
    user_input = input("\nYour message: ").strip()
    if not user_input:
        continue  # Skip empty input
    if user_input.lower() in ["exit", "quit", "close"]:
        print("✅ Thank you for chatting! It was great talking to you. Have a wonderful day! 😊")
        break  # Exit the chat
    # Prevent duplicate messages
    if user_input == last_message:
        continue
    last_message = user_input
    # Check what type of query this is
    query_type = is_mortgage_related(user_input)
    # Handle based on query type
    if query_type == "application":
        # Full mortgage application process
        print("\n🤖 AI: Sure! I can help you with your mortgage application. Let me ask you a few questions to get started.")
        user_data = ask_predefined_questions()
        # Generate Summary Using the Model
        print("\n🤖 AI: Thank you for providing the information! Generating a summary for you...")
        summary = generate_summary(user_data)
        # Display Summary to the User
        print(f"\n🤖 AI: Here's a summary based on your answers:\n{summary}")
        # Ask if they want to save their information
        print("\n🤖 AI: Would you like to save your information for future contact regarding mortgage offers or advice? (Type 'yes' or 'no')")
        save_choice = input("Your choice: ").strip().lower()
        if save_choice == "yes":
            # Collect contact information
            print("\n🤖 AI: Please provide your contact information for future updates.")
            email = input("Your email: ").strip()
            phone = input("Your phone number: ").strip()
            # Save user data to a JSON file
            contact_info = {"email": email, "phone": phone}
            save_user_data(user_data, summary, contact_info)
        # Ask if they want to continue or exit
        print("\n🤖 AI: Would you like to continue chatting or exit? (Type 'continue' or 'exit')")
        continue_choice = input("Your choice: ").strip().lower()
        if continue_choice == "exit":
            print("✅ Thank you for chatting! It was great talking to you. Have a wonderful day! 😊")
            break
    elif query_type == "information":
        # Mortgage information query - use RAG
        response_text = generate_rag_response(user_input)
        # Update chat memory
        update_chat_memory(user_input, response_text)
        print(f"\n🤖 AI: {response_text}")
    else:
        # General chatbot interaction
        general_prompt = f"""
        You are a friendly and engaging mortgage advisor. Respond naturally and conversationally like a human. 
        Avoid using markdown symbols or unnecessary formatting. Provide the answer in plain text.
        Here's the user's message: '{user_input}'
        """
        general_response = llm.invoke([HumanMessage(content=general_prompt)])
        response_text = general_response.content
        # Update chat memory
        update_chat_memory(user_input, response_text)
        print(f"\n🤖 AI: {response_text}")