# 🤖 Hybrid Chatbot System with LangChain & Gradio

A powerful, context-aware chatbot and agent system built using **LangChain** for intelligence and **Gradio** for a clean, accessible user interface. This project features a unique **Query Router** that dynamically decides the best response path for every user input, optimizing both speed and capability.

## ✨ Features

* **Hybrid Architecture:** Dynamically routes queries to either a Basic Chat Model or a specialized Agent based on complexity and tool requirements.
* **Intelligent Router:** An LLM-powered router classifies questions into `BASIC` (general chat) or `AGENT` (tool-requiring tasks).
* **Contextual Memory:** Utilizes LangChain's **`ConversationBufferWindowMemory (k=10)`** to ensure the Agent maintains state and remembers the last 10 conversation turns, enabling natural follow-up questions.
* **Extensive Toolset:** The Agent is equipped with a rich set of external tools (defined in `agent.py`):
    * **Web Search:** DuckDuckGo, Wikipedia, Arxiv.
    * **Real-time Data:** Weather Info Tool.
    * **Analysis:** Python REPL (for calculations/analysis), File Downloader.
    * **Multimedia:** YouTube Transcript, Image Captioner (via local file/Ollama).
* **Local LLM Integration:** Designed to work seamlessly with local LLM environments (e.g., `gemma3:27b`) exposed via an API endpoint (Ollama + Cloudflare Tunnel).

## 📐 Architecture Overview

The system operates based on a clear, two-path routing mechanism managed by the **`hybrid_response_with_router`** function:

1.  **Input:** A user sends a message.
2.  **Routing:** The **`route_question`** function queries a dedicated LLM to determine the query's classification: **`AGENT`** or **`BASIC`**.
3.  **BASIC Path:** For simple, conversational queries (e.g., greetings, opinions), the request is handled directly by a standard chat LLM.
4.  **AGENT Path (The Core):** For complex queries requiring external data or tools:
    * The **`BasicAgent`** automatically loads the full conversation history from the global **`ConversationBufferWindowMemory`** object.
    * The Agent uses the **ReAct framework** to reason, select and execute the necessary tool(s) (e.g., `WeatherInfoTool`, `general_web_search`), and form a final answer.
    * The Agent's input and output are immediately saved back to the memory object, ensuring context is available for the next turn.

## 🚀 Setup & Installation

### Prerequisites

1.  **Python 3.9+**
2.  **Ollama:** Running a compatible model (e.g., `gemma3:27b`) locally.
3.  **Cloudflare Tunnel:** Recommended for exposing your Ollama API endpoint publicly (or adjust the URL for local-only use).
4.  **API Key:** Set up your WeatherStack API key (or similar) in your environment:
    ```bash
    export WEATHER_API="YOUR_WEATHERSTACK_API_KEY"
    ```

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/baloglu321/Hibrit_Chatbot_with_langchain.git](https://github.com/baloglu321/Hibrit_Chatbot_with_langchain.git)
    cd Hibrit_Chatbot_with_langchain
    ```

2.  **Install Dependencies:**
    You will need to create a `requirements.txt` file listing all necessary packages (LangChain components, Gradio, requests, pandas, Pillow, yt-dlp, whisper etc.) and run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure URLs:**
    * Open `app.py` and `agent.py`.
    * Update the `CLOUDFLARE_TUNNEL_URL` variable to match your Ollama API's publicly accessible URL.

## ▶️ Usage

1.  **Run the application:**
    ```bash
    python app.py
    ```

2.  **Access the Interface:**
    * Open your web browser and navigate to the local URL provided by Gradio (e.g., `http://127.0.0.1:7860`).

3.  **Testing Memory (Example):**
    * **User 1:** "What is the current population of Turkey?"
    * **System:** (Routes to AGENT, uses search tool) "The current population is X."
    * **User 2 (Follow-up):** "Who is the founding leader of this country?"
    * **System:** (Routes to AGENT, uses memory and search) "The founding leader of Turkey is Mustafa Kemal Atatürk."
