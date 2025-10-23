# Vue-Ollama-Chat Server (Backend)

This is the Python-based WebSocket server for the `Vue-Ollama-Chat` project. It serves as the bridge between the Vue frontend and a local Ollama instance.

## Key Features

* **WebSocket Server**: Provides a real-time, bidirectional communication channel using the `websockets` library.
* **Ollama Integration**: Interfaces with the Ollama API (`/api/chat`) using the `requests` library to fetch responses from local language models, supporting streaming output.
* **Database Persistence**: Utilizes SQLite (`sqlite3`) to store conversation history (`conversations` and `messages` tables with a one-to-many relationship), enabling multi-turn dialogue context.
* **Context Management**: Loads and sends the full message history of a conversation to the Ollama `/api/chat` endpoint to maintain conversational context.
* **Client Communication**: Sends structured status updates, logs, and AI responses (streamed) back to the connected frontend client via WebSocket messages.

## Tech Stack

* Python (3.9+)
* `websockets`
* `requests`
* `sqlite3` (built-in)

## Quick Start

**Prerequisites**:
* Python 3.9+ installed.
* [Ollama](https://ollama.com/) running locally at `http://localhost:11434`.
* An Ollama model (e.g., `qwen:0.5b`) downloaded and configured in `webmanager.py`.

1.  **Clone**: `git clone https://github.com/SIAK-q/local-llm-chat-server.git && cd local-llm-chat-server`
2.  **Setup Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # .\venv\Scripts\activate  # Windows
    ```
3.  **Install Dependencies**: `pip install -r requirements.txt`
4.  **Initialize Database** (If `chat.db` doesn't exist): `python database.py`
5.  **Configure Model**: Edit `webmanager.py` (around line 140) to set the correct Ollama `"model"` name in the `payload`.
6.  **Run Server**: `python run_web.py`
    * The server will start at `ws://127.0.0.1:8765`.

## Frontend Repository

Requires the companion frontend interface: [Vue-Ollama-Chat-Client](https://github.com/SIAK-q/local-llm-chat-ui)