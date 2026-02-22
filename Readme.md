# 🧠 Multi-Agent LLM Blog Generation System

A production-style **AI Agent system** built using LangGraph that goes far beyond simple prompt chaining.

This project implements a **planning-based, orchestrator–worker architecture** capable of:

- Planning before execution  
- Deciding when internet research is required  
- Breaking tasks into parallel subtasks  
- Using multiple worker agents  
- Adding citations and images automatically  
- Generating a complete blog end-to-end  

It demonstrates how modern AI agents are designed using structured multi-node workflows rather than single LLM calls.

---

## 🚀 Project Overview

Traditional LLM applications rely on:

> Prompt → Model → Output  

This system implements a **real agent architecture** using:

- Router
- Planner
- Worker Agents
- Reducer
- Research Tools
- Image Generation
- Structured State Management

Built using:

- **LangGraph** – for multi-agent orchestration  
- **LangChain** – tool & LLM integration  
- **Groq API** – LLM reasoning  
- **Google Gemini API** – image generation  
- **Tavily API** – internet research  
- **Python** – backend logic  
- **Streamlit** – interactive frontend  

---

# 🏗️ Architecture

The system follows a **Planning-Based Multi-Agent Architecture**.

User Topic
    ↓
Router Node
    ↓
Planner Node
    ↓
Parallel Worker Agents
    ↓
Research Tool (Tavily, if needed)
    ↓
Image Generator (Gemini)
    ↓
Reducer Node
    ↓
Final Blog Output


---

## 🧩 Core Components

### 1️⃣ Router Node

Decides:

- Is this a simple topic?
- Does it require research?
- Should it use hybrid or open-book mode?

Routes execution path accordingly.

---

### 2️⃣ Planner Node

Breaks the blog topic into structured sections:

- Introduction
- Technical Deep Dive
- Benchmark Analysis
- Case Studies
- Future Outlook

Creates parallel tasks for worker agents.

---

### 3️⃣ Worker Agents

Each worker:

- Writes a specific section
- Uses research tool if required
- Adds citations
- Generates structured output

Workers operate in **parallel**, improving scalability.

---

### 4️⃣ Research Tool Integration

Powered by **Tavily API**:

- Retrieves real-time internet information
- Filters evidence
- Extracts citations
- Supports recency filtering (e.g., last 7 days)

Enables research-powered blog generation.

---

### 5️⃣ Image Generation Module

Uses **Gemini Image API** to:

- Generate benchmark charts
- Create architecture diagrams
- Produce energy efficiency visuals

Includes:

- Automatic retry handling
- Quota detection
- Error fallback logic

---

### 6️⃣ Reducer Node

- Merges all worker outputs
- Formats markdown
- Cleans citations
- Assembles final blog

Produces polished output ready for publishing.

---

# 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph |
| LLM Framework | LangChain |
| LLM Provider | Groq, Ollama |
| Image Generation | Gemini |
| Research | Tavily |
| Backend | Python |
| Frontend | Streamlit |

---

# 🔬 Key Features

✅ Planning-based execution  
✅ Internet-aware agent  
✅ Parallel worker agents  
✅ Automatic citation insertion  
✅ Automatic image generation  
✅ Error-handling & retry logic  
✅ Research-mode routing  
✅ Production-style architecture  
✅ Interview-ready project  

---

# 📂 Project Structure


Multi-Agent-LLM-Blog-System/
│
├── Backend/
│ ├── main.py # LangGraph workflow definition
│ ├── prompts.py # System prompts
│ ├── models.py # LLMs
│ └── schemas.py # State definitions
│
├── frontend/
│ └── app.py # Streamlit interface
│
├── requirements.txt
└── README.md


---

# ⚙️ Installation



```bash
1️⃣ Clone Repository

git clone https://github.com/kshivayadav/multi-agent-llm-blog-system.git
cd multi-agent-llm-blog-system

2️⃣ Create Virtual Environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure Environment Variables
Create .env file:

OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key

▶️ Run Application

streamlit run frontend/app.py
```

# 🧠 Agent Execution Flow (Detailed)
Step 1: User enters blog topic

Example:

The Latest LLM Releases of 2026
Step 2: Router decides mode

> Closed-book | Hybrid | Open-book


Step 3: Planner creates section plan

Example:

Context

Recent Releases

Benchmarks

Case Studies

Future Outlook

Step 4: Parallel Workers execute

Each worker:

Writes section

Uses Tavily if needed

Adds citations

Generates image

Step 5: Reducer merges outputs

Final blog generated in markdown format.

## 📊 Why This Project is Different

Most AI projects:

Use single LLM call

Rely on prompt engineering

Have no orchestration

This project demonstrates:

Graph-based workflow

Deterministic node execution

Research-aware routing

Scalable parallelism

Real-world agent architecture

## 🚧 Known Challenges & Learnings

Handling Gemini API quota errors (429)

Managing async parallel execution

Ensuring citation credibility

Preventing hallucinated benchmarks

State management across nodes

## 🔮 Future Improvements

Vector DB memory integration

Caching research results

Cost-aware routing

Model auto-selection

Observability dashboard

Human-in-the-loop review mode

## 📌 Example Output Capabilities

The system can generate:

Research-backed AI blogs

Technical deep dives

Benchmark comparisons

Architecture explainers

Industry trend analysis

All fully formatted in Markdown.

## 📄 License

MIT License

## 🤝 Contribution

Contributions are welcome.

If you'd like to extend:

Add new tools

Improve planning logic

Integrate additional LLM providers

Enhance observability

Open a PR 🚀

## 👨‍💻 Author

K Shiva Kumar

Machine Learning & GenAI Engineer

## ⭐ Final Thoughts

This project showcases how modern AI agents are actually built:

Not just prompts —
but structured planning, orchestration, tool usage, and production-grade workflows.