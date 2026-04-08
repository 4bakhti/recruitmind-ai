# RecruitMind AI 🧠
**Multi-Agent CV Intelligence System**

## 📁 Project Structure
```text
recruitmind/
├── README.md
├── .gitignore
├── main.py                 #FastAPI application entry point
├── requirements.txt        #Project dependencies
├── core/
│   └── config.py           #Environment variables (API keys)
├── agents/
│   ├── orchestrator.py     #LangGraph state management
│   ├── agent_01_parser.py  #CV Parsing & Extraction
│   └── agent_02_crawler.py #GitHub/LinkedIn crawling
└── models/
    └── candidate.py        #Pydantic schemas for structured JSON output