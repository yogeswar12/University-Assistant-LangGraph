#  Multi-Agent University Assistant using LangGraph

A multi-agent university assistant built using **LangGraph**, **LangChain**, **Ollama (Qwen2.5:3b)**, and **SQLite persistent memory**.

## Features

* ✅ Student profile extraction
* ✅ Supervisor/Router agent
* ✅ Four specialist agents:

  * Admissions
  * Hostel
  * Placement
  * Academic
* ✅ Knowledge-grounded responses
* ✅ SQLite persistent memory
* ✅ Multi-agent routing architecture

---

## Architecture

```text
                 USER
                   |
                   v
          extract_profile
                   |
                   v
            classify_query
                   |
      --------------------------------
      |        |        |           |
      v        v        v           v
 admission hostel placement academic
                   |
                   v
             SQLite Memory
```

---

## Technologies Used

* Python
* LangGraph
* LangChain
* Ollama
* Qwen2.5:3b
* SQLite
* Rich

---

## Installation

```bash
pip install -r requirements.txt
python university_assistant.py
```

---

## Example Query

```text
Hi, I'm Priya from CSE with CGPA 8.2.
What is the hostel curfew?
```

### Output

```text
Routed to: HOSTEL AGENT

Hi Priya! The hostel curfew is 10 PM.
No visitors are allowed after 8 PM.
```

---

## Concepts Demonstrated

* Multi-Agent Systems
* Supervisor Pattern
* LangGraph State Management
* Conditional Routing
* Knowledge Grounding (RAG-style)
* Persistent Memory
* Agent Orchestration
