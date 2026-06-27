# =============================================================
# CAPSTONE -- Multi-Agent University Assistant
# =============================================================

from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import json
import os
import sqlite3

console = Console()

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.3
)

DB_PATH = "university_memory.db"


# =============================================================
# UNIVERSITY KNOWLEDGE BASES
# =============================================================

ADMISSION_KB = {
    "btech_fees":
        "B.Tech fees: Rs 1,50,000 per year. Hostel: Rs 60,000 per year.",

    "eligibility":
        "Minimum 60% in Class 12 with Physics, Chemistry, Maths.",

    "courses":
        "B.Tech: CSE, AI&DS, ECE, Mechanical, Civil. "
        "M.Tech: AI, VLSI, Robotics.",

    "application":
        "Apply at admissions.vit.ac.in. Last date: March 31.",

    "entrance":
        "VITEEE conducted every April. "
        "125 questions, 2.5 hours.",
}

HOSTEL_KB = {
    "availability":
        "Boys: 4000 seats. Girls: 3500 seats. Apply early.",

    "room_types":
        "Single: Rs 80,000/yr. "
        "Double: Rs 60,000/yr. "
        "Triple: Rs 50,000/yr.",

    "rules":
        "Curfew: 10 PM. "
        "No visitors after 8 PM. "
        "Ragging strictly prohibited.",

    "mess":
        "Veg and non-veg options. "
        "Meal timings: 7AM, 1PM, 7PM. "
        "Monthly pass: Rs 3,500.",

    "facilities":
        "24x7 Wi-Fi, gym, common room, "
        "laundry service, medical center.",
}

PLACEMENT_KB = {
    "companies":
        "Microsoft, Amazon, Zoho, TCS, Infosys, "
        "Wipro, Cognizant, Accenture.",

    "packages":
        "Average: 6.5 LPA. "
        "Highest 2024: 42 LPA (Microsoft). "
        "Median: 5.8 LPA.",

    "internships":
        "Summer internships: Dec-Jan registrations. "
        "Companies: 200+.",

    "eligibility":
        "CGPA above 6.0 required for most companies. "
        "No active backlogs.",

    "preparation":
        "Training provided: aptitude, coding, "
        "communication. Starts Semester 5.",
}

ACADEMIC_KB = {
    "attendance":
        "Minimum 75% attendance required. "
        "Below 75%: not allowed in exams.",

    "credits":
        "B.Tech total: 160 credits. "
        "Each subject: 3-4 credits.",

    "semester":
        "2 semesters per year. "
        "Odd: July-Nov. "
        "Even: Jan-May. "
        "Summer: optional.",

    "grading":
        "GPA out of 10. "
        "Above 9: S grade. "
        "8-9: A. "
        "7-8: B. "
        "Pass: 5 and above.",

    "backlog":
        "Failed subjects can be cleared "
        "in supplementary exams or re-registration.",
}


# =============================================================
# STATE
# =============================================================

class UniversityState(TypedDict):
    messages: Annotated[list, add_messages]

    student_name: Optional[str]
    register_no: Optional[str]
    department: Optional[str]
    cgpa: Optional[str]

    query_category: str
    kb_context: str


# =============================================================
# NODE 1 : PROFILE EXTRACTOR
# =============================================================

def extract_profile(
        state: UniversityState
) -> UniversityState:

    last_msg = state["messages"][-1].content

    prompt = (
        "Extract student profile info from this message.\n"
        f'Message: "{last_msg}"\n\n'
        "Respond with ONLY valid JSON:\n"
        '{"name": null, '
        '"register_no": null, '
        '"department": null, '
        '"cgpa": null}'
    )

    result = llm.invoke(prompt)

    raw = (
        result.content
        .strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        extracted = json.loads(raw)

        if extracted.get("name") and not state.get("student_name"):
            state["student_name"] = extracted["name"]

        if extracted.get("register_no") and not state.get("register_no"):
            state["register_no"] = extracted["register_no"]

        if extracted.get("department") and not state.get("department"):
            state["department"] = extracted["department"]

        if extracted.get("cgpa") and not state.get("cgpa"):
            state["cgpa"] = str(extracted["cgpa"])

    except json.JSONDecodeError:
        pass

    return state


# =============================================================
# NODE 2 : SUPERVISOR / ROUTER
# =============================================================

def classify_query(
        state: UniversityState
) -> UniversityState:

    last_msg = state["messages"][-1].content

    prompt = (
        "Classify this university student query "
        "into ONE category:\n"
        "admission, hostel, placement, academic\n\n"

        "Query: " + last_msg + "\n\n"

        "Rules:\n"
        "- admission: fees, courses, eligibility, "
        "application, entrance exam\n"
        "- hostel: rooms, mess, curfew, "
        "facilities, hostel rules\n"
        "- placement: jobs, companies, salary, "
        "packages, internships\n"
        "- academic: attendance, grades, "
        "credits, semester, backlog\n\n"

        "Reply with ONLY one word:"
    )

    result = llm.invoke(prompt)

    category = result.content.strip().lower()

    valid = {
        "admission",
        "hostel",
        "placement",
        "academic"
    }

    state["query_category"] = (
        category
        if category in valid
        else "admission"
    )

    console.print(
        f"[cyan]Routed to: "
        f"[bold]{state['query_category'].upper()} "
        f"AGENT[/bold][/cyan]"
    )

    return state
# =============================================================
# NODE 3 : GENERIC DOMAIN AGENT
# =============================================================

def _agent_respond(
        state: UniversityState,
        knowledge_base: dict,
        domain_name: str
) -> UniversityState:
    """
    Generic agent:
    1. Search its knowledge base
    2. Build context
    3. Build prompt
    4. Ask LLM
    """

    last_msg = state["messages"][-1].content

    # ---------------------------------------------------------
    # Simple keyword retrieval
    # ---------------------------------------------------------
    relevant_entries = []

    for key, value in knowledge_base.items():

        if any(
            word in last_msg.lower()
            for word in key.split("_")
        ):
            relevant_entries.append(value)

    # fallback
    if not relevant_entries:
        relevant_entries = list(knowledge_base.values())

    context = "\n".join(relevant_entries[:4])

    state["kb_context"] = context

    # ---------------------------------------------------------
    # Student profile memory
    # ---------------------------------------------------------
    student_info = ""

    if state.get("student_name"):
        student_info += (
            "Student name: "
            + state["student_name"]
            + "\n"
        )

    if state.get("department"):
        student_info += (
            "Department: "
            + state["department"]
            + "\n"
        )

    if state.get("cgpa"):
        student_info += (
            "CGPA: "
            + state["cgpa"]
            + "\n"
        )

    # ---------------------------------------------------------
    # Final prompt
    # ---------------------------------------------------------
    prompt = (
        "You are the "
        + domain_name
        + " specialist at VIT University.\n\n"

        "Student information:\n"
        + (student_info if student_info else "Unknown\n")
        + "\n"

        "Relevant university information:\n"
        + context
        + "\n\n"

        "Student question: "
        + last_msg
        + "\n\n"

        "Answer helpfully in 3-5 sentences. "
        "Use student's name if known. "
        "Be specific with numbers and facts "
        "from the information above.\n\n"

        "Answer:"
    )

    result = llm.invoke(prompt)

    return {
        **state,
        "messages": [
            AIMessage(
                content=result.content.strip()
            )
        ]
    }


# =============================================================
# ADMISSION AGENT
# =============================================================

def admission_agent(
        state: UniversityState
) -> UniversityState:

    return _agent_respond(
        state,
        ADMISSION_KB,
        "Admissions"
    )


# =============================================================
# HOSTEL AGENT
# =============================================================

def hostel_agent(
        state: UniversityState
) -> UniversityState:

    return _agent_respond(
        state,
        HOSTEL_KB,
        "Hostel Administration"
    )


# =============================================================
# PLACEMENT AGENT
# =============================================================

def placement_agent(
        state: UniversityState
) -> UniversityState:

    return _agent_respond(
        state,
        PLACEMENT_KB,
        "Placement Cell"
    )


# =============================================================
# ACADEMIC AGENT
# =============================================================

def academic_agent(
        state: UniversityState
) -> UniversityState:

    return _agent_respond(
        state,
        ACADEMIC_KB,
        "Academic Affairs"
    )


# =============================================================
# ROUTER FUNCTION
# =============================================================

def agent_router(
        state: UniversityState
) -> str:

    return state["query_category"]

# =============================================================
# BUILD THE GRAPH
# =============================================================

def build_university_graph(db_path: str):

    builder = StateGraph(UniversityState)

    # Register nodes
    builder.add_node(
        "extract_profile",
        extract_profile
    )

    builder.add_node(
        "classify_query",
        classify_query
    )

    builder.add_node(
        "admission",
        admission_agent
    )

    builder.add_node(
        "hostel",
        hostel_agent
    )

    builder.add_node(
        "placement",
        placement_agent
    )

    builder.add_node(
        "academic",
        academic_agent
    )

    # Fixed edge
    builder.add_edge(
        "extract_profile",
        "classify_query"
    )

    # Conditional routing
    builder.add_conditional_edges(
        "classify_query",
        agent_router,
        {
            "admission": "admission",
            "hostel": "hostel",
            "placement": "placement",
            "academic": "academic",
        }
    )

    # Entry point
    builder.set_entry_point(
        "extract_profile"
    )

    # Finish points
    builder.set_finish_point(
        "admission"
    )

    builder.set_finish_point(
        "hostel"
    )

    builder.set_finish_point(
        "placement"
    )

    builder.set_finish_point(
        "academic"
    )

    # SQLite memory
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False
    )

    memory = SqliteSaver(conn)

    return builder.compile(
        checkpointer=memory
    )


# =============================================================
# MAIN PROGRAM
# =============================================================

if __name__ == "__main__":

    console.print(
        "\n[bold magenta]"
        "=== VIT University Multi-Agent Assistant ==="
        "[/bold magenta]"
    )

    console.print(
        f"[dim]Persistent memory: "
        f"{os.path.abspath(DB_PATH)}"
        f"[/dim]\n"
    )

    console.print(
        "[cyan]Enter your student ID "
        "or name as a session key:[/cyan]"
    )

    thread_id = (
        console.input(
            "[bold cyan]Session ID: "
            "[/bold cyan]"
        ).strip()
        or "student_001"
    )

    console.print(
        f"\n[green]Session: "
        f"[bold]{thread_id}[/bold] "
        f"| Type 'quit' to exit"
        f"[/green]\n"
    )

    graph = build_university_graph(DB_PATH)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    table = Table(
        title="Available Agents",
        style="bold"
    )

    table.add_column(
        "Agent",
        style="cyan",
        width=20
    )

    table.add_column(
        "Handles",
        style="white",
        width=50
    )

    table.add_row(
        "Admissions",
        "Fees, courses, eligibility, "
        "entrance exam, application"
    )

    table.add_row(
        "Hostel",
        "Room availability, mess, "
        "curfew, facilities, rules"
    )

    table.add_row(
        "Placement",
        "Companies, salary packages, "
        "internships, eligibility"
    )

    table.add_row(
        "Academic",
        "Attendance, GPA, credits, "
        "semester, backlog"
    )

    console.print(table)
    console.print()

    while True:

        user_input = console.input(
            "[bold cyan]You: [/bold cyan]"
        ).strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            console.print(
                "[yellow]"
                "Goodbye! "
                "Your conversation is saved."
                "[/yellow]"
            )
            break

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            },
            config=config
        )

        last_msg = result["messages"][-1]

        category = (
            result.get(
                "query_category",
                ""
            ).upper()
        )

        console.print(
            Panel(
                last_msg.content,
                title=f"[bold green]"
                      f"{category} AGENT"
                      f"[/bold green]",
                border_style="green"
            )
        )