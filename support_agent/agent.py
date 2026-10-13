"""A support-reply agent built with LangGraph.

Shape (mirrors the Finch support agent):

    START -> classify --privacy/legal--> abstain -> END
                 |
                 v
             retrieve -> draft -> validate --ok--> human_review --approve/edit--> send -> END
                           ^          |                  |
                           |          |                  +--reject--> escalate -> END
                           +--repair--+  (once)
                                      |
                                      +--still invalid--> decline -> END

The four LangGraph ideas this file shows:

1. STATE   - one typed dict every node reads; each node returns only the keys it changes.
2. NODES   - plain Python functions: (state) -> partial state update.
3. EDGES   - fixed (`add_edge`) or conditional (`add_conditional_edges` + a router
             function), including a loop back for the repair turn.
4. CHECKPOINT + INTERRUPT - with a checkpointer, state is saved after every node.
             `interrupt()` pauses the run inside `human_review`; the process can exit,
             and `Command(resume=...)` later continues from exactly that point.
"""

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .kb import retrieve as kb_retrieve
from .llm import LLM

MAX_DRAFT_ATTEMPTS = 2  # first draft + one repair turn, then decline


# --- 1. STATE ---------------------------------------------------------------

class SupportState(TypedDict, total=False):
    ticket: str            # the customer's message
    customer: dict         # live account context (plan, banks, ...)
    category: str          # set by classify
    docs: list[dict]       # set by retrieve
    draft: str             # set by draft
    citations: list[str]   # KB ids the draft relies on
    errors: list[str]      # set by validate; fed back into the repair turn
    attempts: int          # how many drafts so far
    review: dict           # the human's decision, set by human_review
    outcome: str           # sent | escalated | declined | abstained
    final_reply: str


# --- 2. NODES ---------------------------------------------------------------
# Nodes that need the model are built by a factory so the LLM can be injected
# (real NVIDIA client in the CLI, scripted fake in the tests).

CLASSIFY_PROMPT = """You triage support tickets for a Swiss personal-finance app.
Return JSON only: {"category": one of ["bank_connection","billing","data","general","privacy_legal"],
"reason": short string}.
Use "privacy_legal" for data-protection requests (GDPR/FADP, deletion, who sees my data),
legal threats, or anything a lawyer should answer."""

DRAFT_PROMPT = """You draft replies for a human support agent to review and send.
Rules:
- Answer ONLY from the knowledge-base articles provided. If they do not cover it, say a colleague will follow up.
- Cite the article ids you used.
- Friendly, concise (2-5 sentences). No promises, no guarantees, never include account numbers.
Return JSON only: {"reply": string, "citations": [article ids]}."""


def make_classify(llm: LLM):
    def classify(state: SupportState) -> dict:
        result = llm.complete_json(CLASSIFY_PROMPT, state["ticket"])
        return {"category": result.get("category", "general")}
    return classify


def retrieve(state: SupportState) -> dict:
    return {"docs": kb_retrieve(state["ticket"])}


def make_draft(llm: LLM):
    def draft(state: SupportState) -> dict:
        articles = "\n\n".join(
            f"[{d['id']}] {d['title']}\n{d['body']}" for d in state.get("docs", [])
        ) or "(no matching articles)"
        user = (
            f"Customer plan: {state.get('customer', {}).get('plan', 'unknown')}\n"
            f"Ticket: {state['ticket']}\n\nKnowledge base:\n{articles}"
        )
        if state.get("errors"):  # the repair turn: tell the model what was wrong
            user += "\n\nYour previous draft was rejected for: " + "; ".join(state["errors"])
        result = llm.complete_json(DRAFT_PROMPT, user)
        return {
            "draft": result.get("reply", ""),
            "citations": result.get("citations", []),
            "attempts": state.get("attempts", 0) + 1,
        }
    return draft


BANNED = re.compile(r"\b(guarantee[ds]?|100%|definitely|promise[ds]?)\b", re.I)
IBAN_LIKE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9 ]{10,30}\b")


def validate(state: SupportState) -> dict:
    """Deterministic checks in code. Never ask the model to grade itself."""
    draft = state.get("draft", "")
    allowed = {d["id"] for d in state.get("docs", [])}
    cited = set(state.get("citations", []))
    errors = []
    if not 40 <= len(draft) <= 1200:
        errors.append(f"length {len(draft)} outside 40-1200 characters")
    if allowed and not cited:
        errors.append("no citations")
    if cited - allowed:
        errors.append(f"cites articles not provided: {sorted(cited - allowed)}")
    if BANNED.search(draft):
        errors.append("contains a promise or guarantee")
    if IBAN_LIKE.search(draft):
        errors.append("contains something that looks like an account number")
    return {"errors": errors}


def human_review(state: SupportState) -> dict:
    """Pause for a human. `interrupt` saves state and stops the run here.

    Whatever the caller passes to Command(resume=...) becomes the return value
    of interrupt(). Note: on resume, LangGraph re-runs this node from the top,
    so code before interrupt() must be safe to run twice (it is: no side effects).
    """
    decision = interrupt({
        "question": "Approve, edit or reject this draft?",
        "draft": state["draft"],
        "citations": state.get("citations", []),
    })
    return {"review": decision}


def send(state: SupportState) -> dict:
    review = state.get("review", {})
    text = review.get("text") if review.get("action") == "edit" else state["draft"]
    # In production this is where the reply is posted to the ticket system.
    return {"outcome": "sent", "final_reply": text}


def escalate(state: SupportState) -> dict:
    return {"outcome": "escalated", "final_reply": ""}


def decline(state: SupportState) -> dict:
    return {"outcome": "declined",
            "final_reply": "Could not produce a valid draft; routed to a human."}


def abstain(state: SupportState) -> dict:
    return {"outcome": "abstained",
            "final_reply": "Privacy/legal topic: no AI draft, routed to the data-protection team."}


# --- 3. ROUTERS (conditional edges) ----------------------------------------

def route_after_classify(state: SupportState) -> Literal["abstain", "retrieve"]:
    return "abstain" if state.get("category") == "privacy_legal" else "retrieve"


def route_after_validate(state: SupportState) -> Literal["human_review", "draft", "decline"]:
    if not state.get("errors"):
        return "human_review"
    if state.get("attempts", 0) < MAX_DRAFT_ATTEMPTS:
        return "draft"  # loop back: the repair turn
    return "decline"


def route_after_review(state: SupportState) -> Literal["send", "escalate"]:
    return "send" if state.get("review", {}).get("action") in ("approve", "edit") else "escalate"


# --- 4. ASSEMBLE ------------------------------------------------------------

def build_graph(llm: LLM, checkpointer=None):
    g = StateGraph(SupportState)

    g.add_node("classify", make_classify(llm))
    g.add_node("retrieve", retrieve)
    g.add_node("draft", make_draft(llm))
    g.add_node("validate", validate)
    g.add_node("human_review", human_review)
    g.add_node("send", send)
    g.add_node("escalate", escalate)
    g.add_node("decline", decline)
    g.add_node("abstain", abstain)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify)
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "validate")
    g.add_conditional_edges("validate", route_after_validate)
    g.add_conditional_edges("human_review", route_after_review)
    for terminal in ("send", "escalate", "decline", "abstain"):
        g.add_edge(terminal, END)

    # Without a checkpointer the graph still runs, but interrupt() cannot pause it.
    return g.compile(checkpointer=checkpointer)
