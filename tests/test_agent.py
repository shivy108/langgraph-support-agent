"""Every path through the graph, tested with a scripted fake model.

No network, no API key: the fake returns canned JSON in order. This is the
payoff of keeping the model behind a one-method interface.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from support_agent.agent import build_graph


class FakeLLM:
    def __init__(self, *responses: dict):
        self.responses = list(responses)
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        return self.responses.pop(0)


GOOD_DRAFT = {
    "reply": "Your bank consent has expired. Open Accounts, tap the bank and renew the consent; your history is kept.",
    "citations": ["KB-101"],
}
TICKET = {"ticket": "My bank connection says needs reauthorisation", "customer": {"plan": "premium"}}


def _run(llm, payload=TICKET, thread="t1"):
    graph = build_graph(llm, InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    graph.invoke(payload, config)
    return graph, config


def test_privacy_ticket_abstains_without_drafting():
    llm = FakeLLM({"category": "privacy_legal"})
    graph, config = _run(llm, {"ticket": "Delete all my data under GDPR", "customer": {}})
    state = graph.get_state(config).values
    assert state["outcome"] == "abstained"
    assert "draft" not in state
    assert llm.calls == 1  # never reached the drafting model


def test_happy_path_pauses_then_sends_on_approve():
    graph, config = _run(FakeLLM({"category": "bank_connection"}, GOOD_DRAFT))
    paused = graph.get_state(config)
    assert paused.next == ("human_review",)  # waiting for a human
    assert paused.tasks[0].interrupts[0].value["citations"] == ["KB-101"]

    graph.invoke(Command(resume={"action": "approve"}), config)
    final = graph.get_state(config).values
    assert final["outcome"] == "sent"
    assert final["final_reply"] == GOOD_DRAFT["reply"]


def test_human_edit_replaces_the_draft():
    graph, config = _run(FakeLLM({"category": "bank_connection"}, GOOD_DRAFT))
    graph.invoke(Command(resume={"action": "edit", "text": "Edited by a human."}), config)
    assert graph.get_state(config).values["final_reply"] == "Edited by a human."


def test_reject_escalates():
    graph, config = _run(FakeLLM({"category": "bank_connection"}, GOOD_DRAFT))
    graph.invoke(Command(resume={"action": "reject"}), config)
    assert graph.get_state(config).values["outcome"] == "escalated"


def test_invalid_draft_gets_one_repair_turn():
    bad = {"reply": "We guarantee this will be fixed within the hour, no problem at all.", "citations": ["KB-999"]}
    graph, config = _run(FakeLLM({"category": "bank_connection"}, bad, GOOD_DRAFT))
    state = graph.get_state(config)
    assert state.next == ("human_review",)  # repaired draft passed validation
    assert state.values["attempts"] == 2


def test_still_invalid_after_repair_declines():
    bad = {"reply": "We guarantee it.", "citations": []}
    graph, config = _run(FakeLLM({"category": "bank_connection"}, bad, bad))
    state = graph.get_state(config).values
    assert state["outcome"] == "declined"
    assert state["attempts"] == 2


def test_checkpoints_form_an_audit_trail():
    graph, config = _run(FakeLLM({"category": "bank_connection"}, GOOD_DRAFT))
    graph.invoke(Command(resume={"action": "approve"}), config)
    steps = list(graph.get_state_history(config))
    assert len(steps) >= 6  # one snapshot per step: START, classify, retrieve, draft, validate, review, send
