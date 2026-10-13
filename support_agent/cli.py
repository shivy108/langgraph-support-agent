"""Command-line runner. Checkpoints live in checkpoints.db, so a run that pauses
for human review survives the process exiting.

    python -m support_agent.cli run "My bank says needs reauthorisation" --plan premium
    python -m support_agent.cli resume <thread-id> approve
    python -m support_agent.cli resume <thread-id> edit "Hi! Here is my edited reply..."
    python -m support_agent.cli resume <thread-id> reject
    python -m support_agent.cli history <thread-id>
    python -m support_agent.cli graph
"""

import argparse
import json
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .agent import build_graph
from .llm import NvidiaLLM

DB = "checkpoints.db"


def _summary(node: str, update: dict) -> str:
    if not update:
        return ""
    if node == "classify":
        return f"category={update.get('category')}"
    if node == "retrieve":
        return "docs=" + ", ".join(d["id"] for d in update.get("docs", []))
    if node == "draft":
        return f"attempt {update.get('attempts')} citations={update.get('citations')}"
    if node == "validate":
        return "OK" if not update.get("errors") else f"errors={update['errors']}"
    return f"outcome={update.get('outcome', '')}"


def _stream(graph, payload, config) -> None:
    """Run the graph, printing each node as it finishes."""
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                pause = update[0].value
                print("\n  PAUSED for human review")
                print(f"  draft:     {pause['draft']}")
                print(f"  citations: {pause['citations']}")
                tid = config["configurable"]["thread_id"]
                print(f"\n  continue with:  python -m support_agent.cli resume {tid} approve|edit|reject")
            else:
                print(f"  -> {node:<13} {_summary(node, update)}")

    state = graph.get_state(config)
    if not state.next:  # nothing left to run: finished
        print(f"\n  DONE: {state.values.get('outcome')}")
        if state.values.get("final_reply"):
            print(f"  reply: {state.values['final_reply']}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("ticket"); r.add_argument("--plan", default="premium")
    s = sub.add_parser("resume"); s.add_argument("thread"); s.add_argument("action", choices=["approve", "edit", "reject"]); s.add_argument("text", nargs="?")
    h = sub.add_parser("history"); h.add_argument("thread")
    sub.add_parser("graph")
    args = p.parse_args()

    with SqliteSaver.from_conn_string(DB) as checkpointer:
        llm = NvidiaLLM() if args.cmd in ("run", "resume") else None
        graph = build_graph(llm, checkpointer)

        if args.cmd == "graph":
            print(graph.get_graph().draw_mermaid())
            return

        if args.cmd == "run":
            tid = uuid.uuid4().hex[:8]
            print(f"thread {tid}")
            config = {"configurable": {"thread_id": tid}}
            _stream(graph, {"ticket": args.ticket, "customer": {"plan": args.plan}}, config)

        elif args.cmd == "resume":
            config = {"configurable": {"thread_id": args.thread}}
            decision = {"action": args.action, "text": args.text}
            print(f"thread {args.thread}: resuming with {decision}")
            _stream(graph, Command(resume=decision), config)

        elif args.cmd == "history":
            # Every checkpoint is kept: this is the audit trail (and what enables time travel).
            config = {"configurable": {"thread_id": args.thread}}
            for snap in reversed(list(graph.get_state_history(config))):
                step = snap.metadata.get("step")
                keys = sorted(k for k in snap.values if k not in ("ticket", "customer", "docs"))
                print(f"  step {step:>2}  next={list(snap.next) or ['END']}  state keys={keys}")
            final = graph.get_state(config).values
            print("\n  final state:", json.dumps({k: final.get(k) for k in ("category", "attempts", "outcome")}))


if __name__ == "__main__":
    main()
