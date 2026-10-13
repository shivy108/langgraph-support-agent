"""A tiny in-memory knowledge base and keyword retriever.

In Finch the support agent retrieves from Confluence and a pgvector memory of
past replies. Here a list of articles and a word-overlap score stand in for
that, so the example runs with no database. The point is the graph, not the
retriever: swap `retrieve` for a vector search and nothing else changes.
"""

import re

ARTICLES = [
    {
        "id": "KB-101",
        "title": "Bank connection shows 'needs reauthorisation'",
        "body": (
            "Open banking consents expire, usually after 90 or 180 days depending on "
            "the bank. When a connection shows 'needs reauthorisation', the user opens "
            "Accounts, taps the bank and follows the consent flow again. Their history "
            "is kept; only new transactions pause until consent is renewed."
        ),
    },
    {
        "id": "KB-102",
        "title": "Transactions missing after connecting a bank",
        "body": (
            "After a first connection, the initial sync can take up to 15 minutes. Some "
            "banks only share 90 days of history. A manual refresh is available from "
            "the account screen once every 6 hours."
        ),
    },
    {
        "id": "KB-201",
        "title": "Changing or cancelling a subscription",
        "body": (
            "Subscriptions are managed through the App Store or Google Play, not inside "
            "the app. Cancelling stops renewal at the end of the current period; access "
            "continues until then. Refunds are handled by Apple or Google."
        ),
    },
    {
        "id": "KB-202",
        "title": "Premium plan limits",
        "body": (
            "Premium includes up to 10 connected banks and a monthly AI allowance. When "
            "the allowance runs out, AI features pause until the next billing period; "
            "all other features keep working."
        ),
    },
    {
        "id": "KB-301",
        "title": "Exporting your data",
        "body": (
            "Users can export transactions as CSV from Settings, Data export. The export "
            "covers all connected accounts and is generated on the device."
        ),
    },
]

_WORD = re.compile(r"[a-zA-Z]{3,}")
_STOP = {"the", "and", "for", "are", "but", "not", "you", "your", "with", "this",
         "that", "have", "has", "can", "was", "from", "after", "when", "only"}


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)} - _STOP


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Return the k articles sharing the most words with the query."""
    q = _words(query)
    scored = []
    for art in ARTICLES:
        overlap = len(q & _words(art["title"] + " " + art["body"]))
        if overlap:
            scored.append((overlap, art))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [art for _, art in scored[:k]]
