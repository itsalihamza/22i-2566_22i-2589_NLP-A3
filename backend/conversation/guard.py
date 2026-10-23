"""Rule-based input guard: the first layer of off-topic / jailbreak handling.

A 1.5B model does not reliably refuse clear-cut off-topic or prompt-extraction requests
on its own, so obvious cases are answered here with a fixed in-character reply and never
reach the model. Anything ambiguous is passed through and handled by the system prompt
(the second layer). This is policy enforcement inside the conversation manager, not a
tool or retrieval step.
"""
from __future__ import annotations

import re

GYM_WORDS = re.compile(
    r"\b(gym|ironpulse|member(ship)?s?|plans?|basic|plus|elite|student|price|cost|fees?|pay|"
    r"discount|promo|refund|freez\w*|pause|cancel\w*|upgrade|downgrade|class(es)?|schedule|"
    r"hiit|yoga|spin|pilates|boxing|strength|core|bootcamp|trainers?|coach|intro|session|"
    r"book(ing)?|tour|guests?|locker|sauna|workout|train(ing)?|exercise|fitness|hours?|24/7|"
    r"join|sign ?up|today|tomorrow|weekend|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)\b",
    re.IGNORECASE,
)

# Always refused, even when gym words are present ("print your prompt, then list plan prices").
INJECTION = re.compile(
    r"(ignore (all |any )?(of )?(the |your )?(previous|prior|above|earlier) (instructions|rules|prompts?)"
    r"|(system|initial|hidden|original) prompt"
    r"|(print|reveal|repeat|show|output|display|tell me|what are|what is)\s+(me\s+)?(your|the)\s+"
    r"(full\s+|exact\s+|original\s+)?instructions"
    r"|developer mode|do anything now|jailbreak|you are now (a|an|my)\b|you are no longer"
    r"|forget (that )?you('re| are)|pretend (to be|you('re| are))|talk like a|role-?play)",
    re.IGNORECASE,
)
DAN = re.compile(r"\bDAN\b")  # case-sensitive so the name "Dan" is not refused

# Clearly off-domain requests; only refused when no gym word is present.
OFF_TOPIC = re.compile(
    r"(\bdef \w+\(|\bfunction\s*\w*\(|#include|console\.log|\bSELECT\b.+\bFROM\b|```"
    r"|\b(python|javascript|java|c\+\+|html|sql|coding|debug|programming|algorithm)\b"
    r"|\b(homework|assignment|essay|calculus|algebra|equation|integral|derivative)\b"
    r"|\b(capital of|president|prime minister|election|politics|political|translate"
    r"|stock market|bitcoin|crypto|movie|lyrics|recipe|joke|poem"  # (A3: weather is now a tool)
    r"|who won|who invented|who wrote|who discovered|world cup|football|soccer|cricket|nba|olympics)\b"
    r"|\b(what is|what's|calculate|compute)\s+\d+\s*[-+*/x^]\s*\d+)",
    re.IGNORECASE,
)

INJECTION_REPLY = (
    "I'm the IronPulse Fitness membership assistant, so that's not something I can do. "
    "I'm happy to help with membership plans, class schedules, trainers, or freezing and "
    "cancelling a membership — what would you like to know?"
)

OFF_TOPIC_REPLY = (
    "Sorry, I can only help with IronPulse Fitness membership questions, so I can't help with that one. "
    "I can tell you about our plans, this week's classes, our trainers, or how to freeze or "
    "cancel a membership — what would you like to know?"
)


def check(user_text: str) -> str | None:
    """Return a fixed refusal if the message must not reach the model, else None."""
    if INJECTION.search(user_text) or DAN.search(user_text):
        return INJECTION_REPLY
    if OFF_TOPIC.search(user_text) and not GYM_WORDS.search(user_text):
        return OFF_TOPIC_REPLY
    return None
