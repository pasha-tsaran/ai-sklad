"""Read-only rehearsal against configured knowledge/model; never connects to Talk or its database."""

import argparse
import json

from app.core.config import Settings
from app.services.chat.agent_dialogue import run_agent
from app.services.llm.openwebui import OpenWebUILLM
from app.services.rag.talk import TalkKnowledge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn", action="append", required=True, help="User message; repeat for a dialogue")
    args = parser.parse_args()
    settings = Settings()
    if settings.llm_backend != "openwebui" or not settings.openwebui_knowledge_ids:
        parser.error("Configure LLM_BACKEND=openwebui and OPENWEBUI_KNOWLEDGE_IDS first")
    model, knowledge = OpenWebUILLM(settings), TalkKnowledge(settings)
    flow, messages = {}, []
    for message in args.turn:
        if not message.strip() or len(message) > 6000:
            parser.error("Each turn must contain 1–6000 characters")
        history = messages[-settings.memory_max_messages:]
        while history and len(json.dumps(history, ensure_ascii=False)) > settings.memory_max_chars:
            history = history[1:]
        result = run_agent(message, flow, {"short_term": {"messages": history}}, model, knowledge, settings)
        flow = result.flow if result.flow is not None else flow
        blocks = [{"text": text, "image_description": picture.caption if picture else None,
                   "source": source} for (text, picture), source in zip(
                       result.blocks, result.block_sources or [None] * len(result.blocks), strict=True)]
        print(json.dumps({"user": message, "route": result.action, "blocks": blocks,
                          "memory": flow.get("agent", {}).get("memory"), "delivered": False}, ensure_ascii=False))
        messages.append({"role": "user", "content": message})
        messages.extend({"role": "assistant", "content": text} for text, _ in result.blocks if text)


if __name__ == "__main__":
    main()
