"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive
    uv run python -m app.ask --interactive "LoRAとは何ですか？"

ツールを持たない「素の」呼び出し。--interactive では同じ会話履歴を使って質問を続けられる。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors, types
from google.genai.chats import Chat

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def create_chat() -> tuple[genai.Client, Chat]:
    client = genai.Client(api_key=get_api_key())
    return client, client.chats.create(model=MODEL)


def ask(question: str, chat: Chat | None = None) -> str:
    if chat is None:
        client, chat = create_chat()
    started_at = time.perf_counter()
    response = chat.send_message(
        question,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    elapsed_seconds = time.perf_counter() - started_at

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(f"[response time] {elapsed_seconds:.2f} sec")
    print(f"[tokens] input: {input_tokens}, output: {output_tokens}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="会話履歴を保ったまま続けて質問する",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.interactive:
            client, chat = create_chat()
            question = args.question
            while True:
                if question is None:
                    try:
                        question = input("質問を入力してください（終了するには空行）: ").strip()
                    except EOFError:
                        break
                if not question:
                    break
                print(f"[質問] {question}\n")
                print(ask(question, chat))
                question = None
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
