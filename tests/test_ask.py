from types import SimpleNamespace

from google.genai import types

import app.ask as ask_module


class FakeChat:
    def __init__(self):
        self.questions = []
        self.configs = []

    def send_message(self, question: str, config: types.GenerateContentConfig):
        self.questions.append(question)
        self.configs.append(config)
        return SimpleNamespace(
            text=f"answer to {question}",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=2,
            ),
        )


def test_one_shot_mode_keeps_default_question(monkeypatch, capsys):
    chat = FakeChat()
    monkeypatch.setattr(ask_module, "create_chat", lambda: (object(), chat))
    monkeypatch.setattr("sys.argv", ["app.ask"])

    ask_module.main()

    assert chat.questions == [ask_module.DEFAULT_QUESTION]
    assert ask_module.DEFAULT_QUESTION in capsys.readouterr().out


def test_ask_sends_question_and_reports_usage(monkeypatch, capsys):
    chat = FakeChat()
    monkeypatch.setattr(ask_module, "create_chat", lambda: (object(), chat))
    times = iter([10.0, 11.25])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(times))

    answer = ask_module.ask("first question")

    assert answer == "answer to first question"
    assert chat.questions == ["first question"]
    assert chat.configs[0].temperature == 0.7
    output = capsys.readouterr().out
    assert "[response time] 1.25 sec" in output
    assert "[tokens] input: 1, output: 2" in output


def test_create_chat_returns_client_to_keep_it_alive(monkeypatch):
    chat = FakeChat()

    class FakeClient:
        class chats:
            @staticmethod
            def create(model):
                return chat

    client = FakeClient()
    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.create_chat() == (client, chat)


def test_interactive_mode_reuses_chat_for_follow_up_questions(monkeypatch, capsys):
    chat = FakeChat()
    questions = iter(["first question", "follow-up question", ""])
    prompts = []
    monkeypatch.setattr(ask_module, "create_chat", lambda: (object(), chat))

    def fake_input(prompt):
        prompts.append(prompt)
        return next(questions)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("sys.argv", ["app.ask", "--interactive"])

    ask_module.main()

    assert chat.questions == ["first question", "follow-up question"]
    assert len(prompts) == 3
    output = capsys.readouterr().out
    assert "answer to first question" in output
    assert "answer to follow-up question" in output
