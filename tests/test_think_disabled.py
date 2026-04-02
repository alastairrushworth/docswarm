"""Tests that thinking/reasoning is disabled for all Ollama calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from docswarm.agents.swarm import DocSwarm
from docswarm.agents.swarm import _strip_think_tags

# ---------------------------------------------------------------------------
# ChatOllama (used by the swarm LLM)
# ---------------------------------------------------------------------------


class TestChatOllamaThinkDisabled:
    """Verify ChatOllama is constructed with reasoning=False."""

    def test_chat_ollama_reasoning_false(self, tmp_config, db):
        """DocSwarm must create ChatOllama with reasoning=False so the Ollama
        API receives think=false and no <think> blocks are emitted."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        wiki_client = MagicMock()
        wiki_client.list_pages.return_value = []

        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm) as mock_cls:
            DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        # ChatOllama is called twice (researcher + writer), both with reasoning=False
        assert mock_cls.call_count == 2
        for call in mock_cls.call_args_list:
            assert (
                call.kwargs.get("reasoning") is False
            ), f"ChatOllama must be created with reasoning=False, got: {call.kwargs}"

    def test_reasoning_maps_to_think_false_in_api_params(self):
        """Verify that langchain_ollama maps reasoning=False to think=False
        in the parameters sent to the Ollama API."""
        from langchain_core.messages import HumanMessage as HM
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model="test-model", reasoning=False)
        assert llm.reasoning is False

        # _chat_params builds the dict sent to the Ollama client.
        params = llm._chat_params(messages=[HM(content="hi")])
        assert (
            params["think"] is False
        ), f"Expected think=False in API params, got think={params.get('think')}"


# ---------------------------------------------------------------------------
# Raw Ollama API call (pdf_tools classify_page_content)
# ---------------------------------------------------------------------------


class TestClassifyPageThinkDisabled:
    """Verify the raw /api/generate payload includes think: false."""

    def test_classify_payload_has_think_false(self, db, sample_page, tmp_path):
        """classify_page_content must send think=False in the JSON payload."""
        from docswarm.agents.tools.pdf_tools import create_classification_tools

        # Create a dummy image file so the tool doesn't bail out early
        img_path = tmp_path / "page_0001.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        db._exec(
            "UPDATE docswarm.pages SET image_path = ? WHERE id = ?",
            [str(img_path), sample_page["id"]],
        )

        from docswarm.config import Config

        test_config = Config(model="test-model", ollama_base_url="http://localhost:11434")
        tools = create_classification_tools(db, config=test_config)
        classify_fn = tools[0]

        # Mock urlopen to capture the request payload
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps(
            {"response": "CLASSIFICATION: editorial — test page"}
        ).encode()
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_response) as mock_urlopen:
            classify_fn.invoke({"page_id": sample_page["id"]})

        mock_urlopen.assert_called_once()
        request_obj = mock_urlopen.call_args[0][0]
        payload = json.loads(request_obj.data.decode())
        assert (
            payload.get("think") is False
        ), f"Expected think=False in classify payload, got: {payload}"


class TestClassifyOpenAI:
    """Verify _classify_openai calls the OpenAI API correctly."""

    def test_classify_openai_uses_max_completion_tokens(self, db, sample_page, tmp_path):
        """_classify_openai must use max_completion_tokens, not max_tokens."""
        from docswarm.agents.tools.pdf_tools import create_classification_tools
        from docswarm.config import Config

        img_path = tmp_path / "page_0001.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        db._exec(
            "UPDATE docswarm.pages SET image_path = ? WHERE id = ?",
            [str(img_path), sample_page["id"]],
        )

        test_config = Config(
            use_ollama=False,
            openai_api_key="test-key",
            openai_model="gpt-5.4-nano",
        )
        tools = create_classification_tools(db, config=test_config)
        classify_fn = tools[0]

        mock_message = MagicMock()
        mock_message.content = "CLASSIFICATION: editorial — test page"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            result = classify_fn.invoke({"page_id": sample_page["id"]})

        assert "editorial" in result
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["model"] == "gpt-5.4-nano"

    def test_classify_openai_sends_image(self, db, sample_page, tmp_path):
        """_classify_openai must include the page image as base64."""
        from docswarm.agents.tools.pdf_tools import create_classification_tools
        from docswarm.config import Config

        img_path = tmp_path / "page_0001.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        db._exec(
            "UPDATE docswarm.pages SET image_path = ? WHERE id = ?",
            [str(img_path), sample_page["id"]],
        )

        test_config = Config(
            use_ollama=False,
            openai_api_key="test-key",
            openai_model="gpt-5.4-nano",
        )
        tools = create_classification_tools(db, config=test_config)
        classify_fn = tools[0]

        mock_message = MagicMock()
        mock_message.content = "CLASSIFICATION: advertisement — ad page"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            classify_fn.invoke({"page_id": sample_page["id"]})

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        content = messages[0]["content"]
        image_block = [c for c in content if c.get("type") == "image_url"]
        assert len(image_block) == 1
        assert image_block[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Strip think tags (defence in depth)
# ---------------------------------------------------------------------------


class TestStripThinkTags:
    """DocSwarm._strip_think_tags must remove any <think>...</think> blocks
    that slip through despite think=False."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            ("<think>internal reasoning</think>editorial", "editorial"),
            ("<think>\nstep 1\nstep 2\n</think>\n\nThe answer is 42.", "The answer is 42."),
            ("No think tags here.", "No think tags here."),
            ("", ""),
            ("<think>one</think>middle<think>two</think>end", "middleend"),
        ],
    )
    def test_strip_think_tags(self, input_text, expected):
        assert _strip_think_tags(input_text) == expected
