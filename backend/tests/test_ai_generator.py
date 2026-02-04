import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator


def _make_text_response(text, stop_reason="end_turn"):
    """Helper to create a mock Anthropic API response with text content"""
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = [content_block]
    return response


def _make_tool_use_response(tool_name, tool_input, tool_id="tool_123"):
    """Helper to create a mock Anthropic API response with tool_use content"""
    content_block = MagicMock()
    content_block.type = "tool_use"
    content_block.name = tool_name
    content_block.input = tool_input
    content_block.id = tool_id
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [content_block]
    return response


class TestAIGeneratorDirectResponse:
    """Tests for AIGenerator when Claude answers directly (no tool use)"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_direct_response_no_tools(self, mock_anthropic_cls):
        """When stop_reason='end_turn', should return content[0].text"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_text_response("Hello!")

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(query="What is Python?")

        assert result == "Hello!"
        mock_client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_conversation_history_in_system_prompt(self, mock_anthropic_cls):
        """System prompt should include conversation history when provided"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_text_response("follow up answer")

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.generate_response(
            query="follow up",
            conversation_history="User: hi\nAssistant: hello"
        )

        call_kwargs = mock_client.messages.create.call_args
        system_content = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert "User: hi" in system_content
        assert "Assistant: hello" in system_content
        assert "Previous conversation:" in system_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_adds_tool_choice_auto(self, mock_anthropic_cls):
        """When tools are provided, tool_choice should be set to auto"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_text_response("answer")

        tools = [{"name": "search_course_content", "description": "test", "input_schema": {"type": "object", "properties": {}}}]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.generate_response(query="test", tools=tools)

        call_kwargs = mock_client.messages.create.call_args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["tools"] == tools


class TestAIGeneratorToolExecution:
    """Tests for AIGenerator tool-use round-trip behavior"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_use_round_trip(self, mock_anthropic_cls):
        """Full tool-use flow: Claude requests tool -> execute -> send results -> get final answer"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # First call: Claude wants to use a tool
        tool_response = _make_tool_use_response(
            "search_course_content",
            {"query": "MCP basics"},
            "tool_123"
        )
        # Second call: Claude synthesizes the answer
        final_response = _make_text_response("Here is what I found about MCP...")

        mock_client.messages.create.side_effect = [tool_response, final_response]

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "[Introduction to MCP - Lesson 1]\nMCP content here"

        tools = [{"name": "search_course_content", "description": "test", "input_schema": {"type": "object", "properties": {}}}]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="Tell me about MCP",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify result
        assert result == "Here is what I found about MCP..."

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP basics"
        )

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

        # Verify second call includes tool result
        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        kwargs = second_call_kwargs.kwargs if second_call_kwargs.kwargs else second_call_kwargs[1]
        messages = kwargs["messages"]
        # Last message should be the tool result
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tool_123"

    @patch("ai_generator.anthropic.Anthropic")
    def test_second_call_omits_tools(self, mock_anthropic_cls):
        """The follow-up API call after tool execution should NOT include tools"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        tool_response = _make_tool_use_response("search_course_content", {"query": "test"})
        final_response = _make_text_response("Final answer")
        mock_client.messages.create.side_effect = [tool_response, final_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search results"

        tools = [{"name": "search_course_content", "description": "test", "input_schema": {"type": "object", "properties": {}}}]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

        # Second call should not have "tools" key
        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        kwargs = second_call_kwargs.kwargs if second_call_kwargs.kwargs else second_call_kwargs[1]
        assert "tools" not in kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_api_error_propagates(self, mock_anthropic_cls):
        """Anthropic API errors should propagate (no try/except in AIGenerator)"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("401 Unauthorized: Invalid API key")

        generator = AIGenerator(api_key="bad-key", model="claude-sonnet-4-20250514")

        with pytest.raises(Exception, match="401 Unauthorized"):
            generator.generate_response(query="test")

    @patch("ai_generator.anthropic.Anthropic")
    def test_empty_api_key_creates_client(self, mock_anthropic_cls):
        """AIGenerator with empty api_key should still construct, but fail on API call"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("Authentication error: API key is empty")

        generator = AIGenerator(api_key="", model="claude-sonnet-4-20250514")

        with pytest.raises(Exception, match="Authentication error"):
            generator.generate_response(query="test")
