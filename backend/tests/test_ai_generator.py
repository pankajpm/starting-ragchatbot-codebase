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

        # Verify second call includes tool result and tools (for possible second round)
        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        kwargs = second_call_kwargs.kwargs if second_call_kwargs.kwargs else second_call_kwargs[1]
        messages = kwargs["messages"]
        # Last message should be the tool result
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tool_123"
        # Tools should be included (round 0 < MAX_TOOL_ROUNDS - 1)
        assert "tools" in kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_round_tool_use(self, mock_anthropic_cls):
        """Two sequential tool calls: tool_use -> tool_use -> text response"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # Round 1: Claude calls get_course_outline
        first_tool = _make_tool_use_response(
            "get_course_outline", {"course_name": "MCP"}, "tool_1"
        )
        # Round 2: Claude calls search_course_content
        second_tool = _make_tool_use_response(
            "search_course_content", {"query": "lesson 3 topics"}, "tool_2"
        )
        # Final: text response
        final = _make_text_response("Lesson 3 covers X, Y, Z.")

        mock_client.messages.create.side_effect = [first_tool, second_tool, final]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = [
            "Lesson 1: Intro\nLesson 2: Basics\nLesson 3: Advanced",
            "Lesson 3 content about X, Y, Z"
        ]

        tools = [
            {"name": "get_course_outline", "description": "test", "input_schema": {"type": "object", "properties": {}}},
            {"name": "search_course_content", "description": "test", "input_schema": {"type": "object", "properties": {}}},
        ]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="What topics does lesson 3 cover?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        assert result == "Lesson 3 covers X, Y, Z."
        assert mock_tool_manager.execute_tool.call_count == 2
        assert mock_client.messages.create.call_count == 3

        # Verify message structure: user, assistant(tool1), tool_result1, assistant(tool2), tool_result2
        final_call_kwargs = mock_client.messages.create.call_args_list[2]
        kwargs = final_call_kwargs.kwargs if final_call_kwargs.kwargs else final_call_kwargs[1]
        messages = kwargs["messages"]
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"  # tool_result 1
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"  # tool_result 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_terminates(self, mock_anthropic_cls):
        """Loop should stop after MAX_TOOL_ROUNDS even if Claude keeps requesting tools"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # All responses request tools (more than MAX_TOOL_ROUNDS)
        tool1 = _make_tool_use_response("search_course_content", {"query": "a"}, "t1")
        tool2 = _make_tool_use_response("search_course_content", {"query": "b"}, "t2")
        # Third response also requests tool, but loop should have ended
        tool3 = _make_tool_use_response("search_course_content", {"query": "c"}, "t3")

        mock_client.messages.create.side_effect = [tool1, tool2, tool3]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "results"

        tools = [{"name": "search_course_content", "description": "test", "input_schema": {"type": "object", "properties": {}}}]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="test", tools=tools, tool_manager=mock_tool_manager
        )

        # Should return fallback since the final response has no text block
        assert result == "I'm sorry, I wasn't able to generate a response. Please try again."
        # Only 2 tool executions (MAX_TOOL_ROUNDS)
        assert mock_tool_manager.execute_tool.call_count == 2
        # 3 API calls: initial + 2 follow-ups
        assert mock_client.messages.create.call_count == 3

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_error_stops_loop(self, mock_anthropic_cls):
        """If execute_tool raises, error is sent as tool_result, one final call without tools, then loop stops"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        tool_response = _make_tool_use_response(
            "search_course_content", {"query": "test"}, "tool_err"
        )
        error_reply = _make_text_response("Sorry, I encountered an error searching.")

        mock_client.messages.create.side_effect = [tool_response, error_reply]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = Exception("Connection timeout")

        tools = [{"name": "search_course_content", "description": "test", "input_schema": {"type": "object", "properties": {}}}]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="test", tools=tools, tool_manager=mock_tool_manager
        )

        # After error, a follow-up call is made without tools so Claude can respond
        assert result == "Sorry, I encountered an error searching."
        assert mock_client.messages.create.call_count == 2

        # The follow-up call should not include tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        kwargs = second_call_kwargs.kwargs if second_call_kwargs.kwargs else second_call_kwargs[1]
        assert "tools" not in kwargs

        # The error tool_result should be in the messages
        messages = kwargs["messages"]
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["is_error"] is True
        assert "Connection timeout" in tool_result_msg["content"][0]["content"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_final_call_omits_tools(self, mock_anthropic_cls):
        """After 2 tool rounds, the 3rd API call should NOT include tools"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        tool1 = _make_tool_use_response("search_course_content", {"query": "a"}, "t1")
        tool2 = _make_tool_use_response("search_course_content", {"query": "b"}, "t2")
        final = _make_text_response("Final answer")

        mock_client.messages.create.side_effect = [tool1, tool2, final]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "results"

        tools = [{"name": "search_course_content", "description": "test", "input_schema": {"type": "object", "properties": {}}}]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        result = generator.generate_response(
            query="test", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Final answer"

        # The third (final) API call should NOT include tools
        third_call_kwargs = mock_client.messages.create.call_args_list[2]
        kwargs = third_call_kwargs.kwargs if third_call_kwargs.kwargs else third_call_kwargs[1]
        assert "tools" not in kwargs

        # The second API call SHOULD include tools (round 0 < MAX_TOOL_ROUNDS - 1)
        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        kwargs2 = second_call_kwargs.kwargs if second_call_kwargs.kwargs else second_call_kwargs[1]
        assert "tools" in kwargs2

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
