"""Live diagnostic tests that make real API calls to identify the actual failure.
These tests require a valid ANTHROPIC_API_KEY and loaded ChromaDB data.
"""
import pytest
import os


class TestLiveDiagnostics:
    """Tests that exercise the real system to find the actual failure point"""

    def test_anthropic_api_key_works(self):
        """DIAGNOSTIC: Verify the API key can actually authenticate with Anthropic"""
        from config import config
        if not config.ANTHROPIC_API_KEY:
            pytest.fail("No API key configured")

        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        try:
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hello in one word"}]
            )
            assert response.content[0].text
        except anthropic.AuthenticationError as e:
            pytest.fail(f"API key is invalid: {e}")
        except anthropic.NotFoundError as e:
            pytest.fail(f"Model '{config.ANTHROPIC_MODEL}' not found: {e}")
        except Exception as e:
            pytest.fail(f"Anthropic API call failed: {type(e).__name__}: {e}")

    def test_anthropic_tool_calling_works(self):
        """DIAGNOSTIC: Verify tool calling works with the actual API"""
        from config import config
        if not config.ANTHROPIC_API_KEY:
            pytest.skip("No API key configured")

        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        tools = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"]
            }
        }]

        try:
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": "Search for information about MCP protocol in the course materials"}],
                tools=tools,
                tool_choice={"type": "auto"}
            )
            # Just verify the call succeeds and returns a valid response
            assert response.stop_reason in ("end_turn", "tool_use")
        except Exception as e:
            pytest.fail(f"Tool calling failed: {type(e).__name__}: {e}")

    def test_full_rag_query_content_question(self):
        """DIAGNOSTIC: Execute a real content query through the full RAG system"""
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        if not os.path.exists(chroma_path):
            pytest.skip("chroma_db not found")

        from config import config
        if not config.ANTHROPIC_API_KEY:
            pytest.skip("No API key configured")

        from rag_system import RAGSystem

        try:
            rag = RAGSystem(config)
        except Exception as e:
            pytest.fail(f"RAGSystem initialization failed: {type(e).__name__}: {e}")

        try:
            response, sources = rag.query("What is this course about?")
        except Exception as e:
            pytest.fail(f"RAGSystem.query() raised: {type(e).__name__}: {e}")

        assert isinstance(response, str), f"Response is not a string: {type(response)}"
        assert len(response) > 0, "Response is empty"

    def test_full_rag_query_with_session(self):
        """DIAGNOSTIC: Execute a content query with session management"""
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        if not os.path.exists(chroma_path):
            pytest.skip("chroma_db not found")

        from config import config
        if not config.ANTHROPIC_API_KEY:
            pytest.skip("No API key configured")

        from rag_system import RAGSystem

        rag = RAGSystem(config)
        session_id = rag.session_manager.create_session()

        try:
            response, sources = rag.query("What courses are available?", session_id)
        except Exception as e:
            pytest.fail(f"RAGSystem.query() with session raised: {type(e).__name__}: {e}")

        assert isinstance(response, str)
        assert isinstance(sources, list)
        # sources should be either empty or a list of dicts
        for s in sources:
            assert isinstance(s, dict), f"Source is not a dict: {type(s)} = {s}"
