import pytest
import os
from unittest.mock import MagicMock, patch, PropertyMock


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() orchestration"""

    def _create_rag_system_with_mocks(self):
        """Create a RAGSystem with all external dependencies mocked"""
        with patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_ai_cls, \
             patch("rag_system.DocumentProcessor"):

            from config import Config
            config = Config()
            config.ANTHROPIC_API_KEY = "test-key"
            config.CHROMA_PATH = "/tmp/test_chroma"

            from rag_system import RAGSystem
            rag = RAGSystem(config)

            # Get references to mocked components
            mock_ai = rag.ai_generator
            mock_ai.generate_response.return_value = "Test response about MCP"

            return rag, mock_ai

    def test_query_returns_tuple(self):
        """query() should return a (str, list) tuple"""
        rag, mock_ai = self._create_rag_system_with_mocks()

        result = rag.query("What is MCP?")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)

    def test_query_wraps_prompt(self):
        """User query should be wrapped in the prompt template"""
        rag, mock_ai = self._create_rag_system_with_mocks()

        rag.query("What is MCP?")

        call_kwargs = mock_ai.generate_response.call_args
        query_arg = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query") or call_kwargs[0][0]
        assert "Answer this question about course materials: What is MCP?" in query_arg

    def test_query_passes_tools_and_manager(self):
        """Tools and tool_manager should be passed to generate_response"""
        rag, mock_ai = self._create_rag_system_with_mocks()

        rag.query("test query")

        call_kwargs = mock_ai.generate_response.call_args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "tools" in kwargs
        assert isinstance(kwargs["tools"], list)
        assert len(kwargs["tools"]) > 0  # Should have at least search tool
        assert "tool_manager" in kwargs
        assert kwargs["tool_manager"] is rag.tool_manager

    def test_query_passes_history(self):
        """Pre-existing session history should be forwarded to generate_response"""
        rag, mock_ai = self._create_rag_system_with_mocks()

        # Create a session and add history
        session_id = rag.session_manager.create_session()
        rag.session_manager.add_exchange(session_id, "previous question", "previous answer")

        rag.query("follow up question", session_id)

        call_kwargs = mock_ai.generate_response.call_args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        history = kwargs.get("conversation_history")
        assert history is not None
        assert "previous question" in history
        assert "previous answer" in history

    def test_query_resets_sources(self):
        """reset_sources() should be called after get_last_sources()"""
        rag, mock_ai = self._create_rag_system_with_mocks()

        # Spy on tool_manager methods
        rag.tool_manager.get_last_sources = MagicMock(return_value=[{"label": "Test", "url": None}])
        rag.tool_manager.reset_sources = MagicMock()

        rag.query("test")

        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()

    def test_query_updates_session(self):
        """Session history should be updated with query and response after call"""
        rag, mock_ai = self._create_rag_system_with_mocks()

        session_id = rag.session_manager.create_session()
        rag.query("test question", session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert "test question" in history
        assert "Test response about MCP" in history

    def test_exception_propagates_to_caller(self):
        """Exceptions in generate_response should propagate (triggers HTTP 500)"""
        rag, mock_ai = self._create_rag_system_with_mocks()
        mock_ai.generate_response.side_effect = Exception("API key invalid")

        with pytest.raises(Exception, match="API key invalid"):
            rag.query("anything")


class TestRAGSystemIntegration:
    """Integration tests using real components to diagnose the 'query failed' issue"""

    def test_config_has_api_key(self):
        """DIAGNOSTIC: Config should have a non-empty API key loaded from .env"""
        from config import config
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is empty - check .env file"
        assert len(config.ANTHROPIC_API_KEY) > 10, "ANTHROPIC_API_KEY looks too short to be valid"

    def test_config_model_valid(self):
        """DIAGNOSTIC: Model name should be a valid Claude model identifier"""
        from config import config
        assert config.ANTHROPIC_MODEL.startswith("claude-"), (
            f"Model '{config.ANTHROPIC_MODEL}' doesn't look like a valid Claude model"
        )

    def test_vector_store_has_data(self):
        """DIAGNOSTIC: ChromaDB should have courses loaded from docs/"""
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        if not os.path.exists(chroma_path):
            pytest.skip("chroma_db directory not found - server may not have been started yet")

        from vector_store import VectorStore
        store = VectorStore(
            chroma_path=chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        count = store.get_course_count()
        titles = store.get_existing_course_titles()

        assert count > 0, "ChromaDB has 0 courses - documents were never loaded"
        assert len(titles) > 0, "No course titles found in ChromaDB"

    def test_vector_store_search_works(self):
        """DIAGNOSTIC: Semantic search should return results for a generic query"""
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        if not os.path.exists(chroma_path):
            pytest.skip("chroma_db directory not found")

        from vector_store import VectorStore
        store = VectorStore(
            chroma_path=chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        # Skip if no data loaded
        if store.get_course_count() == 0:
            pytest.skip("No courses in ChromaDB")

        results = store.search(query="introduction")

        assert results.error is None, f"Search returned error: {results.error}"
        assert not results.is_empty(), "Search for 'introduction' returned no results"
        assert len(results.documents) > 0
        assert len(results.metadata) > 0

    def test_search_tool_end_to_end(self):
        """DIAGNOSTIC: CourseSearchTool should produce formatted output with real data"""
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        if not os.path.exists(chroma_path):
            pytest.skip("chroma_db directory not found")

        from vector_store import VectorStore
        from search_tools import CourseSearchTool

        store = VectorStore(
            chroma_path=chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        if store.get_course_count() == 0:
            pytest.skip("No courses in ChromaDB")

        tool = CourseSearchTool(store)
        result = tool.execute(query="introduction")

        # Should not be an error message
        assert "error" not in result.lower() or "No relevant content" not in result
        # Should contain formatted headers
        assert "[" in result, "Result doesn't contain bracket-formatted headers"
        assert len(result) > 50, "Result seems too short to be real content"
