import pytest
from unittest.mock import MagicMock, patch
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() output behavior"""

    def test_execute_returns_formatted_results(self, mock_vector_store, sample_search_results):
        """execute() should return formatted string with [Course - Lesson N] headers"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="what is MCP")

        # Should contain course title in brackets
        assert "[Introduction to MCP" in result
        # Should contain lesson numbers
        assert "Lesson 1" in result
        assert "Lesson 2" in result
        # Should contain the actual document content
        assert "MCP stands for Model Context Protocol" in result
        assert "client-server pattern" in result
        # Should have populated last_sources
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["label"] == "Introduction to MCP - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson/1"
        # Should have called vector store with correct args
        mock_vector_store.search.assert_called_once_with(
            query="what is MCP", course_name=None, lesson_number=None
        )

    def test_execute_with_filters(self, mock_vector_store):
        """course_name and lesson_number should be passed through to store.search()"""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="embeddings", course_name="MCP", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="embeddings", course_name="MCP", lesson_number=2
        )

    def test_execute_returns_error_on_search_error(self, mock_vector_store, error_search_results):
        """When SearchResults has .error set, execute() should return that error string"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything", course_name="NonExistent")

        assert result == "No course found matching 'NonExistent'"

    def test_execute_returns_no_content_on_empty(self, mock_vector_store, empty_search_results):
        """Empty results (no error) should return 'No relevant content found'"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="obscure topic")

        assert "No relevant content found" in result

    def test_execute_no_content_includes_filter_info(self, mock_vector_store, empty_search_results):
        """Empty results message should include filter context when filters are used"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="obscure", course_name="MCP", lesson_number=3)

        assert "course 'MCP'" in result
        assert "lesson 3" in result

    def test_execute_propagates_vector_store_exception(self, mock_vector_store):
        """If store.search() raises an exception, it should propagate uncaught.
        This is a key failure mode -- CourseSearchTool.execute() has no try/except."""
        mock_vector_store.search.side_effect = Exception("ChromaDB connection failed")
        tool = CourseSearchTool(mock_vector_store)

        with pytest.raises(Exception, match="ChromaDB connection failed"):
            tool.execute(query="anything")

    def test_format_results_handles_missing_metadata(self, mock_vector_store):
        """Missing metadata keys should not crash formatting"""
        results = SearchResults(
            documents=["Some content here"],
            metadata=[{}],  # No course_title or lesson_number
            distances=[0.3]
        )
        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results)

        # Should use 'unknown' for missing course_title (line 94 uses .get default)
        assert "[unknown]" in formatted
        assert "Some content here" in formatted


class TestToolManager:
    """Tests for ToolManager registration, dispatch, and source tracking"""

    def test_register_and_execute_tool(self, mock_vector_store):
        """ToolManager should dispatch execute_tool() to the correct registered tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert isinstance(result, str)
        assert len(result) > 0
        mock_vector_store.search.assert_called_once()

    def test_execute_unknown_tool(self):
        """Executing an unregistered tool name should return an error string"""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool")

        assert result == "Tool 'nonexistent_tool' not found"

    def test_tool_definitions_valid_format(self, mock_vector_store):
        """Tool definitions should have required Anthropic API fields"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store))
        manager.register_tool(CourseOutlineTool(mock_vector_store))

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        for defn in definitions:
            assert "name" in defn
            assert "description" in defn
            assert "input_schema" in defn
            schema = defn["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_source_tracking_and_reset(self, mock_vector_store):
        """Sources should be available after search and cleared after reset"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) > 0

        manager.reset_sources()
        assert manager.get_last_sources() == []


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool.execute()"""

    def test_execute_returns_formatted_outline(self, mock_vector_store):
        """Should return formatted course outline with title, link, instructor, lessons"""
        import json
        mock_vector_store._resolve_course_name.return_value = "Introduction to MCP"
        mock_vector_store.course_catalog = MagicMock()
        mock_vector_store.course_catalog.get.return_value = {
            "metadatas": [{
                "title": "Introduction to MCP",
                "course_link": "https://example.com/mcp",
                "instructor": "Test Instructor",
                "lessons_json": json.dumps([
                    {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "https://example.com/0"},
                    {"lesson_number": 1, "lesson_title": "Getting Started", "lesson_link": "https://example.com/1"},
                ])
            }]
        }

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="MCP")

        assert "Course: Introduction to MCP" in result
        assert "https://example.com/mcp" in result
        assert "Test Instructor" in result
        assert "Lesson 0: Introduction" in result
        assert "Lesson 1: Getting Started" in result
        assert "2 total" in result

    def test_execute_returns_not_found(self, mock_vector_store):
        """Should return not-found message when course doesn't exist"""
        mock_vector_store._resolve_course_name.return_value = None
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Nonexistent")

        assert "No course found matching 'Nonexistent'" in result
