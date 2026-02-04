import sys
import os
import pytest
from unittest.mock import MagicMock

# Add backend directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


@pytest.fixture
def sample_search_results():
    """SearchResults with 2 documents and realistic metadata"""
    return SearchResults(
        documents=[
            "MCP stands for Model Context Protocol. It allows AI models to interact with external tools.",
            "The MCP architecture uses a client-server pattern for tool communication."
        ],
        metadata=[
            {"course_title": "Introduction to MCP", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Introduction to MCP", "lesson_number": 2, "chunk_index": 3}
        ],
        distances=[0.25, 0.42]
    )


@pytest.fixture
def empty_search_results():
    """SearchResults with no documents and no error"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """SearchResults with an error message"""
    return SearchResults.empty("No course found matching 'NonExistent'")


@pytest.fixture
def mock_vector_store(sample_search_results):
    """MagicMock spec'd to VectorStore with default search behavior"""
    store = MagicMock(spec=VectorStore)
    store.search.return_value = sample_search_results
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store._resolve_course_name.return_value = "Introduction to MCP"
    store.get_existing_course_titles.return_value = ["Introduction to MCP"]
    store.get_course_count.return_value = 1
    return store


@pytest.fixture
def sample_course():
    """A realistic Course object"""
    return Course(
        title="Introduction to MCP",
        course_link="https://example.com/course/mcp",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson/0"),
            Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson/1"),
            Lesson(lesson_number=2, title="Architecture", lesson_link="https://example.com/lesson/2"),
        ]
    )


@pytest.fixture
def sample_chunks():
    """Realistic CourseChunk objects"""
    return [
        CourseChunk(content="MCP stands for Model Context Protocol.", course_title="Introduction to MCP", lesson_number=1, chunk_index=0),
        CourseChunk(content="The architecture uses client-server pattern.", course_title="Introduction to MCP", lesson_number=2, chunk_index=1),
        CourseChunk(content="Tools are defined with JSON schemas.", course_title="Introduction to MCP", lesson_number=2, chunk_index=2),
    ]
