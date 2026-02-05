import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may use up to 2 tool calls sequentially when needed (e.g., get a course outline first, then search for specific content)
- Prefer a single tool call when possible; use a second only when the first result is insufficient
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Course Outline Tool Usage:
- Use the `get_course_outline` tool for questions about course structure, lesson lists, or outlines (e.g., "What lessons are in...", "Show me the outline of...", "What does the course cover?")
- When presenting outline results, include the course title, course link, and each lesson's number and title

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls per query.
        """

        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare initial API call parameters
        api_params = {
            **self.base_params,
            "system": system_content
        }

        messages = [{"role": "user", "content": query}]

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get initial response
        api_params["messages"] = messages
        response = self.client.messages.create(**api_params)

        # Tool-use loop
        for round in range(self.MAX_TOOL_ROUNDS):
            if response.stop_reason != "tool_use" or not tool_manager:
                break

            # Execute tools and append results to messages
            tool_error = not self._execute_tool_round(response, messages, tool_manager)

            # Build follow-up params; include tools only if more rounds remain and no error
            follow_up_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            if tools and not tool_error and round < self.MAX_TOOL_ROUNDS - 1:
                follow_up_params["tools"] = tools
                follow_up_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**follow_up_params)

            if tool_error:
                break

        # Extract text from final response
        for block in response.content:
            if block.type == "text":
                return block.text

        return "I'm sorry, I wasn't able to generate a response. Please try again."

    def _execute_tool_round(self, response, messages: List, tool_manager) -> bool:
        """
        Execute tool calls from a response and append results to messages.

        Returns True on success, False on error.
        """
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {e}",
                        "is_error": True
                    })
                    if tool_results:
                        messages.append({"role": "user", "content": tool_results})
                    return False

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        return True