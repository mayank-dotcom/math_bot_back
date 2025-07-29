import os
import aiohttp
import json
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import asyncio

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to be executed")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Useful for searching the web for current information or specific questions "
        "that cannot be answered from the syllabus context. "
        "Use this tool when you need to find information that is not available in the provided mathematical context."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _get_mcp_url(self) -> str:
        """Get the MCP server URL"""
        return "http://math-bot-back.onrender.com/mcp"

    async def _arun(self, query: str) -> str:
        """Run web search asynchronously using DuckDuckGo MCP endpoint."""
        mcp_server_url = self._get_mcp_url()
        print(f"Performing web search for: {query}")
        print(f"Using MCP URL: {mcp_server_url}")

        timeout = aiohttp.ClientTimeout(total=120, connect=120)

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    payload = {
                        "method": "search",
                        "params": {
                            "query": query,
                            "max_results": 5
                        }
                    }

                    print(f"Sending request to MCP endpoint: {mcp_server_url}")
                    print(f"Payload: {json.dumps(payload, indent=2)}")

                    async with session.post(
                        mcp_server_url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                    ) as response:

                        print(f"Response status: {response.status}")

                        if response.status != 200:
                            error_text = await response.text()
                            print(f"MCP endpoint error: Status {response.status}, Body: {error_text}")
                            return (
                                f"Error: Received status code {response.status} from search endpoint. "
                                f"This might be a temporary issue with the search service."
                            )

                        try:
                            result = await response.json()
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {str(e)}")
                            return f"Error: Invalid JSON response from search service: {str(e)}"

                        print(f"MCP endpoint response: {result}")

                        if "error" in result:
                            return f"Search service error: {result['error']}"

                        search_results = result.get("result", "No results found")

                        if isinstance(search_results, str):
                            return search_results

                        if isinstance(search_results, (list, dict)):
                            return json.dumps(search_results, indent=2)

                        return str(search_results)

            except asyncio.TimeoutError:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt == 2:
                    return "Error: Search request timed out. The search service may be busy. Please try again."
                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientConnectorError as e:
                print(f"Connection error: {str(e)}")
                return f"Error: Could not connect to search service. Please check if the server is running."

            except aiohttp.ClientError as e:
                print(f"Client error: {str(e)}")
                return f"Error: Network issue while connecting to search service: {str(e)}"

            except asyncio.CancelledError:
                print("Search request was cancelled.")
                return "Error: Search request was cancelled. Please try again."

            except Exception as e:
                print(f"Unexpected error in web search: {str(e)}")
                import traceback
                error_traceback = traceback.format_exc()
                print(f"Full traceback: {error_traceback}")
                return f"Error executing web search: {str(e)}. Full error: {error_traceback[:500]}..."

    def _run(self, query: str) -> str:
        """Synchronous version - not supported for this async tool."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return "Error: This tool requires async execution context."
            else:
                return loop.run_until_complete(self._arun(query))
        except RuntimeError:
            return asyncio.run(self._arun(query))
# mw fd ,d