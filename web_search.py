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
        # Check if we're running on Render or another cloud platform
        base_url = os.environ.get("BASE_URL")
        
        if not base_url:
            # Try to detect the current environment
            if os.environ.get("RENDER"):
                # Running on Render
                service_name = os.environ.get("RENDER_SERVICE_NAME", "math-bot-back")
                base_url = f"https://{service_name}.onrender.com"
            else:
                # Local development
                base_url = "http://localhost:8000"
        
        return f"{base_url}/mcp"

    async def _arun(self, query: str) -> str:
        """Run web search asynchronously using DuckDuckGo MCP endpoint."""
        mcp_server_url = self._get_mcp_url()
        print(f"Performing web search for: {query}")
        print(f"Using MCP URL: {mcp_server_url}")

        # Increased timeout for better reliability
        timeout = aiohttp.ClientTimeout(total=60, connect=30)

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

                    print(f"Attempt {attempt + 1}: Sending request to MCP endpoint: {mcp_server_url}")
                    print(f"Payload: {json.dumps(payload, indent=2)}")

                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "DuckDuckGoSearchTool/1.0"
                    }

                    async with session.post(
                        mcp_server_url,
                        json=payload,
                        headers=headers
                    ) as response:

                        print(f"Response status: {response.status}")
                        response_text = await response.text()
                        print(f"Response body: {response_text[:500]}...")

                        if response.status != 200:
                            print(f"MCP endpoint error: Status {response.status}, Body: {response_text}")
                            if attempt == 2:  # Last attempt
                                return (
                                    f"Error: Search service returned status {response.status}. "
                                    f"Response: {response_text[:200]}..."
                                )
                            await asyncio.sleep(2 ** attempt)
                            continue

                        try:
                            result = await response.json()
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {str(e)}")
                            print(f"Raw response: {response_text}")
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
                    return "Error: Search request timed out after 3 attempts. The search service may be busy."
                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientConnectorError as e:
                print(f"Connection error on attempt {attempt + 1}: {str(e)}")
                if attempt == 2:
                    return f"Error: Could not connect to search service at {mcp_server_url}. Please check if the server is running."
                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                print(f"Client error on attempt {attempt + 1}: {str(e)}")
                if attempt == 2:
                    return f"Error: Network issue while connecting to search service: {str(e)}"
                await asyncio.sleep(2 ** attempt)

            except asyncio.CancelledError:
                print("Search request was cancelled.")
                return "Error: Search request was cancelled. Please try again."

            except Exception as e:
                print(f"Unexpected error in web search attempt {attempt + 1}: {str(e)}")
                import traceback
                error_traceback = traceback.format_exc()
                print(f"Full traceback: {error_traceback}")
                if attempt == 2:
                    return f"Error executing web search: {str(e)}"
                await asyncio.sleep(2 ** attempt)

        return "Error: All search attempts failed."

    def _run(self, query: str) -> str:
        """Synchronous version - runs the async version."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to handle this differently
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(query))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(query))
        except RuntimeError:
            return asyncio.run(self._arun(query))