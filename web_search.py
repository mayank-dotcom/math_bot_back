import os
import aiohttp
import json
from typing import Optional, Dict, Any, ClassVar, List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to be executed")

class DuckDuckGoSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Useful for searching the web for current information or specific questions that cannot be answered from the syllabus context. Use this tool when you need to find information that is not available in the provided mathematical context."
    args_schema: Type[BaseModel] = WebSearchInput
    mcp_server_url: str = "http://127.0.0.1:8765/mcp"  # Default value
    
    def __init__(self, mcp_server_url: str = "http://127.0.0.1:8765/mcp", **kwargs):
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url
    
    async def _arun(self, query: str) -> str:
        """Run web search asynchronously using DuckDuckGo MCP server."""
        try:
            print(f"Performing web search for: {query}")
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "method": "search",  
                    "params": {
                        "query": query,
                        "max_results": 5
                    }
                }
                
                print(f"Sending request to MCP server: {self.mcp_server_url}")
                
                async with session.post(
                    self.mcp_server_url, 
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"MCP server error: Status {response.status}, Body: {error_text}")
                        return f"Error: Received status code {response.status} from MCP server. Response: {error_text}"
                    
                    result = await response.json()
                    print(f"MCP server response: {result}")
                    
                    if "error" in result:
                        return f"Error from search service: {result['error']}"
                    
                    search_results = result.get("result", "No results found")
                    
                    # If result is a string, return it directly
                    if isinstance(search_results, str):
                        return search_results
                    
                    # If result is a list or dict, format it nicely
                    if isinstance(search_results, (list, dict)):
                        return json.dumps(search_results, indent=2)
                    
                    return str(search_results)
                    
        except aiohttp.ClientTimeout:
            return "Error: Search request timed out. The MCP server may be unavailable."
        except aiohttp.ClientError as e:
            return f"Error: Network error while connecting to MCP server: {str(e)}"
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON response from MCP server: {str(e)}"
        except Exception as e:
            print(f"Unexpected error in web search: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error executing web search: {str(e)}"
    
    def _run(self, query: str) -> str:
        """Synchronous version - not supported for this async tool."""
        import asyncio
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete
                return "Error: This tool requires async execution context."
            else:
                return loop.run_until_complete(self._arun(query))
        except RuntimeError:
            # No event loop, create a new one
            return asyncio.run(self._arun(query))