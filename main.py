

# from fastapi import FastAPI, HTTPException
# from socketio_instance import sio
# from routes.route import router
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import os
# import socketio
# from pydantic import BaseModel
# from typing import Dict, Any, List
# import aiohttp
# import asyncio
# from bs4 import BeautifulSoup
# import logging

# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Create the main FastAPI app
# fastapi_app = FastAPI()

# # Add CORS middleware
# fastapi_app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # MCP Request Model
# class SearchRequest(BaseModel):
#     method: str
#     params: Dict[str, Any]

# # Simple Search Service (copied from your mcp_http_wrapper.py)
# class SimpleSearchService:
#     def __init__(self):
#         self.session = None
    
#     async def get_session(self):
#         if not self.session:
#             self.session = aiohttp.ClientSession(
#                 timeout=aiohttp.ClientTimeout(total=30),
#                 headers={
#                     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#                 }
#             )
#         return self.session
    
#     async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
#         """Perform direct search using DuckDuckGo HTML interface"""
#         try:
#             session = await self.get_session()
            
#             search_url = "https://html.duckduckgo.com/html/"
#             params = {
#                 'q': query,
#                 'b': '',
#                 'kl': 'us-en',
#             }
            
#             logger.info(f"Searching for: {query}")
            
#             async with session.get(search_url, params=params) as response:
#                 if response.status != 200:
#                     logger.error(f"DuckDuckGo returned status {response.status}")
#                     return [{"error": f"HTTP {response.status}: Failed to fetch search results"}]
                
#                 html_content = await response.text()
#                 soup = BeautifulSoup(html_content, 'html.parser')
                
#                 results = []
#                 result_containers = soup.find_all('div', class_='result__body')
                
#                 logger.info(f"Found {len(result_containers)} result containers")
                
#                 for container in result_containers[:max_results]:
#                     try:
#                         title_element = container.find('a', class_='result__a')
#                         title = title_element.get_text(strip=True) if title_element else "No title"
                        
#                         url = title_element.get('href', '') if title_element else ''
#                         if url.startswith('/l/?uddg='):
#                             url = url.replace('/l/?uddg=', '')
                        
#                         snippet_element = container.find('a', class_='result__snippet')
#                         snippet = snippet_element.get_text(strip=True) if snippet_element else "No description available"
                        
#                         if title and url:
#                             results.append({
#                                 "title": title,
#                                 "url": url,
#                                 "snippet": snippet
#                             })
                            
#                     except Exception as e:
#                         logger.error(f"Error parsing search result: {e}")
#                         continue
                
#                 if not results:
#                     links = soup.find_all('a', class_='result__a')
#                     for i, link in enumerate(links[:max_results]):
#                         if link.get('href'):
#                             results.append({
#                                 "title": link.get_text(strip=True) or f"Result {i+1}",
#                                 "url": link.get('href', ''),
#                                 "snippet": "No description available"
#                             })
                
#                 logger.info(f"Extracted {len(results)} results")
#                 return results if results else [{"error": "No search results found"}]
                
#         except asyncio.TimeoutError:
#             logger.error("Search request timed out")
#             return [{"error": "Search request timed out"}]
#         except Exception as e:
#             logger.error(f"Search error: {e}")
#             return [{"error": f"Search error: {str(e)}"}]
    
#     async def close(self):
#         if self.session:
#             await self.session.close()

# # Global search service instance
# search_service = SimpleSearchService()

# # Basic health check route
# @fastapi_app.get("/")
# def health_check():
#     return {"status": "ok"}

# # MCP endpoint (directly in main app)
# @fastapi_app.post("/mcp")
# async def handle_search_request(request: SearchRequest):
#     """Handle search requests via HTTP"""
#     try:
#         logger.info(f"Received request: {request.method} with params: {request.params}")
        
#         if request.method != "search":
#             raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")
        
#         query = request.params.get("query", "")
#         max_results = request.params.get("max_results", 5)
        
#         if not query:
#             raise HTTPException(status_code=400, detail="Query parameter is required")
        
#         # Perform search
#         results = await search_service.search_duckduckgo(query, max_results)
        
#         # Format results for response
#         if len(results) == 1 and "error" in results[0]:
#             return {"error": results[0]["error"]}
        
#         # Format results as a readable string
#         formatted_results = []
#         formatted_results.append(f"Search Results for '{query}':")
        
#         for i, result in enumerate(results, 1):
#             if "error" in result:
#                 formatted_results.append(f"{i}. Error: {result['error']}")
#             else:
#                 title = result.get('title', 'No title')
#                 url = result.get('url', 'No URL')
#                 snippet = result.get('snippet', 'No description')
                
#                 formatted_results.append(f"\n{i}. {title}")
#                 formatted_results.append(f"   URL: {url}")
#                 formatted_results.append(f"   Description: {snippet}")
        
#         result_text = "\n".join(formatted_results)
        
#         return {"result": result_text}
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error handling search request: {e}")
#         raise HTTPException(status_code=500, detail=f"Search server error: {str(e)}")




# # Add these routes to your main FastAPI app (in the alternative_main_py)

# @fastapi_app.get("/debug/mcp-test")
# async def test_mcp_directly():
#     """Test the MCP endpoint directly"""
#     try:
#         # Test the search service directly
#         results = await search_service.search_duckduckgo("test math query", 3)
#         return {
#             "status": "success", 
#             "results": results,
#             "message": "MCP search service is working"
#         }
#     except Exception as e:
#         import traceback
#         return {
#             "status": "error", 
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }

# @fastapi_app.post("/debug/mcp-full-test")
# async def test_mcp_full_flow():
#     """Test the full MCP flow like the tool would"""
#     try:
#         request = SearchRequest(
#             method="search",
#             params={"query": "algebra basics", "max_results": 3}
#         )
#         result = await handle_search_request(request)
#         return {"status": "success", "result": result}
#     except Exception as e:
#         import traceback
#         return {
#             "status": "error", 
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }

# @fastapi_app.get("/debug/routes")
# def debug_routes():
#     """Debug endpoint to see all available routes"""
#     routes = []
#     for route in fastapi_app.routes:
#         routes.append({
#             "path": route.path,
#             "methods": getattr(route, 'methods', []),
#             "name": getattr(route, 'name', 'unknown')
#         })
#     return {"routes": routes}
# # Include your API routes under /api
# fastapi_app.include_router(router, prefix="/api")

# # Cleanup on shutdown
# @fastapi_app.on_event("shutdown")
# async def shutdown_event():
#     await search_service.close()

# # Mount Socket.IO with FastAPI app
# app = socketio.ASGIApp(
#     socketio_server=sio,
#     other_asgi_app=fastapi_app,
#     socketio_path="/socket.io"
# )

from fastapi import FastAPI, HTTPException
from socketio_instance import sio
from routes.route import router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import socketio
from pydantic import BaseModel
from typing import Dict, Any, List
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the main FastAPI app
fastapi_app = FastAPI()

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add debugging middleware
@fastapi_app.middleware("http")
async def debug_middleware(request, call_next):
    if request.url.path.startswith("/api/ask-ai"):
        print(f"Request to {request.url.path}")
        
    response = await call_next(request)
    
    if request.url.path.startswith("/api/ask-ai"):
        print(f"Response status: {response.status_code}")
        
    return response

# MCP Request Model
class SearchRequest(BaseModel):
    method: str
    params: Dict[str, Any]

# Simple Search Service (copied from your mcp_http_wrapper.py)
class SimpleSearchService:
    def __init__(self):
        self.session = None
    
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
        return self.session
    
    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform direct search using DuckDuckGo HTML interface"""
        try:
            session = await self.get_session()
            
            search_url = "https://html.duckduckgo.com/html/"
            params = {
                'q': query,
                'b': '',
                'kl': 'us-en',
            }
            
            logger.info(f"Searching for: {query}")
            
            async with session.get(search_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"DuckDuckGo returned status {response.status}")
                    return [{"error": f"HTTP {response.status}: Failed to fetch search results"}]
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                results = []
                result_containers = soup.find_all('div', class_='result__body')
                
                logger.info(f"Found {len(result_containers)} result containers")
                
                for container in result_containers[:max_results]:
                    try:
                        title_element = container.find('a', class_='result__a')
                        title = title_element.get_text(strip=True) if title_element else "No title"
                        
                        url = title_element.get('href', '') if title_element else ''
                        if url.startswith('/l/?uddg='):
                            url = url.replace('/l/?uddg=', '')
                        
                        snippet_element = container.find('a', class_='result__snippet')
                        snippet = snippet_element.get_text(strip=True) if snippet_element else "No description available"
                        
                        if title and url:
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet
                            })
                            
                    except Exception as e:
                        logger.error(f"Error parsing search result: {e}")
                        continue
                
                if not results:
                    links = soup.find_all('a', class_='result__a')
                    for i, link in enumerate(links[:max_results]):
                        if link.get('href'):
                            results.append({
                                "title": link.get_text(strip=True) or f"Result {i+1}",
                                "url": link.get('href', ''),
                                "snippet": "No description available"
                            })
                
                logger.info(f"Extracted {len(results)} results")
                return results if results else [{"error": "No search results found"}]
                
        except asyncio.TimeoutError:
            logger.error("Search request timed out")
            return [{"error": "Search request timed out"}]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [{"error": f"Search error: {str(e)}"}]
    
    async def close(self):
        if self.session:
            await self.session.close()

# Global search service instance
search_service = SimpleSearchService()

# Basic health check route
@fastapi_app.get("/")
def health_check():
    return {"status": "ok"}

# MCP endpoint (directly in main app)
@fastapi_app.post("/mcp")
async def handle_search_request(request: SearchRequest):
    """Handle search requests via HTTP"""
    try:
        logger.info(f"Received request: {request.method} with params: {request.params}")
        
        if request.method != "search":
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")
        
        query = request.params.get("query", "")
        max_results = request.params.get("max_results", 5)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        # Perform search
        results = await search_service.search_duckduckgo(query, max_results)
        
        # Format results for response
        if len(results) == 1 and "error" in results[0]:
            return {"error": results[0]["error"]}
        
        # Format results as a readable string
        formatted_results = []
        formatted_results.append(f"Search Results for '{query}':")
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                formatted_results.append(f"{i}. Error: {result['error']}")
            else:
                title = result.get('title', 'No title')
                url = result.get('url', 'No URL')
                snippet = result.get('snippet', 'No description')
                
                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   URL: {url}")
                formatted_results.append(f"   Description: {snippet}")
        
        result_text = "\n".join(formatted_results)
        
        return {"result": result_text}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling search request: {e}")
        raise HTTPException(status_code=500, detail=f"Search server error: {str(e)}")

# Debug routes
@fastapi_app.get("/debug/test-web-search-tool")
async def test_web_search_tool():
    """Test the web search tool directly"""
    try:
        from web_search import DuckDuckGoSearchTool
        
        # Create and test the tool
        tool = DuckDuckGoSearchTool()
        result = await tool._arun("basic algebra concepts")
        
        return {
            "status": "success", 
            "tool_result": result,
            "message": "Web search tool is working"
        }
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@fastapi_app.get("/debug/mcp-test")
async def test_mcp_directly():
    """Test the MCP endpoint directly"""
    try:
        # Test the search service directly
        results = await search_service.search_duckduckgo("test math query", 3)
        return {
            "status": "success", 
            "results": results,
            "message": "MCP search service is working"
        }
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@fastapi_app.post("/debug/mcp-full-test")
async def test_mcp_full_flow():
    """Test the full MCP flow like the tool would"""
    try:
        request = SearchRequest(
            method="search",
            params={"query": "algebra basics", "max_results": 3}
        )
        result = await handle_search_request(request)
        return {"status": "success", "result": result}
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@fastapi_app.get("/debug/routes")
def debug_routes():
    """Debug endpoint to see all available routes"""
    routes = []
    for route in fastapi_app.routes:
        routes.append({
            "path": route.path,
            "methods": getattr(route, 'methods', []),
            "name": getattr(route, 'name', 'unknown')
        })
    return {"routes": routes}

# Add these routes to your main FastAPI app

@fastapi_app.get("/debug/env-check")
async def check_environment():
    """Check environment variables and configuration"""
    return {
        "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "openai_api_key_length": len(os.environ.get("OPENAI_API_KEY", "")) if os.environ.get("OPENAI_API_KEY") else 0,
        "mcp_server_url": os.environ.get("MCP_SERVER_URL", "not_set"),
        "available_routes": [route.path for route in fastapi_app.routes],
        "current_host": "self",
        "environment_vars": {k: v for k, v in os.environ.items() if "API" in k or "MCP" in k or "KEY" in k}
    }

@fastapi_app.get("/debug/test-internal-mcp")
async def test_internal_mcp():
    """Test the internal MCP endpoint"""
    try:
        # Test the internal search directly
        request = SearchRequest(
            method="search",
            params={"query": "test query", "max_results": 3}
        )
        result = await handle_search_request(request)
        return {
            "status": "success",
            "internal_mcp_working": True,
            "result": result
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "internal_mcp_working": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@fastapi_app.get("/debug/test-web-search-connection")
async def test_web_search_connection():
    """Test web search tool connection without going through LangGraph"""
    try:
        import aiohttp
        import asyncio
        
        # Test connection to our own MCP endpoint
        mcp_url = "http://localhost:8000/mcp"  # Assuming we're running on port 8000
        
        payload = {
            "method": "search",
            "params": {
                "query": "simple test",
                "max_results": 2
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                mcp_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                
                return {
                    "status": "success",
                    "connection_working": True,
                    "response_status": response.status,
                    "result": result
                }
                
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "connection_working": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@fastapi_app.get("/debug/simple-ai-test")
async def simple_ai_test():
    """Test AI processing without web search"""
    try:
        from routes.route import process_ai_query
        
        # Simple math question that shouldn't need web search
        simple_query = "What is 2 + 2?"
        result = await process_ai_query(simple_query, "debug_session", 5)
        
        return {
            "status": "success",
            "ai_working": True,
            "query": simple_query,
            "result": result
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "ai_working": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Include your API routes under /api
fastapi_app.include_router(router, prefix="/api")

# Cleanup on shutdown
@fastapi_app.on_event("shutdown")
async def shutdown_event():
    await search_service.close()

# Mount Socket.IO with FastAPI app
app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=fastapi_app,
    socketio_path="/socket.io"
)