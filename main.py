

# from fastapi import FastAPI
# from socketio_instance import sio
# from routes.route import router
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import os
# import subprocess
# import socketio

# load_dotenv()

# # Create the main FastAPI app
# fastapi_app = FastAPI()

# # Add CORS middleware
# fastapi_app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Or set to your frontend origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Basic health check route (optional but useful for Render)
# @fastapi_app.get("/")
# def health_check():
#     return {"status": "ok"}

# # Include your API routes under /api
# fastapi_app.include_router(router, prefix="/api")

# # Launch mcp_http_wrapper.py on startup
# @fastapi_app.on_event("startup")
# async def run_wrapper():
#     subprocess.Popen(["python", "mcp_http_wrapper.py"])

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




# Add these routes to your main FastAPI app (in the alternative_main_py)

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