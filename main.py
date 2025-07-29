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

# Simple Search Service (improved version)
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
        for attempt in range(3):
            try:
                session = await self.get_session()
                search_url = "https://html.duckduckgo.com/html/"
                params = {'q': query, 'b': '', 'kl': 'us-en', 's': '0'}
                logger.info(f"Attempt {attempt + 1}: Searching for: {query}")

                async with session.get(search_url, params=params) as response:
                    logger.info(f"DuckDuckGo returned status: {response.status}")
                    if not (200 <= response.status < 300):
                        logger.warning(f"DuckDuckGo returned status {response.status}, retrying...")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return [{"error": f"HTTP {response.status}: Failed to fetch search results after 3 attempts"}]

                    html_content = await response.text()
                    if not html_content or len(html_content) < 100:
                        logger.warning(f"Received minimal content on attempt {attempt + 1}")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return [{"error": "Received empty or minimal response from search service"}]

                    soup = BeautifulSoup(html_content, 'html.parser')
                    results = []
                    result_containers = soup.find_all('div', class_='result__body')
                    for container in result_containers[:max_results]:
                        try:
                            title_element = container.find('a', class_='result__a')
                            title = title_element.get_text(strip=True) if title_element else "No title"
                            url = title_element.get('href', '') if title_element else ''
                            if url.startswith('/l/?uddg='):
                                url = url.replace('/l/?uddg=', '')
                            snippet_element = container.find('a', class_='result__snippet')
                            snippet = snippet_element.get_text(strip=True) if snippet_element else "No description available"
                            if title and url and url.startswith('http'):
                                results.append({"title": title, "url": url, "snippet": snippet})
                        except Exception as e:
                            logger.error(f"Error parsing search result: {e}")
                            continue
                    if results:
                        logger.info(f"Successfully extracted {len(results)} results")
                        return results
                    else:
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return [{
                            "title": f"Search completed for: {query}",
                            "url": "https://duckduckgo.com",
                            "snippet": f"Search was performed but specific results could not be parsed. Status: {response.status}"
                        }]
            except asyncio.TimeoutError:
                logger.error(f"Search request timed out on attempt {attempt + 1}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return [{"error": "Search request timed out"}]
            except Exception as e:
                logger.error(f"Search error on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return [{"error": f"Search error: {str(e)}"}]
        return [{"error": "All search attempts failed"}]

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
    try:
        logger.info(f"Received request: {request.method} with params: {request.params}")
        if request.method != "search":
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")
        query = request.params.get("query", "")
        max_results = request.params.get("max_results", 5)
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        results = await search_service.search_duckduckgo(query, max_results)
        if len(results) == 1 and "error" in results[0]:
            return {"error": results[0]["error"]}
        formatted_results = [f"Search Results for '{query}':"]
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
