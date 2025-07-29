from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import aiohttp
import asyncio
from typing import Dict, Any, List
import json
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple Search HTTP Server")

class SearchRequest(BaseModel):
    method: str
    params: Dict[str, Any]

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
                'b': '',  # No ads
                'kl': 'us-en',  # Language
            }
            
            logger.info(f"Searching for: {query}")
            
            async with session.get(search_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"DuckDuckGo returned status {response.status}")
                    return [{"error": f"HTTP {response.status}: Failed to fetch search results"}]
                
                html_content = await response.text()
                
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                results = []
                result_containers = soup.find_all('div', class_='result__body')
                
                logger.info(f"Found {len(result_containers)} result containers")
                
                for container in result_containers[:max_results]:
                    try:
                        # Extract title
                        title_element = container.find('a', class_='result__a')
                        title = title_element.get_text(strip=True) if title_element else "No title"
                        
                        # Extract URL
                        url = title_element.get('href', '') if title_element else ''
                        if url.startswith('/l/?uddg='):
                            # Decode DuckDuckGo redirect URL
                            url = url.replace('/l/?uddg=', '')
                        
                        # Extract snippet
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
                    # Try alternative parsing if no results found
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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Simple Search HTTP Server started successfully")
    yield
    # Shutdown
    await search_service.close()
    logger.info("Search service closed")

# Update the FastAPI app initialization
app = FastAPI(title="Simple Search HTTP Server", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Simple Search HTTP Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "search_service_running": True}

@app.post("/mcp")
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

if __name__ == "__main__":
    import os
    import sys

    # Get dynamic port or fallback to internal one
    port = int(os.environ.get("MCP_PORT", "8765"))  # Not 8080!
    host = os.environ.get("MCP_HOST", "127.0.0.1")

    print(f"Starting MCP server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
