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
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                }
            )
        return self.session
    
    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform direct search using DuckDuckGo HTML interface with better status handling"""
        
        for attempt in range(3):  # Try up to 3 times
            try:
                session = await self.get_session()
                
                # Use the simple HTML interface
                search_url = "https://html.duckduckgo.com/html/"
                params = {
                    'q': query,
                    'b': '',
                    'kl': 'us-en',
                    's': '0',  # Start from result 0
                }
                
                logger.info(f"Attempt {attempt + 1}: Searching for: {query}")
                
                async with session.get(search_url, params=params) as response:
                    logger.info(f"DuckDuckGo returned status: {response.status}")
                    
                    # Accept 200, 202, and other 2xx status codes
                    if not (200 <= response.status < 300):
                        logger.warning(f"DuckDuckGo returned status {response.status}, retrying...")
                        if attempt < 2:  # Not the last attempt
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            return [{"error": f"HTTP {response.status}: Failed to fetch search results after 3 attempts"}]
                    
                    html_content = await response.text()
                    
                    # Check if we got actual content
                    if not html_content or len(html_content) < 100:
                        logger.warning(f"Received minimal content on attempt {attempt + 1}")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            return [{"error": "Received empty or minimal response from search service"}]
                    
                    # Parse HTML with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    results = []
                    
                    # Try multiple selectors to find results
                    result_selectors = [
                        'div.result__body',  # Primary selector
                        'div.result',        # Alternative selector
                        'div[class*="result"]',  # Any div with "result" in class
                        '.web-result',       # Alternative selector
                        '.result-item',      # Another common selector
                    ]
                    
                    result_containers = []
                    for selector in result_selectors:
                        result_containers = soup.select(selector)
                        if result_containers:
                            logger.info(f"Found {len(result_containers)} results using selector: {selector}")
                            break
                    
                    if not result_containers:
                        logger.warning("No result containers found, trying to extract any links")
                        # Fallback: look for any links that might be search results
                        links = soup.find_all('a', href=True)
                        valid_links = []
                        
                        for link in links:
                            href = link.get('href', '')
                            text = link.get_text(strip=True)
                            
                            # Filter out navigation and internal links
                            if (href.startswith('http') and 
                                text and 
                                len(text) > 10 and 
                                not any(skip in href.lower() for skip in ['duckduckgo.com', 'javascript:', 'mailto:'])):
                                valid_links.append(link)
                        
                        for i, link in enumerate(valid_links[:max_results]):
                            results.append({
                                "title": link.get_text(strip=True)[:150] or f"Result {i+1}",
                                "url": link.get('href', ''),
                                "snippet": "No description available"
                            })
                        
                        if results:
                            logger.info(f"Fallback extraction found {len(results)} results")
                            return results
                        else:
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return [{"error": "No search results found"}]
                    
                    # Parse result containers
                    for container in result_containers[:max_results]:
                        try:
                            # Try multiple ways to extract title and URL
                            title_element = (
                                container.find('a', class_='result__a') or
                                container.find('a', class_=lambda x: x and 'result' in x) or
                                container.find('h2') or
                                container.find('h3') or
                                container.find('a', href=True)
                            )
                            
                            if not title_element:
                                continue
                                
                            title = title_element.get_text(strip=True)[:150] or "No title"
                            
                            # Extract URL
                            url = title_element.get('href', '')
                            
                            # Clean up DuckDuckGo redirect URLs
                            if url.startswith('/l/?uddg='):
                                url = url.replace('/l/?uddg=', '')
                            elif url.startswith('/l/?kh=-1&uddg='):
                                url = url.replace('/l/?kh=-1&uddg=', '')
                            
                            # Extract snippet
                            snippet_element = (
                                container.find('a', class_='result__snippet') or
                                container.find('span', class_='result__snippet') or
                                container.find('div', class_='snippet') or
                                container.find('p')
                            )
                            
                            snippet = snippet_element.get_text(strip=True)[:300] if snippet_element else "No description available"
                            
                            if title and url and url.startswith('http'):
                                results.append({
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet
                                })
                                
                        except Exception as e:
                            logger.error(f"Error parsing search result: {e}")
                            continue
                    
                    if results:
                        logger.info(f"Successfully extracted {len(results)} results")
                        return results
                    else:
                        logger.warning(f"No results extracted on attempt {attempt + 1}")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            # Try a different approach - simple text extraction
                            logger.info("Trying simple text extraction as final fallback")
                            text_content = soup.get_text()
                            if "No results found" in text_content or len(text_content) < 200:
                                return [{"error": "No search results found by DuckDuckGo"}]
                            else:
                                # Return a generic message with some content indication
                                return [{
                                    "title": f"Search completed for: {query}",
                                    "url": "https://duckduckgo.com",
                                    "snippet": f"Found content related to '{query}' but could not parse specific results. The search service returned a {response.status} response with content."
                                }]
                            
            except asyncio.TimeoutError:
                logger.error(f"Search request timed out on attempt {attempt + 1}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return [{"error": "Search request timed out after 3 attempts"}]
                    
            except Exception as e:
                logger.error(f"Search error on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return [{"error": f"Search error: {str(e)}"}]
        
        return [{"error": "All search attempts failed"}]
    
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
    """Handle search requests via HTTP with improved error handling"""
    try:
        logger.info(f"Received request: {request.method} with params: {request.params}")
        
        if request.method != "search":
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")
        
        query = request.params.get("query", "").strip()
        max_results = min(request.params.get("max_results", 5), 10)  # Cap at 10 results
        
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        logger.info(f"Performing search for: '{query}' (max_results: {max_results})")
        
        # Perform search
        results = await search_service.search_duckduckgo(query, max_results)
        
        # Check if we got an error
        if len(results) == 1 and "error" in results[0]:
            error_msg = results[0]["error"]
            logger.warning(f"Search returned error: {error_msg}")
            
            # Try to provide a helpful fallback
            fallback_response = f"I apologize, but I encountered an issue searching for information about '{query}'. "
            fallback_response += f"The search service returned: {error_msg}. "
            fallback_response += "Please let me know if you'd like me to try a different approach or if you can provide more context about your question."
            
            return {"result": fallback_response}
        
        # Format results for response
        if not results:
            return {"result": f"No search results found for '{query}'. Please try rephrasing your question or provide more specific terms."}
        
        # Format results as a readable string
        formatted_results = []
        formatted_results.append(f"Search Results for '{query}':")
        formatted_results.append("=" * 50)
        
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
                formatted_results.append("")  # Empty line for readability
        
        result_text = "\n".join(formatted_results)
        
        logger.info(f"Successfully formatted {len(results)} search results")
        return {"result": result_text}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling search request: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a user-friendly error message
        error_response = f"I apologize, but I encountered a technical issue while searching for information about '{request.params.get('query', 'your query')}'. "
        error_response += "Please try again in a moment, or let me know if you can provide your question in a different way."
        
        return {"result": error_response}

# Add test endpoints
@app.get("/debug/test-search")
async def test_search():
    """Test the search service directly"""
    try:
        test_query = "mathematics algebra basics"
        results = await search_service.search_duckduckgo(test_query, 3)
        
        return {
            "status": "success",
            "query": test_query,
            "results_count": len(results),
            "results": results,
            "has_errors": any("error" in result for result in results),
            "message": "Search service test completed"
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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