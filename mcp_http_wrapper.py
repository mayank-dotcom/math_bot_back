from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import aiohttp
import asyncio
from typing import Dict, Any, List
import json
from bs4 import BeautifulSoup
import logging
from contextlib import asynccontextmanager

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

    async def close(self):
        if self.session:
            await self.session.close()

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
                        logger.warning(f"Status {response.status}, retrying...")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return [{"error": f"HTTP {response.status}: Failed after 3 attempts"}]

                    html_content = await response.text()
                    if not html_content or len(html_content) < 100:
                        logger.warning("Received minimal content")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return [{"error": "Received minimal response"}]

                    soup = BeautifulSoup(html_content, 'html.parser')
                    results = []
                    containers = soup.select('div.result__body') or soup.select('div.result')

                    for container in containers[:max_results]:
                        try:
                            link = container.find('a', class_='result__a')
                            if not link:
                                continue
                            title = link.get_text(strip=True)
                            url = link.get('href')
                            snippet = container.get_text(strip=True).replace(title, '').strip()
                            results.append({"title": title, "url": url, "snippet": snippet})
                        except Exception as e:
                            logger.error(f"Error parsing result: {e}")
                            continue

                    if results:
                        return results

            except asyncio.TimeoutError:
                logger.error("Timeout occurred")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return [{"error": "Timeout after 3 attempts"}]

            except Exception as e:
                logger.error(f"Search error: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return [{"error": str(e)}]

        return [{"error": "All attempts failed"}]

search_service = SimpleSearchService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting search server")
    yield
    await search_service.close()
    logger.info("Shutting down search server")

app = FastAPI(title="Simple Search HTTP Server", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Simple Search HTTP Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "search_service_running": True}

@app.post("/mcp")
async def handle_search_request(request: SearchRequest):
    try:
        if request.method != "search":
            raise HTTPException(status_code=400, detail="Unsupported method")

        query = request.params.get("query", "").strip()
        max_results = min(request.params.get("max_results", 5), 10)

        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")

        results = await search_service.search_duckduckgo(query, max_results)

        if len(results) == 1 and "error" in results[0]:
            return {"result": f"Error: {results[0]['error']}"}

        formatted_results = [f"Search Results for '{query}':", "=" * 50]
        for i, result in enumerate(results, 1):
            formatted_results.append(f"\n{i}. {result.get('title', 'No title')}")
            formatted_results.append(f"   URL: {result.get('url', 'No URL')}")
            formatted_results.append(f"   Description: {result.get('snippet', 'No description')}")
        return {"result": "\n".join(formatted_results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"result": "An internal error occurred. Please try again."}

@app.get("/debug/test-search")
async def test_search():
    try:
        query = "mathematics algebra basics"
        results = await search_service.search_duckduckgo(query, 3)
        return {
            "status": "success",
            "query": query,
            "results": results,
            "message": "Test completed"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("MCP_PORT", "8765"))
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="info")
