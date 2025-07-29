import asyncio
import aiohttp
import json
import os

async def test_mcp_server():
    """Test if the MCP server is running and responding correctly"""
    
    # Get MCP server URL from environment or use default
    mcp_server_url = os.environ.get("DDG_MCP_SERVER_URL", "http://localhost:8080/mcp")
    
    print(f"Testing MCP server at: {mcp_server_url}")
    
    # Test payload
    payload = {
        "method": "search",
        "params": {
            "query": "test search",
            "max_results": 3
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print("Sending test request...")
            
            async with session.post(
                mcp_server_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                print(f"Response status: {response.status}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ MCP Server is working!")
                    print(f"Response: {json.dumps(result, indent=2)}")
                else:
                    error_text = await response.text()
                    print(f"❌ MCP Server returned error: {response.status}")
                    print(f"Error response: {error_text}")
                    
    except aiohttp.ClientTimeout:
        print("❌ Request timed out - MCP server may not be running")
        print("Make sure your MCP server is started on the correct port")
        
    except aiohttp.ClientError as e:
        print(f"❌ Network error: {e}")
        print("Check if the MCP server is running and accessible")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

async def test_simple_connection():
    """Test just basic connectivity to the server"""
    mcp_server_url = os.environ.get("DDG_MCP_SERVER_URL", "http://localhost:8080/mcp")
    base_url = mcp_server_url.replace("/mcp", "")
    
    print(f"Testing basic connectivity to: {base_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                print(f"✅ Server is reachable! Status: {response.status}")
                text = await response.text()
                print(f"Response: {text[:200]}...")
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")

if __name__ == "__main__":
    print("=== Testing MCP Server ===")
    asyncio.run(test_simple_connection())
    print("\n=== Testing MCP Search Functionality ===")
    asyncio.run(test_mcp_server())