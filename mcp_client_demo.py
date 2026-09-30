"""
Minimal MCP client for testing the FinDoc server without Claude Desktop.

It starts mcp_server.py as a subprocess over stdio, lists the tools,
calls each one once, and times the calls.

Usage:
    python mcp_client_demo.py
    python mcp_client_demo.py "What was Apple's net income in 2024?"
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).with_name("mcp_server.py")


def show(title: str, result, elapsed_ms: float, max_chars: int = 1200) -> None:
    text = result.content[0].text if result.content else ""
    try:
        text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        pass
    print(f"\n=== {title}  ({elapsed_ms:.0f} ms) ===")
    print(text[:max_chars] + ("\n..." if len(text) > max_chars else ""))


async def call(session, name: str, args: dict):
    t0 = time.perf_counter()
    result = await session.call_tool(name, args)
    return result, (time.perf_counter() - t0) * 1000


async def main(question: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            res, ms = await call(session, "list_sections", {})
            show("list_sections", res, ms)

            # The first hybrid call loads the embedding model, so it is slower.
            res, ms = await call(session, "search_filing", {"query": question, "top_k": 3})
            show(f"search_filing (hybrid): {question}", res, ms)

            res, ms = await call(session, "search_filing", {"query": question, "top_k": 3})
            show("search_filing again (model already loaded)", res, ms, max_chars=300)

            res, ms = await call(session, "search_filing", {
                "query": question, "top_k": 3, "tables_only": True,
                "sections": ["Financial Statements and Supplementary Data"]})
            show("search_filing with section + table filters", res, ms)

            first_id = json.loads(res.content[0].text)["results"][0]["chunk_id"]
            res, ms = await call(session, "get_chunk", {"chunk_id": first_id})
            show(f"get_chunk {first_id}", res, ms, max_chars=600)

            page = int(first_id.split("_")[0][1:])
            res, ms = await call(session, "get_page", {"page_num": page})
            show(f"get_page {page}", res, ms, max_chars=600)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What was Apple's net income in 2024?"
    asyncio.run(main(q))
