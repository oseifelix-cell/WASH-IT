import asyncio, websockets

async def test():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send("reset")
        for i in range(10):
            msg = await ws.recv()
            kind = "BINARY" if isinstance(msg, bytes) else "TEXT"
            size = len(msg)
            preview = msg[:80] if kind == "TEXT" else f"<{size} bytes>"
            print(f"{i}: {kind}  size={size}  {preview}")

asyncio.run(test())
