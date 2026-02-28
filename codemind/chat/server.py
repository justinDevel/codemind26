"""
CodeMind Chat Server
FastAPI + WebSocket streaming chat interface.
"""

import asyncio
import json
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from ..inference import CodeMindEngine

app = FastAPI(title="CodeMind Chat")

# Load model once at startup
engine: CodeMindEngine = None


@app.on_event("startup")
async def load_model():
    global engine
    import os
    ckpt_dir = "checkpoints"
    checkpoints = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]) if os.path.exists(ckpt_dir) else []
    if checkpoints:
        engine = CodeMindEngine(os.path.join(ckpt_dir, checkpoints[-1]))
    else:
        print("No checkpoint found. Train the model first.")


@app.get("/")
async def root():
    with open("codemind/chat/ui/index.html") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    conversation: List[Dict[str, str]] = []

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get('type') == 'message':
                user_text = msg['content']
                conversation.append({"role": "user", "content": user_text})

                # Stream response
                full_response = ""
                loop = asyncio.get_event_loop()

                def stream_tokens():
                    if engine is None:
                        return ["[Model not loaded. Train first.]"]
                    return engine.generate_stream(
                        conversation,
                        max_new_tokens=msg.get('max_tokens', 512),
                        temperature=msg.get('temperature', 0.7),
                    )

                for token in stream_tokens():
                    full_response += token
                    await websocket.send_text(json.dumps({
                        "type": "token",
                        "content": token
                    }))
                    await asyncio.sleep(0)  # yield to event loop

                conversation.append({"role": "assistant", "content": full_response})
                await websocket.send_text(json.dumps({"type": "done"}))

            elif msg.get('type') == 'reset':
                conversation = []
                await websocket.send_text(json.dumps({"type": "reset_ok"}))

    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    run()
