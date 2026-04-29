from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from board import Board, generate_layout
import numpy as np
import random
import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=5000, stdout_to_server=True, stderr_to_server=True)
app = FastAPI()
board_instance: Optional[Board] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <body>
            <h1>Board UI</h1>
            <p>前端稍后写</p>
        </body>
    </html>
    """


# ====== WebSocket ======

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()

            type = data.get("type")

            if type == "init":
                layout = [generate_layout() for _ in range(4)]
                await ws.send_json({"type": "init",
                                    "layout": layout})
            if type == "begin":
                current_player = 0
                board_instance = Board(data.get("layout"))
                action_mask = board_instance.get_action_mask(0)
                await ws.send_json({"type": "begin",
                                    "mask": action_mask.tolist()})
            if type == 'step':
                action = data.get("action")
                result, path = board_instance.update(current_player, action)
                nxt_action_mask = board_instance.get_action_mask((current_player + 1) % 4)
                await ws.send_json({"type": "step",
                                    "action": int(action),
                                    "player": current_player,
                                    "result": result,
                                    "path": [int(x) for x in path],
                                    'mask': nxt_action_mask.tolist()})
                current_player = (current_player + 1) % 4
            if type == 'ask':
                action_mask = board_instance.get_action_mask(current_player)
                if action_mask is None:
                    continue
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    board_instance.update(current_player, 5934)
                    continue
                action = random.choice(valid_actions)
                result, path = board_instance.update(current_player, action)
                nxt_action_mask = board_instance.get_action_mask((current_player + 1) % 4)
                await ws.send_json({"type": "step",
                                    "action":int(action),
                                    "player":current_player,
                                    "result": result,
                                    "path": [int(x) for x in path],
                                    'mask':nxt_action_mask.tolist()})
                current_player = (current_player + 1) % 4

    except WebSocketDisconnect:
        print("WebSocket disconnected")
