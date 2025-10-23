# 这是一个临时脚本，用于向数据库插入超长文本以进行压力测试

# 导入我们已有的数据库函数和模块
from database import get_db_connection
import datetime

# --- 【请修改这里】 ---
# 1. 请在您的应用前端，创建一个新聊天
# 2. 把那个新聊天的 ID (数字) 填在下面
# (如果您不知道 ID，就填 1，它会插入到第一个会话中)
TARGET_CONVERSATION_ID = 1 
# ----------------------


# 这是一个超长的 Markdown 文本（约 700 行），用于模拟大数据
LONG_MARKDOWN = """
# A Very Long Markdown Document for Stress Testing

This document simulates a large piece of text that an AI might return, such as a full code file, a long explanation, or a detailed report. The goal is to test if parsing this document blocks the main browser thread.

If the **Web Worker** is implemented correctly, the UI should remain perfectly smooth and responsive while this text is being parsed in the background. The user should see a "Parsing..." message, and then the full content will appear, all without freezing the page.

---

## Sample Code Block

Here is a large sample of Python code:

```python
import re
from dlframe.logger import Logger
from dlframe.manager import Manager, ManagerConfig
import asyncio
import websockets
import json
import queue
import threading
import traceback
import base64
import requests
import time
import datetime
from database import get_db_connection

class WebLogger(Logger):
    def __init__(self, socket, name='untitled') -> None:
        super().__init__()
        self.socket = socket
        self.name = name

    def print(self, *params, end='\\n') -> Logger:
        self.socket.send(json.dumps({
            'status': 200, 
            'type': 'print', 
            'data': {
                'content': ' '.join([str(_) for _ in params]) + end
            }
        }))
        return super().print(*params, end=end)
    
    def image(self, image) -> Logger:
        with open(image,'rb') as f:
            base64code = base64.b64encode(f.read())
        base64code=base64code.decode("utf-8")
        imgdata=base64.b64decode(base64code)
        self.socket.send(json.dumps({
            'status': 200, 
            'type': 'image', 
            'data': {
                'content': '{}'.format(base64code)
            }
        }))
        return super().image(image)
    
    def plot(self, plot) -> Logger:
        self.socket.send(json.dumps({
            'status': 200, 
            'type': 'plot', 
            'data': {
                'content': '[{}]: '.format(plot)
            }
        }))
        return super().plot(plot)

class SendSocket:
    def __init__(self, socket) -> None:
        self.sendBuffer = queue.Queue()
        self.socket = socket
        self.sendThread = threading.Thread(target=self.threadWorker, daemon=True)
        self.sendThread.start()
    def threadWorker(self):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(self.sendWorker())
    async def sendWorker(self):
        while True:
            content = self.sendBuffer.get()
            await self.socket.send(content)
    def send(self, content: str):
        self.sendBuffer.put(content)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super(MyEncoder, self).default(obj)
        except TypeError:
            module_name = getattr(type(obj), '__module__', '')
            if module_name.startswith('numpy'):
                if hasattr(obj, 'item'):
                    try:
                        return obj.item()
                    except Exception:
                        pass
                if hasattr(obj, 'tolist'):
                    try:
                        return obj.tolist()
                    except Exception:
                        pass
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            if hasattr(obj, 'tolist'):
                try:
                    return obj.tolist()
                except Exception:
                    pass
            if hasattr(obj, 'item'):
                try:
                    return obj.item()
                except Exception:
                    pass
            return str(obj)

class WebManager(Manager):
    def __init__(self) -> None:
        super().__init__()
        print("WebManager initialized.")

    def _get_timestamp(self):
        return datetime.datetime.now().isoformat()

    def _db_execute(self, query, params=()):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            last_id = cursor.lastrowid
            return last_id
        except sqlite3.Error as e:
            print(f"数据库错误: {e}")
            print(f"查询: {query}")
            print(f"参数: {params}")
        finally:
            if conn:
                conn.close()
    
    def _db_query_all(self, query, params=()):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"数据库错误: {e}")
        finally:
            if conn:
                conn.close()

    def handle_send_message(self, conversation_id: int, prompt: str, logger: WebLogger):
        logger.print(f"[系统] 正在保存用户消息到数据库 (会话ID: {conversation_id})...")
        user_message_sql = "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)"
        self._db_execute(user_message_sql, (conversation_id, 'user', prompt, self._get_timestamp()))

        logger.print(f"[系统] -----------------------------------------------------")
        
        # ... (rest of the file) ..."""