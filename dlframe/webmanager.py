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
import datetime # <-- 【新】用于处理时间戳

# --- 【新】从我们刚创建的 database.py 中导入函数 ---
from database import get_db_connection

class WebLogger(Logger):
    def __init__(self, socket, name='untitled') -> None:
        super().__init__()
        self.socket = socket
        self.name = name

    def print(self, *params, end='\n') -> Logger:
        self.socket.send(json.dumps({
            'status': 200, 
            'type': 'print', 
            'data': {
                'content': ' '.join([str(_) for _ in params]) + end
            }
        }))
        return super().print(*params, end=end)
    
    # ... (image 和 plot 函数未修改) ...
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

    def progess(self, percentage: float) -> Logger:
        return super().progess(percentage)

class SendSocket:
    # ... (此类未修改) ...
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
    # ... (此类未修改) ...
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

    # --- 【新】数据库辅助函数 ---

    def _get_timestamp(self):
        """获取当前的 ISO 格式时间戳"""
        return datetime.datetime.now().isoformat()

    def _db_execute(self, query, params=()):
        """执行数据库插入/更新操作"""
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
        """执行数据库查询操作（返回多行）"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            # 将 sqlite3.Row 对象转换为普通字典
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"数据库错误: {e}")
        finally:
            if conn:
                conn.close()

    # --- 【新】处理Ollama多轮对话的核心函数 ---
    
    def handle_send_message(self, conversation_id: int, prompt: str, logger: WebLogger):
        """
        处理一个新消息：存用户消息 -> 加载历史 -> 调Ollama -> 存AI消息
        """
        
        # 1. 将用户的当前消息存入数据库
        logger.print(f"[系统] 正在保存用户消息到数据库 (会话ID: {conversation_id})...")
        user_message_sql = "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)"
        self._db_execute(user_message_sql, (conversation_id, 'user', prompt, self._get_timestamp()))

        # 2. 从数据库加载**所有**历史消息
        logger.print(f"[系统] 正在从数据库加载会话历史...")
        history_sql = "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
        history_rows = self._db_query_all(history_sql, (conversation_id,))
        
        # 将历史记录转换为 Ollama /api/chat 需要的格式
        # history_rows 已经是 [{ 'role': 'user', 'content': '...' }, ...]
        
        ollama_url = 'http://localhost:11434/api/chat' # <-- 【重要】切换到 /api/chat
        payload = {
            "model": "qwen:0.5b",  # 【重要】请改成您的模型名
            "messages": history_rows, # <-- 【重要】发送完整的历史记录
            "stream": True
        }
        
        logger.print(f"[系统] 正在连接 Ollama (/api/chat) 并发送 {len(history_rows)} 条历史消息...")
        
        full_response = "" # 用来累积 AI 的完整回复
        try:
            response = requests.post(ollama_url, json=payload, stream=True)
            response.raise_for_status()
            
            logger.print("[系统] 已连接。正在等待模型响应...", end='\n\n')

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    
                    # /api/chat 的返回格式是 chunk['message']['content']
                    if chunk.get("message") and chunk['message'].get('content'):
                        text_chunk = chunk['message']['content']
                        full_response += text_chunk
                        logger.print(text_chunk, end='') # 流式发回前端
                    
                    if chunk.get("done", False):
                        logger.print("\n\n[系统] 模型响应结束。")
                        break

        except Exception as e:
            logger.print(f"\n[系统错误] 调用 Ollama 时发生错误: {e}", end='\n')
            return # 出错了，不保存

        # 3. 将 AI 的**完整**回复存入数据库
        if full_response:
            logger.print("[系统] 正在保存 AI 回复到数据库...")
            ai_message_sql = "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)"
            self._db_execute(ai_message_sql, (conversation_id, 'assistant', full_response, self._get_timestamp()))
            logger.print("[系统] 保存完毕。")


    def start(self, host='0.0.0.0', port=8765) -> None:
        async def onRecv(socket, path=None):
            if path is None and hasattr(socket, 'path'):
                path = socket.path
            msgIdx = -1
            sendSocket = SendSocket(socket)
            async for message in socket:
                msgIdx += 1
                message = json.loads(message)

                if not all([key in message.keys() for key in ['type', 'params']]):
                    response = json.dumps({'status': 500, 'data': 'no key param'})
                    await socket.send(response)
                
                else:
                    params = message.get('params', {})
                    logger = WebLogger(sendSocket, 'MainHandler') # 创建一个通用 logger

                    try:
                        # --- 【新】处理前端加载时，请求所有历史会话 ---
                        if message['type'] == 'load_conversations':
                            conv_sql = "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
                            conversations = self._db_query_all(conv_sql)
                            response = json.dumps({
                                'status': 200,
                                'type': 'all_conversations',
                                'data': conversations
                            })
                            await socket.send(response)

                        # --- 【新】处理前端点击某个会话，请求其所有消息 ---
                        elif message['type'] == 'load_messages':
                            conv_id = params['conversation_id']
                            msg_sql = "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
                            messages = self._db_query_all(msg_sql, (conv_id,))
                            response = json.dumps({
                                'status': 200,
                                'type': 'all_messages',
                                'data': messages
                            })
                            await socket.send(response)

                        # --- 【新】处理前端点击“新聊天”按钮 ---
                        elif message['type'] == 'create_new_chat':
                            # 用当前时间创建一个默认标题
                            title = f"新聊天 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                            sql = "INSERT INTO conversations (title, created_at) VALUES (?, ?)"
                            new_id = self._db_execute(sql, (title, self._get_timestamp()))
                            response = json.dumps({
                                'status': 200,
                                'type': 'new_chat_created',
                                'data': { 'id': new_id, 'title': title, 'created_at': self._get_timestamp() }
                            })
                            await socket.send(response)

                        # --- 【新】修改：处理用户发送的消息 ---
                        elif message['type'] == 'send_message': # 原来的 'run_ollama_prompt'
                            conv_id = params['conversation_id']
                            prompt = params['prompt']
                            
                            # 使用线程来处理，防止阻塞 WebSocket
                            threading.Thread(
                                target=self.handle_send_message,
                                args=(conv_id, prompt, WebLogger(sendSocket, 'Ollama')),
                                daemon=True
                            ).start()

                        # --- 【旧】保留 'run' 逻辑（虽然新前端不用了，但保留总没错） ---
                        elif message['type'] == 'run':
                            conf = ManagerConfig(
                                params['datasetName'], params['splitterName'], params['ratio'],
                                params['model'], params['judgerName'],
                                dataset_params={'logger': WebLogger(sendSocket, params['datasetName'])}, 
                                splitter_params={'logger': WebLogger(sendSocket, params['splitterName'])}, 
                                model_params={'logger': WebLogger(sendSocket, params['model'][0])}, 
                                judger_params={'logger': WebLogger(sendSocket, params['judgerName'])}, 
                            )
                            self.run(conf) # self.run 可能是同步的，最好也用线程
                        
                        # --- 【旧】保留 'overview' 逻辑 ---
                        elif message['type'] == 'overview':
                            response = json.dumps({
                                'status': 200, 'type': 'overview', 
                                'data': {
                                    'datasets': list(self.datasets.keys()), 
                                    'details':{name: content.__getcontent__() for name, content in self.datasets.items()},
                                    'details_2':{name: content.__getnewcontent__() for name, content in self.datasets.items()},
                                    'splitters': list(self.splitters.keys()), 
                                    'models': list(self.models.keys()), 
                                    'params':{name: content.__getparams__() for name, content in self.models.items()},
                                    'judgers': list(self.judgers.keys())
                                }
                            },cls=MyEncoder)
                            await socket.send(response)

                        else:
                            response = json.dumps({'status': 500, 'data': 'unknown type'})
                            await socket.send(response)

                    except Exception as e:
                        # 通用错误处理
                        logger.print(f"[系统严重错误] {traceback.format_exc()}")
                        response = json.dumps({
                            'status': 500, 
                            'type': 'error', 
                            'data': traceback.format_exc()
                        })
                        await socket.send(response)


        async def run_server():
            server = await websockets.serve(onRecv, host, port)
            print('server is running at [{}:{}]...'.format(host, port))
            try:
                await asyncio.Event().wait()
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(run_server())