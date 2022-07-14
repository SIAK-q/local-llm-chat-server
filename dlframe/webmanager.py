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
                'content': '[{}]: '.format(self.name) + ' '.join([str(_) for _ in params]) + end
            }
        }))
        return super().print(*params, end=end)
    
    def image(self, image) -> Logger:
        
        with open(image,'rb') as f:
            base64code = base64.b64encode(f.read())
              
        base64code=base64code.decode("utf-8")


        imgdata=base64.b64decode(base64code)
        # print(imgdata)
        # print("****************")
        # file=open('D:/tui.jpg','wb') #新建一个jpg文件，把信息写入该文件，第一个参数是文件路径
        # file.write(imgdata)
        # file.close()
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
import numpy as np
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
class WebManager(Manager):
    def __init__(self) -> None:
        super().__init__()

    def start(self, host='0.0.0.0', port=8765) -> None:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        async def onRecv(socket, path):
            msgIdx = -1
            sendSocket = SendSocket(socket)
            async for message in socket:
                msgIdx += 1
                message = json.loads(message)
                # print(msgIdx, message)

                # key error
                if not all([key in message.keys() for key in ['type', 'params']]):
                    response = json.dumps({
                        'status': 500, 
                        'data': 'no key param'
                    })
                    await socket.send(response)
                
                # key correct
                else:
                    if message['type'] == 'overview':
                        response = json.dumps({
                            'status': 200, 
                            'type': 'overview', 
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

                    elif message['type'] == 'run':
                        params = message['params']
                        conf = ManagerConfig(
                            params['datasetName'], 
                            params['splitterName'], 
                            params['ratio'],
                            params['model'], 
                            params['judgerName'],
                            

                            dataset_params={'logger': WebLogger(sendSocket, params['datasetName'])}, 
                            splitter_params={'logger': WebLogger(sendSocket, params['splitterName'])}, 
                            model_params={'logger': WebLogger(sendSocket, params['model'][0])}, 
                            judger_params={'logger': WebLogger(sendSocket, params['judgerName'])}, 
                        )


                        try:
                            self.run(conf)
                        except Exception as e:
                            response = json.dumps({
                                'status': 200, 
                                'type': 'print', 
                                'data': {
                                    'content': traceback.format_exc()
                                }
                            })
                            await socket.send(response)
                    
                    # unknown key
                    else:
                        response = json.dumps({
                            'status': 500, 
                            'data': 'unknown type'
                        })
                        await socket.send(response)

        print('server is running at [{}:{}]...'.format(host, port))
        event_loop.run_until_complete(websockets.serve(onRecv, host, port))
        event_loop.run_forever()