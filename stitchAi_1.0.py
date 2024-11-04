import concurrent.futures
from openai import OpenAI

class ErnieBotService:
    def chatMessage(self, message):
        # 这里应该是调用文心一言API的代码
        # 模拟返回
        return "这是来自ErnieBotService的回答。"

class KimiService:
    def __init__(self):
        self.client = OpenAI(api_key="sk-lkabAq2MCj41AHTTgkHWmTxnc6tN2L8k4l5iDqzt6ykhb5zx", base_url="https://api.moonshot.cn/v1")
    
    def chatMessage(self, message):
        try:
            completion = self.client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "user", "content": message}],
                temperature=0.3,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"调用Kimi API时出错：{e}")
            return None

class AIAssistantDispatcher:
    def __init__(self):
        self.assistants = {
            'ernie': ErnieBotService(),
            'kimi': KimiService(),
        }
    
    def get_best_response(self, message):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(assistant.chatMessage, message): name for name, assistant in self.assistants.items()}
            responses = {}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    response = future.result()
                except Exception as exc:
                    print(f'{name} generated an exception: {exc}')
                    response = None
                responses[name] = response
        
        # 选择最佳回答的逻辑
        best_response = max(responses.items(), key=lambda item: (len(item[1]) if item[1] else 0, item[0]))[1]
        
        return best_response if best_response else "对不起，我无法回答您的问题。"

# 使用分发器
dispatcher = AIAssistantDispatcher()
user_question = "你好，我想知道故宫介绍"
best_answer = dispatcher.get_best_response(user_question)
print(best_answer)