import requests
import concurrent.futures
from openai import OpenAI

class ErnieBotService:
    def __init__(self, api_key, api_endpoint):
        self.api_key = api_key
        self.api_endpoint = api_endpoint

    def chatMessage(self, message):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'query': message,
            # 根据实际API要求添加其他参数
        }
        response = requests.post(self.api_endpoint, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get('answer', '未能从文心一言API获取答案。')
        else:
            return f'文心一言API调用失败，状态码：{response.status_code}'

class KimiService:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
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
            'ernie': ErnieBotService(api_key="XVdKBjs4gqW15WcrnEAHRIQk", api_endpoint="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.77d1a835729d0197520837d72f489451.2592000.1733388232.282335-116123794"),
            'kimi': KimiService(api_key="sk-8Vur9kyRCCdxXDuKnMPi4NAsgln29BFJ4qUb3KhMRfFzz00n", base_url="https://api.moonshot.cn/v1")
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
user_question = input("请输入您的问题：")
best_answer = dispatcher.get_best_response(user_question)
print(best_answer)