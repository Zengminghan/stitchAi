from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import requests
import concurrent.futures
from openai import OpenAI
from docx import Document
import torch
from transformers import BertTokenizer, BertModel
import json
import re
from datetime import datetime


app = Flask(__name__)

# 设置文件上传和下载目录
UPLOAD_FOLDER = 'uploads/'
DOWNLOAD_FOLDER = 'downloads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

#保存历史记录
conversation_history = []

# WenXinYiYanService服务
class WenXinYiYanService:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint

    def chatMessage(self, message):
        payload = json.dumps({"messages": [{"role": "user", "content": message}]})
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.api_endpoint, headers=headers, data=payload)
        try:
            return response.json().get('result', '未获取答案')
        except Exception as e:
            return f"文心一言API错误: {e}"

# KimiService服务
class KimiService:
    def __init__(self, api_key, base_url, max_tokens=1024):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_tokens = max_tokens

    def chatMessage(self, message):
        try:
            completion = self.client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "user", "content": message}],
                max_tokens=self.max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Kimi API错误: {e}"

#集成其他服务模块......


# AIAssistantDispatcher分发器，实现集成功能
class AIAssistantDispatcher:
    def __init__(self):

        #处理想要集成的Ai平台API密钥
        self.assistants = {
            'wenxinyiyan': WenXinYiYanService(api_endpoint="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.d1ee8d6a70fcd162f7a26936f614650c.2592000.1736343757.282335-116580611"),
            'kimi': KimiService(api_key="sk-7H3ZdLK7dqilVMGQUUbtXQ1hrkf951tNYNR1TnEgNbG9gkGF", base_url="https://api.moonshot.cn/v1")
        }

        #定义初始化BERT分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained('C:/Users/12864/Desktop/bert-base-uncased')
        self.model = BertModel.from_pretrained('C:/Users/12864/Desktop/bert-base-uncased')
      
      # 获取AI回答
    def get_responses(self, message):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(a.chatMessage, message): name for name, a in self.assistants.items()}
            return {futures[future]: future.result() for future in concurrent.futures.as_completed(futures)}
    
      # 结构分类
    def identify_structure(self, text):
        parts = {'start': '', 'body': {}, 'end': ''}
        parts['start'] = text.split('\n', 1)[0] if '\n' in text else text
        if '\n' in text:
            parts['end'] = text.rsplit('\n', 1)[1] if '\n' in text else ''
            body_text = text.split('\n', 1)[1] if '\n' in text else text
            for match in re.finditer(r'(\d+)\.\s(.*?)(?=\d+\.\s|\Z)', body_text, re.S):
                number, content = match.group(1), match.group(2).strip()
                if number not in parts['body']:
                    parts['body'][number] = [content]
                else:
                    parts['body'][number].append(content)
        return parts

      # 合并开头、内容、结尾
    def merge_parts(self, parts_list):
        merged = {'start': '', 'body': {}, 'end': ''}
        merged['start'] = parts_list[0]['start'] if parts_list else ''
        merged['end'] = parts_list[-1]['end'] if parts_list else ''
        for parts in parts_list:
            for number, contents in parts['body'].items():
                if number not in merged['body']:
                    merged['body'][number] = ' '.join(contents)
                else:
                    # 生成衔接词并合并内容
                    connector = self.generate_smooth_connector(merged['body'][number], contents[0])
                    merged['body'][number] = f"{merged['body'][number]} {connector} " + ' '.join(contents)
        return merged

    def generate_smooth_connector(self, previous_text, next_text):
        # 使用BERT模型对两段文本进行编码
        inputs = self.tokenizer(previous_text, next_text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取最后一层隐藏状态
            last_hidden_states = outputs.last_hidden_state
            # 取最后一层的第一个token（通常是[CLS]标记）的隐藏状态
            previous_cls_embedding = last_hidden_states[:, 0, :]
            next_cls_embedding = last_hidden_states[:, 1, :]
            # 计算两个句子的相似度
            similarity = torch.nn.functional.cosine_similarity(previous_cls_embedding, next_cls_embedding)
        
        # 根据上下文生成衔接词，这里我们使用一个简单的启发式方法
        if similarity > 0.8:
            connector = "并且"  # 高相似度使用更平滑的衔接词
        elif similarity < 0.3:
            connector = "然而"  # 低相似度使用转折词
        else:
            connector = "此外"  # 中等相似度使用补充词
        return connector
        
      # 优化合并
    def optimize_with_kkt(self, text1, text2):
        inputs1 = self.tokenizer(text1, return_tensors='pt', truncation=True, max_length=512)
        inputs2 = self.tokenizer(text2, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            emb1 = self.model(**inputs1).last_hidden_state[:, 0, :]
            emb2 = self.model(**inputs2).last_hidden_state[:, 0, :]
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        return text1 if similarity > 0.8 else f"{text1} {text2}"

      # 处理回答
    def process_responses(self, responses):
        structures = [self.identify_structure(resp) for resp in responses.values()]
        merged_parts = self.merge_parts(structures)
        optimized_start = self.optimize_with_kkt(merged_parts['start'], merged_parts['start'])
        optimized_body = {}
        for number, content in merged_parts['body'].items():
            optimized_body[number] = self.optimize_with_kkt(content, content)
        optimized_end = self.optimize_with_kkt(merged_parts['end'], merged_parts['end'])
        result = optimized_start
        for number in sorted(optimized_body.keys(), key=lambda x: int(x.strip())):
            result += '\n' + number + '. ' + optimized_body[number]
        result += '\n' + optimized_end
        return result
      
      # 将当前对话添加到历史记录
    def save_conversation(self, question, result, timestamp):
    # 添加时间戳到对话历史
     conversation_history.append({
        'question': question,
        'answer': result,
        'timestamp': timestamp
    })

dispatcher = AIAssistantDispatcher()

# 首页路由
@app.route('/')
def home():
    return render_template('home.html')

# 上传文件路由
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path})

# 处理提问路由
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    responses = dispatcher.get_responses(question)
    result = dispatcher.process_responses(responses)
    
    # 获取当前时间并格式化为时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 保存当前对话到历史记录
    dispatcher.save_conversation(question, result, timestamp)  # 确保传递所有必需的参数

    # 保存为TXT文件
    txt_file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "answer.txt")
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(result)

    # 保存为DOCX文件
    docx_file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "answer.docx")
    doc = Document()
    doc.add_paragraph(result)
    doc.save(docx_file_path)

    return jsonify({
        'answer': result,
        'txt_download_url': f"/download/{os.path.basename(txt_file_path)}",
        'docx_download_url': f"/download/{os.path.basename(docx_file_path)}",
        'conversation_history': conversation_history  # 将对话历史记录添加到响应中
    })

#下载路由
@app.route('/download/<filename>')
def download(filename):
   return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)

#查看历史记录路由
@app.route('/history', methods=['GET'])
def get_conversation_history():
    print("GET request received for /history")  # 确认请求到达后端

    return jsonify({'conversation_history': conversation_history})

@app.route('/generate', methods=['POST'])
def generate():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    long_text = generate_long_text(query)
    return jsonify({'long_text': long_text})

if __name__ == '__main__':
    print("Starting stitchAi app...")
    app.run(debug=True)