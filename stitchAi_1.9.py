from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import os
import requests
import concurrent.futures
from openai import OpenAI
from docx import Document  # 导入python-docx库
import torch
from transformers import BertTokenizer, BertModel
import json

app = Flask(__name__)

# 设置文件上传的目录
UPLOAD_FOLDER = 'uploads/'
DOWNLOAD_FOLDER = 'downloads/'  # 新增下载文件夹
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

class WenXinYiYanService:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint

    def chatMessage(self, message):
        # 正确缩进的代码
        payload = json.dumps({
            "messages": [{
                "role": "user",
                "content": message
            }],
            "temperature": 0.95,
            "top_p": 0.8,
            "penalty_score": 1,
            "enable_system_memory": False,
            "disable_search": False,
            "enable_citation": False
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.api_endpoint, headers=headers, data=payload)

        # 打印请求内容和响应内容
        print("Request Payload:", payload)
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)

        try:
            # 检查返回的 JSON 格式
            response_json = response.json()  # 尝试将响应解析为 JSON
            print("Response JSON:", response_json)

            # 如果返回的 JSON 格式正确，尝试从中提取 'answer'
            return response_json.get('result', '未能从文心一言API获取答案。')
        except ValueError as e:
            print(f"JSON解析错误：{e}")
            return "文心一言API返回的格式错误。"
        except Exception as e:
            print(f"发生错误：{e}")
            return f"请求文心一言API时发生错误：{e}"

class KimiService:
    def __init__(self, api_key, base_url, max_tokens=1024):  # 增加了max_tokens参数
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_tokens = max_tokens
    
    def chatMessage(self, message):
        try:
            completion = self.client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "user", "content": message}],
                temperature=0.3,
                max_tokens=self.max_tokens  # 设置更大的max_tokens值
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"调用Kimi API时出错：{e}")
            return None

class AIAssistantDispatcher:
    def __init__(self):
        self.assistants = {
            'wenxinyiyan': WenXinYiYanService(api_endpoint="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.d1ee8d6a70fcd162f7a26936f614650c.2592000.1736343757.282335-116580611"),
            'kimi': KimiService(api_key="sk-7H3ZdLK7dqilVMGQUUbtXQ1hrkf951tNYNR1TnEgNbG9gkGF", base_url="https://api.moonshot.cn/v1")
        }

        # 初始化BERT模型和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('C:/Users/12864/Desktop/bert-base-uncased')
        self.model = BertModel.from_pretrained('C:/Users/12864/Desktop/bert-base-uncased')

    def get_best_response(self, message, file_path=None):
        if file_path:
            # 如果有文件，就将文件内容作为消息的一部分
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    message += " " + content
            except Exception as e:
                return f"文件读取失败：{e}"

        # 使用并行请求多个AI助手生成不同部分的回答
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(assistant.chatMessage, message): name for name, assistant in self.assistants.items()}
            responses = {}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    response = future.result()
                    if response:
                        responses[name] = response
                except Exception as e:
                    responses[name] = f"调用{futures[future]}时发生错误：{e}"

        # 合并不同模型的回答，并考虑相关性得分
        merged_response = self.merge_responses(responses, message)

        # 确保回答长度符合要求
        if len(merged_response) < 100:
            merged_response += " 这是为了增加回答的长度而添加的额外内容。"

        return merged_response if merged_response else "对不起，我无法回答您的问题。"

    def calculate_relevance_score(self, response, query):
        # 对问题和回答进行tokenization
        inputs_query = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs_response = self.tokenizer(response, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # 获取BERT的输出
        with torch.no_grad():
            query_output = self.model(**inputs_query)
            response_output = self.model(**inputs_response)

        # 提取[CLS] token的输出作为文本的表示
        query_embedding = query_output.last_hidden_state[:, 0, :]
        response_embedding = response_output.last_hidden_state[:, 0, :]

        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(query_embedding, response_embedding)

        return similarity.item()

    def merge_responses(self, responses, query):
        # 对每个回答计算相关性得分
        scored_responses = []
        for model_name, response in responses.items():
            score = self.calculate_relevance_score(response, query)
            scored_responses.append((score, response))

        # 根据相关性得分降序排序
        scored_responses.sort(reverse=True, key=lambda x: x[0])

        # 合并相关性最高的几个回答（可以根据需要调整合并策略）
        final_response = " ".join([response for score, response in scored_responses[:3]])

        return final_response

# 定义默认路由，显示上传和提问的表单
@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI Assistant</title>
        <!-- 引入Bootstrap CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    </head>
    <body>
        <div class="container mt-5">
            <h1 class="text-center">AI Assistant</h1>
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <form id="askForm" class="mb-3">
                        <div class="input-group">
                            <input type="text" class="form-control" placeholder="Enter your question" name="question" id="question" required>
                            <div class="input-group-append">
                                <button type="submit" class="btn btn-outline-secondary">Ask a Question</button>
                            </div>
                        </div>
                    </form>
                    <div id="answer" class="mt-3">
                        <!-- 答案会显示在这里 -->
                    </div>

                    <div id="fileLinks" class="mt-3">
                        <!-- 文件下载链接会显示在这里 -->
                    </div>
                </div>
            </div>
        </div>

        <script>
            $(document).ready(function(){
                $("#askForm").submit(function(event){
                    event.preventDefault();  // 阻止默认提交行为

                    var question = $("#question").val();  // 获取用户输入的问题

                    $.ajax({
                        url: "/ask",
                        type: "POST",
                        data: { question: question },
                        success: function(response) {
                            // 显示答案
                            $("#answer").html("<h4>Answer:</h4><p>" + response.answer + "</p>");
                            // 显示下载链接
                            var fileLinks = "";
                            if (response.txt_download_url) {
                                fileLinks += "<a href='" + response.txt_download_url + "' class='btn btn-primary'>Download TXT</a> ";
                            }
                            if (response.docx_download_url) {
                                fileLinks += "<a href='" + response.docx_download_url + "' class='btn btn-secondary'>Download DOCX</a>";
                            }
                            $("#fileLinks").html(fileLinks);
                            $("#question").val("");  // 清空输入框
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    ''')

# 文件上传路由
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

# 路由：处理用户提问
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = dispatcher.get_best_response(question)

    # 下载文档生成
    txt_file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "answer.txt")
    docx_file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "answer.docx")

    # 保存为TXT文件
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(answer)

    # 保存为DOCX文件
    doc = Document()
    doc.add_paragraph(answer)
    doc.save(docx_file_path)

    return jsonify({
        'answer': answer,
        'txt_download_url': f"/download/{os.path.basename(txt_file_path)}",
        'docx_download_url': f"/download/{os.path.basename(docx_file_path)}"
    })

# 路由：下载文件
@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)

if __name__ == '__main__':
    dispatcher = AIAssistantDispatcher()
    app.run(debug=True)

