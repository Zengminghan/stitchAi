# longwriter_service.py
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_long_text(query, max_length=32768):
    # 加载模型和分词器
    model_name = "THUDM/LongWriter-glm4-9b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 编码输入文本
    input_ids = tokenizer.encode(query, return_tensors='pt')

    # 生成长文本
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=False,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 解码输出文本
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result