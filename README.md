介绍 该stitchAi集成上市所有Ai平台，可跨多个平台集成ai答案，亦或者二次或更高级数迭代答案，最后输出答案

软件架构说明 该代码实现使用vscode进行，需要自行配置好python环境，搭建好conda，同时配置好python 3.13版本

使用说明

在conda中配置好openai，这使得可以调用不同的API
pip install --upgrade openai
版本迭代

1.stitchAi尚在开发过程中，目前推出1.0版本 2.stitchAi v1.0中支持一种Ai平台：kimi  3.在temperature=0.3中，表示该内容输出更具精确性，简洁性，1.0支持自由调节该维度，使得答案根据实际进行你想要的效果 4.目前尚未实现答案的二次迭代，计划于2024年11月中旬实现该答案自动二次迭代并输出，并上架该v1.

遇到的问题 

1.由于目前是通过API来实现问答，导致目前回答字数各方面不好，由kimi单独实现问答会更为差劲（即实现简单问答） 

2.在精确度方面，单个kimi通过API实现问答存在精度不准确的问题 

3.在准确度函数中，还需要进一步优化，以实现更为准确详细的回答

计划：

1.目前已实现集成文言一心API与kimi实现基本的输出回答，由于1.0只是测试数据，在1.1版本中，可以很明显的感受到集成后会比集成前在语义，问答，格式上具有优势 

2.目前尚未实现答案二次迭代过程（考虑到免费的API本身存在输出语句精度长度较少）所以只能实现答案二次迭代，无法能够证明二次迭代后能否满足字数要求 

3.在1.1版本基础上，希望能够开发出能够集成豆包大模型同时后续版本会推出答案二次迭代增加准确性和精度

后续： 添加对应一个kimiAPI数据进行对比 2024年11月5日更新

字数获取长度建议已放置在包中