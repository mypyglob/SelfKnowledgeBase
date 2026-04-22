# SelfKnowledgeBase - 搭建自己的知识库系统
# 安装依赖
pip install -r requirements.txt
# 方式一：直接运行
python -m app.main

# 方式二：uvicorn 启动（推荐生产环境）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


# 默认使用GLM的模型，在.env 文件中更新自己的OPENAI_API_KEY 即可使用

<img width="1524" height="985" alt="image" src="https://github.com/user-attachments/assets/edb3f820-803e-4751-ad3f-1f7c922762e0" />
