from chemcrow.tools.search import paper_search
from langchain.chat_models import ChatOpenAI

# 这里假设你已经配置好了 OPENAI_API_KEY，并且安装了相应依赖
llm = ChatOpenAI(model="gpt-4", temperature=0.1)

papers = paper_search(llm, "organocatalyst carbon dioxide conversion")
print("检索到的文献：", papers)