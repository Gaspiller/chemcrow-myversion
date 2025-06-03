from chemcrow.agents import ChemCrow

# 初始化 ChemCrow 代理
agent = ChemCrow(model="gpt-4", temp=0.1)

# 示例任务：计算 Aspirin 的分子量
result = agent.run("What is the molecular weight of aspirin?")
print(result)