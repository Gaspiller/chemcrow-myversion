from rxn4chemistry import RXN4ChemistryWrapper

# 替换成你的真实 API 密钥 和 项目 ID
api_key = 'YOUR_API_KEY'
project_id = '6840ac33c41c812d278451bb'

rxn_wrapper = RXN4ChemistryWrapper(api_key=api_key)

# 设置已有的项目
rxn_wrapper.set_project(project_id)

# 提交反应物 SMILES，预测产物
response = rxn_wrapper.predict_product('CCO')  # 乙醇

# 打印响应内容（调试用）
print("Prediction submission response:", response)

# 获取 prediction_id 并查询结果
prediction_id = response.get('prediction_id')
if prediction_id:
    result = rxn_wrapper.get_predict_product_results(prediction_id)
    print("✅ Predicted product result:", result)
else:
    print("❌ Failed to get prediction_id from response:", response)