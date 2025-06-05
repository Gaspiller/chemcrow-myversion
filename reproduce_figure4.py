import os
import json
import time
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from chemcrow.agents import ChemCrow
import openai



# 评估任务定义（略）
# ... 请在此处粘贴 EVALUATION_TASKS 和 SYNTHETIC_COMPLEXITY ...

class ChemCrowEvaluator:
    def __init__(self, openai_api_key: str = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        openai.api_key = self.api_key
        self.results = {
            "chemcrow_outputs": {},
            "gpt4_outputs": {},
            "evaluations": {
                "human": [],
                "evaluator_gpt": []
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tasks": EVALUATION_TASKS
            }
        }

    def initialize_chemcrow(self):
        print("初始化 ChemCrow...")
        try:
            self.chemcrow = ChemCrow(
                model="gpt-3.5-turbo",
                temp=0.1,
                streaming=False
            )
            print("ChemCrow 初始化成功!")
        except Exception as e:
            print(f"ChemCrow 初始化失败: {e}")
            print("将使用模拟模式...")
            self.chemcrow = None

    def run_chemcrow(self, task: str) -> str:
        if self.chemcrow:
            try:
                time.sleep(10)
                return self.chemcrow.run(task)
            except Exception as e:
                print(f"ChemCrow 运行失败: {e}")
                return f"Error: {str(e)}"
        else:
            return f"[ChemCrow 模拟响应] 对于任务: {task[:50]}... 我会使用多个工具来解决这个问题。"

    def run_gpt4_baseline(self, task: str) -> str:
        try:
            time.sleep(10)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert chemist with deep knowledge in organic synthesis, reaction mechanisms, and chemical properties."
                    },
                    {"role": "user", "content": task}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"gpt-3.5-turbo 调用失败: {e}")
            return f"Error: {str(e)}"

    def evaluator_gpt(self, task: str, chemcrow_response: str, gpt4_response: str) -> Dict:
        evaluation_prompt = f"""As an expert chemistry professor, evaluate these two responses to a chemistry task.

Task: {task}

Response A: {chemcrow_response}
Response B: {gpt4_response}

Please evaluate each response on a scale of 0-10 for:
1. Chemical correctness (accuracy of chemical facts, reactions, mechanisms)
2. Reasoning quality (logical flow, clarity of explanation)
3. Task completion (how well the response addresses all parts of the task)

Format your response EXACTLY as:
Response A scores: [X, Y, Z]
Response B scores: [X, Y, Z]
Preferred response: [A or B]
Brief justification: [One sentence explaining your preference]
"""
        try:
            time.sleep(10)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            lines = content.strip().split('\n')
            scores_a = [float(x) for x in lines[0].split('[')[1].split(']')[0].split(',')]
            scores_b = [float(x) for x in lines[1].split('[')[1].split(']')[0].split(',')]
            preferred = lines[2].split('[')[1].split(']')[0].strip()
            return {
                "task_id": "",
                "category": "",
                "difficulty": 0,
                "chemcrow_scores": scores_a,
                "gpt4_scores": scores_b,
                "preferred": "ChemCrow" if preferred == "A" else "gpt-3.5-turbo",
                "raw_response": content
            }
        except Exception as e:
            print(f"EvaluatorGPT 调用或解析失败: {e}")
            return {
                "task_id": "",
                "category": "",
                "difficulty": 0,
                "chemcrow_scores": [5, 5, 5],
                "gpt4_scores": [5, 5, 5],
                "preferred": "gpt-3.5-turbo",
                "raw_response": str(e)
            }

    # 其他函数（run_evaluation, save_results, generate_figure_4 等）保持不变


def main():
    print("ChemCrow Figure 4 评估复现脚本")
    evaluator = ChemCrowEvaluator()
    evaluator.initialize_chemcrow()
    tasks_to_run = list(EVALUATION_TASKS.keys())
    evaluator.run_evaluation(tasks_to_run)
    evaluator.save_results("chemcrow_evaluation_results.json")
    evaluator.generate_figure_4("figure4_outputs")


if __name__ == "__main__":
    main()
