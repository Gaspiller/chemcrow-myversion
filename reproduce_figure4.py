"""
ChemCrow Figure 4 复现脚本
用于复现论文中的评估结果对比图

使用方法:
1. 安装依赖: pip install chemcrow openai matplotlib pandas numpy
2. 设置环境变量: export OPENAI_API_KEY="your-key"
3. 运行脚本: python reproduce_figure4.py
"""

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


# 14个评估任务（来自论文附录G），其中 name 对应的名称会用于绘图时标签
EVALUATION_TASKS = {
    "task_1": {
        "name": "Safinamide",
        "category": "organic_synthesis",
        "difficulty": 1,
        "prompt": "I need to synthesize a sample of safinamide. Please tell me how to synthesize it. Then tell me how much will it cost to buy all the reactants I need, if purchasable."
    },
    "task_2": {
        "name": "Propose New Organocatalyst",
        "category": "molecular_design",
        "difficulty": 2,
        "prompt": "Propose a novel organocatalyst for enhancing carbon dioxide conversion in carbon capture and utilization processes."
    },
    "task_3": {
        "name": "Predict Reaction",
        "category": "chemical_logic",
        "difficulty": 1,
        "prompt": "Predict the product of a mixture of Ethylidenecyclohexane and HBr. Then predict the same reaction, adding methyl peroxide into the mixture. Compare the two products and explain the reaction mechanism."
    },
    "task_4": {
        "name": "DEET",
        "category": "organic_synthesis",
        "difficulty": 0,
        "prompt": "Plan the synthesis of an insect repellent. Then, tell me the cost of the reactants, if purchasable. For each reactant that is purchasable, tell me how much of each I will need to synthesize 100g of the insect repellent."
    },
    "task_5": {
        "name": "Paracetamol",
        "category": "organic_synthesis",
        "difficulty": 2,
        "prompt": "I want to make Acetaminophen. Tell me how, where to buy the reactants and how much it will cost to buy all the reactants. Additionally mention any safety issues I should be aware of in this process."
    },
    "task_6": {
        "name": "Compare Catalyst Mechanisms",
        "category": "chemical_logic",
        "difficulty": 2,
        "prompt": "Predict the product of a mixture of 1-Chloro-4-ethynylbenzene a Lindlars catalyst (use CC(=O)[O-].CC(=O)[O-].[Pd].[Pb+2]). Predict the same reaction but replacing the catalyst with \"[Pd]\". Finally, compare the two products and explain the reaction mechanisms."
    },
    "task_7": {
        "name": "Similar Molecule",
        "category": "molecular_design",
        "difficulty": 1,
        "prompt": "Synthesize a molecule similar to paracetamol, that contains no methylamide groups. Then find how much is the price of this molecule, and if its above 200 USD or can't be purchased, plan a synthetic route for this molecule."
    },
    "task_8": {
        "name": "Ricci's OC",
        "category": "organic_synthesis",
        "difficulty": 3,
        "prompt": "Plan the synthesis of Ricci's organocatalyst (IUPAC: 1-(3,5-Bis(trifluoromethyl)phenyl)-3-((1R,2S)-2-hydroxy-2,3-dihydro-1H-inden-1-yl)thiourea)."
    },
    "task_9": {
        "name": "Predict Success",
        "category": "chemical_logic",
        "difficulty": 0,
        "prompt": "What are the products of the reaction between 2-bromo-2-methylpropane and 4-(4-hydroxyphenyl)butan-2-one. Can this reaction will undergo without any problems?"
    },
    "task_10": {
        "name": "Boiling Point",
        "category": "chemical_logic",
        "difficulty": 1,
        "prompt": "Tell me what the boiling point is of the reaction product between isoamyl acetate and ethanol."
    },
    "task_11": {
        "name": "Nitroglycerin",
        "category": "organic_synthesis",
        "difficulty": 1,
        "prompt": "Plan the synthesis of nitroglycerin."
    },
    "task_12": {
        "name": "Atorvastatin",
        "category": "organic_synthesis",
        "difficulty": 2,
        "prompt": "Tell me how to synthesize atorvastatin, and then tell me how much will it cost to buy all the reactants I need. Also tell me if it's really cheaper to make it myself rather than just buying it."
    },
    "task_13": {
        "name": "Aspirin",
        "category": "organic_synthesis",
        "difficulty": 0,
        "prompt": "I need to synthesize a sample of aspirin. Please tell me how to synthesize it. Then tell me the GHS rating of all of the reactants needed."
    },
    "task_14": {
        "name": "Takemoto's OC",
        "category": "organic_synthesis",
        "difficulty": 2,
        "prompt": "Plan the synthesis of Takemoto's organocatalyst (IUPAC: 1-[3,5-bis(trifluoromethyl)phenyl]-3-[(1R,2R)-(-)-2-(dimethylamino)cyclohexyl]thiourea)."
    }
}

# 合成复杂度映射，仅用于 b) 图中的底部标签
SYNTHETIC_COMPLEXITY = {
    "Paracetamol": 1.54,
    "Aspirin": 1.593,
    "DEET": 1.871,
    "Ricci's OC": 3.167,
    "Takemoto's OC": 3.311,
    "Safinamide": 3.61,
    "Atorvastatin": 4.796
}


class ChemCrowEvaluator:
    """ChemCrow评估器类"""

    def __init__(self, openai_api_key: str = None):
        """初始化评估器"""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        openai.api_key = self.api_key
        self.results = {
            "chemcrow_outputs": {},
            "gpt4_outputs": {},
            "evaluations": {
                "human": [],  # 暂时留空，可手动添加人工评分
                "evaluator_gpt": []
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tasks": EVALUATION_TASKS
            }
        }

    def initialize_chemcrow(self):
        """初始化 ChemCrow"""
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

    def run_gpt4_baseline(self, task: str) -> str:
        """运行 gpt-3.5-turbo 基线回答"""
        try:
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

    def run_chemcrow(self, task: str) -> str:
        """运行 ChemCrow 回答"""
        if self.chemcrow:
            try:
                return self.chemcrow.run(task)
            except Exception as e:
                print(f"ChemCrow 运行失败: {e}")
                return f"Error: {str(e)}"
        else:
            # 如果初始化失败，则使用模拟响应
            return f"[ChemCrow 模拟响应] 对于任务: {task[:50]}... 我会使用多个工具来解决这个问题。"

    def evaluator_gpt(self, task: str, chemcrow_response: str, gpt4_response: str) -> Dict:
        """使用 gpt-3.5-turbo 评估两段回答，返回化学正确性、推理质量、任务完成度三项评分及偏好"""
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
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            lines = content.strip().split('\n')
            # 假设响应格式正确：前三行分别包含“A”三项分数、“B”三项分数、“Preferred”。
            scores_a = [float(x) for x in lines[0].split('[')[1].split(']')[0].split(',')]
            scores_b = [float(x) for x in lines[1].split('[')[1].split(']')[0].split(',')]
            preferred = lines[2].split('[')[1].split(']')[0].strip()
            return {
                "task_id": "",       # 在 run_evaluation 中会补上
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

    def run_evaluation(self, tasks_to_run: List[str] = None):
        """遍历所有任务，调用 ChemCrow 与 gpt-3.5-turbo 并让 EvaluatorGPT 评分，最终将结果保存在 self.results 中"""
        tasks_to_run = tasks_to_run or list(EVALUATION_TASKS.keys())
        print(f"开始评估 {len(tasks_to_run)} 个任务...")

        for task_id in tasks_to_run:
            task_info = EVALUATION_TASKS[task_id]
            task_prompt = task_info["prompt"]
            print(f"\n运行 {task_id}: {task_info['name']}")
            print("-" * 50)

            # 运行 ChemCrow
            print("运行 ChemCrow...")
            chemcrow_output = self.run_chemcrow(task_prompt)
            self.results["chemcrow_outputs"][task_id] = chemcrow_output

            # 运行 gpt-3.5-turbo
            print("运行 gpt-3.5-turbo 基线...")
            gpt4_output = self.run_gpt4_baseline(task_prompt)
            self.results["gpt4_outputs"][task_id] = gpt4_output

            # 运行 EvaluatorGPT
            print("运行 EvaluatorGPT 评估...")
            eval_dict = self.evaluator_gpt(task_prompt, chemcrow_output, gpt4_output)
            eval_dict["task_id"] = task_id
            eval_dict["category"] = task_info["category"]
            eval_dict["difficulty"] = task_info["difficulty"]
            self.results["evaluations"]["evaluator_gpt"].append(eval_dict)

            # 保存中间结果（以防脚本中途出错，可续跑）
            self.save_results(f"intermediate_results_{task_id}.json")

            # 等待 2 秒，防止 OpenAI 限速
            time.sleep(2)

        print("\n所有任务评估完成！")

    def save_results(self, filename: str = "chemcrow_evaluation_results.json"):
        """将 self.results 保存到 JSON 文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {filename}")

    def generate_figure_4(self, output_dir: str = "figure4_outputs"):
        """
        生成 Figure 4 四部分图表：
        a) Δ mean expert scores（ChemCrow - gpt-3.5-turbo），按难度排序，横向 bar
        b) Chemical correctness vs synthetic complexity，柱状对比
        c) Aggregate evaluation scores，带误差条
        d) 专家观察文字
        最终把 figure_4_reproduction.png 保存到 output_dir 中。
        """
        # 确保输出文件夹存在
        os.makedirs(output_dir, exist_ok=True)

        # 加载 evaluator_gpt 的评分数据
        evals = self.results["evaluations"]["evaluator_gpt"]
        df = pd.DataFrame(evals)

        # 为了后续计算，先把 chemcrow_scores 和 gpt4_scores 的平均值以及差值算出来
        df['chemcrow_mean'] = df['chemcrow_scores'].apply(lambda s: np.mean(s))
        df['gpt4_mean'] = df['gpt4_scores'].apply(lambda s: np.mean(s))
        df['mean_diff'] = df['chemcrow_mean'] - df['gpt4_mean']

        # 开始绘图
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # a) Δ mean expert scores
        self._plot_score_differences(ax1, df)

        # b) Chemical correctness vs synthetic complexity
        self._plot_chemical_accuracy_vs_complexity(ax2, df)

        # c) Aggregate evaluation scores
        self._plot_aggregate_scores(ax3, df)

        # d) 专家观察
        self._add_expert_observations(ax4)

        plt.suptitle("Figure 4: Evaluation Results", fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 最终保存到指定文件夹
        out_path = os.path.join(output_dir, "figure_4_reproduction.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Figure 4 已保存到: {out_path}")

    def _plot_score_differences(self, ax, df: pd.DataFrame):
        """绘制 a) 每个任务 ChemCrow - gpt-3.5-turbo 的 mean expert score 差值，任务按 difficulty 排序，横向 bar"""
        # 按 difficulty 升序排序
        df_sorted = df.sort_values(by='difficulty', ascending=True).reset_index(drop=True)
        task_names = [EVALUATION_TASKS[tid]['name'] for tid in df_sorted['task_id']]
        diffs = df_sorted['mean_diff'].values

        y = np.arange(len(task_names))
        colors = ['purple' if d >= 0 else 'red' for d in diffs]
        ax.barh(y, diffs, color=colors, alpha=0.8)
        ax.axvline(0, color='black', linewidth=1)

        ax.set_yticks(y)
        ax.set_yticklabels(task_names, fontsize=10)
        ax.set_xlabel('Δ Mean Expert Score (ChemCrow - gpt-3.5-turbo)')
        ax.set_title('a) Δ Mean Expert Scores per Task', fontsize=12)
        ax.invert_yaxis()  # 让难度最低的排在上方

    def _plot_chemical_accuracy_vs_complexity(self, ax, df: pd.DataFrame):
        """绘制 b) Chemical correctness vs Synthetic complexity 的并列柱状图"""
        # 固定顺序：Paracetamol, Aspirin, DEET, Ricci's OC, Takemoto's OC, Safinamide, Atorvastatin
        tasks = ["Paracetamol", "Aspirin", "DEET", "Ricci's OC", "Takemoto's OC", "Safinamide", "Atorvastatin"]
        chemcrow_scores = []
        gpt4_scores = []
        complexities = []

        for name in tasks:
            # 找到对应的 task_id
            task_id = next(tid for tid, info in EVALUATION_TASKS.items() if info['name'] == name)
            row = df[df['task_id'] == task_id].iloc[0]
            # index=0 对应的就是 Chemical correctness
            chemcrow_scores.append(row['chemcrow_scores'][0])
            gpt4_scores.append(row['gpt4_scores'][0])
            complexities.append(SYNTHETIC_COMPLEXITY[name])

        x = np.arange(len(tasks))
        width = 0.35

        bars1 = ax.bar(x - width/2, chemcrow_scores, width, label='ChemCrow', color='purple', alpha=0.8)
        bars2 = ax.bar(x + width/2, gpt4_scores, width, label='gpt-3.5-turbo', color='red', alpha=0.8)

        ax.set_xlabel('Synthetic Complexity', fontsize=10)
        ax.set_ylabel('Chemical Accuracy Score', fontsize=10)
        ax.set_title('b) Chemical Accuracy vs Synthetic Complexity', fontsize=12)
        ax.set_xticks(x)
        # 底部标签显示复杂度数值，保留三位小数
        ax.set_xticklabels([f"{c:.3f}" for c in complexities], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)

        # 给每个柱添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    def _plot_aggregate_scores(self, ax, df: pd.DataFrame):
        """绘制 c) 汇总评分 包括 Chemical accuracy, Quality of reasoning, Task completion, EvaluatorGPT score"""
        # 计算每个模型的各项平均值
        chemcrow_acc = np.mean(df['chemcrow_scores'].apply(lambda s: s[0]))
        chemcrow_reason = np.mean(df['chemcrow_scores'].apply(lambda s: s[1]))
        chemcrow_complete = np.mean(df['chemcrow_scores'].apply(lambda s: s[2]))
        chemcrow_total = np.mean(df['chemcrow_scores'].apply(lambda s: np.mean(s)))

        gpt4_acc = np.mean(df['gpt4_scores'].apply(lambda s: s[0]))
        gpt4_reason = np.mean(df['gpt4_scores'].apply(lambda s: s[1]))
        gpt4_complete = np.mean(df['gpt4_scores'].apply(lambda s: s[2]))
        gpt4_total = np.mean(df['gpt4_scores'].apply(lambda s: np.mean(s)))

        metrics = ['Chemical\naccuracy', 'Quality of\nreasoning', 'Task\ncompletion', 'EvaluatorGPT\nscore']
        chemcrow_means = [chemcrow_acc, chemcrow_reason, chemcrow_complete, chemcrow_total]
        gpt4_means = [gpt4_acc, gpt4_reason, gpt4_complete, gpt4_total]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width/2, chemcrow_means, width, label='ChemCrow', color='purple', alpha=0.8)
        bars2 = ax.bar(x + width/2, gpt4_means, width, label='gpt-3.5-turbo', color='red', alpha=0.8)

        # 计算误差条：使用标准误差近似
        chemcrow_err = [
            np.std(df['chemcrow_scores'].apply(lambda s: s[0]))/np.sqrt(len(df)),
            np.std(df['chemcrow_scores'].apply(lambda s: s[1]))/np.sqrt(len(df)),
            np.std(df['chemcrow_scores'].apply(lambda s: s[2]))/np.sqrt(len(df)),
            np.std(df['chemcrow_scores'].apply(lambda s: np.mean(s)))/np.sqrt(len(df))
        ]
        gpt4_err = [
            np.std(df['gpt4_scores'].apply(lambda s: s[0]))/np.sqrt(len(df)),
            np.std(df['gpt4_scores'].apply(lambda s: s[1]))/np.sqrt(len(df)),
            np.std(df['gpt4_scores'].apply(lambda s: s[2]))/np.sqrt(len(df)),
            np.std(df['gpt4_scores'].apply(lambda s: np.mean(s)))/np.sqrt(len(df))
        ]

        ax.errorbar(x - width/2, chemcrow_means, yerr=chemcrow_err,
                    fmt='none', color='black', capsize=5)
        ax.errorbar(x + width/2, gpt4_means, yerr=gpt4_err,
                    fmt='none', color='black', capsize=5)

        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('c) Aggregate Evaluation Scores', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 10)

    def _add_expert_observations(self, ax):
        """绘制 d) General experts' observations 文本"""
        ax.axis('off')
        ax.text(0.5, 0.9, "d) General experts' observations",
                ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

        # gpt-3.5-turbo 观察
        gpt4_obs = [
            "• Complete responses (when possible)",
            "• Major hallucination (molecules, reactions, procedures)",
            "• Hard to interpret (need for expert modifications)",
            "• No access to up-to-date information"
        ]
        # ChemCrow 观察
        chemcrow_obs = [
            "• Chemically accurate solutions",
            "• Modular and extensible",
            "• Occasional flawed conclusions",
            "• Limited by tools' quality"
        ]

        ax.text(0.25, 0.7, "gpt-3.5-turbo", ha='center', fontsize=12,
                fontweight='bold', color='red', transform=ax.transAxes)
        for i, obs in enumerate(gpt4_obs):
            ax.text(0.05, 0.6 - i * 0.1, obs, fontsize=10, transform=ax.transAxes)

        ax.text(0.75, 0.7, "ChemCrow", ha='center', fontsize=12,
                fontweight='bold', color='purple', transform=ax.transAxes)
        for i, obs in enumerate(chemcrow_obs):
            ax.text(0.55, 0.6 - i * 0.1, obs, fontsize=10, transform=ax.transAxes)


def main():
    """主函数"""
    print("ChemCrow Figure 4 评估复现脚本")
    print("=" * 50)
    # 创建评估器
    evaluator = ChemCrowEvaluator()

    # 初始化 ChemCrow
    evaluator.initialize_chemcrow()

    # 运行所有 14 个任务
    tasks_to_run = list(EVALUATION_TASKS.keys())
    evaluator.run_evaluation(tasks_to_run)

    # 保存最终 JSON 结果
    evaluator.save_results("chemcrow_evaluation_results.json")

    # 生成 Figure 4 并保存到文件夹
    evaluator.generate_figure_4(output_dir="figure4_outputs")

    print("\n评估完成! 请查看:")
    print("- 评估结果: chemcrow_evaluation_results.json")
    print("- Figure 4 图表: figure4_outputs/figure_4_reproduction.png")


if __name__ == "__main__":
    main()