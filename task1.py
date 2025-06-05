import os
from dotenv import load_dotenv
import openai
from datetime import datetime
from chemcrow.agents.tools import make_tools
from chemcrow.agents.eval_utils import NoToolsAgent, Evaluator
from chemcrow.agents.chemcrow import ChemCrow
from langchain.chat_models import ChatOpenAI

# === 环境准备 ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === ChemCrow 任务运行器类 ===
class ChemCrowTaskRunner:
    def __init__(self, model_name="gpt-4", temp=0.1):
        self.model_name = model_name
        self.temp = temp
        self.llm = ChatOpenAI(model_name=model_name, temperature=temp)
        self.all_tools = make_tools(llm=self.llm)

        self.mrkl = ChemCrow(tools=self.all_tools, model=model_name, temp=temp)
        self.notools = NoToolsAgent(model=model_name, temp=temp)
        self.evaluator = Evaluator(model=model_name, temp=temp)

    def run_task(self, task_text, verbose=True, save_name=None):
        if verbose:
            print("=== TASK ===\n", task_text)

        result_tools = self.mrkl.run(task_text)
        if verbose:
            print("\n=== Answer with Tools ===")
            print(result_tools)

        result_notools = self.notools.run(task_text)
        if verbose:
            print("\n=== Answer without Tools ===")
            print(result_notools)

        evaluation = self.evaluator.run(task_text, result_tools, result_notools)
        if verbose:
            print("\n=== Teacher Evaluation ===")
            print(evaluation)

        result = {
            "task": task_text,
            "tool_answer": result_tools,
            "notools_answer": result_notools,
            "evaluation": evaluation
        }

        # 保存结果为 .txt 到 figure4_outputs 文件夹
        self.save_txt(result, save_name)
        return result

    def save_txt(self, result, save_name=None):
        os.makedirs("figure4_outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = save_name or f"task_result_{timestamp}"
        filepath = os.path.join("figure4_outputs", f"{fname}.txt")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== TASK ===\n")
            f.write(result["task"].strip() + "\n\n")

            f.write("=== Answer with Tools ===\n")
            f.write(result["tool_answer"].strip() + "\n\n")

            f.write("=== Answer without Tools ===\n")
            f.write(result["notools_answer"].strip() + "\n\n")

            f.write("=== Teacher Evaluation ===\n")
            f.write(result["evaluation"].strip() + "\n")

        print(f"\n✅ Result saved to: {filepath}")
# === Task 1: 合成 Safinamide 并估算成本 ===

task = """
    I need to synthesize a sample of safinamide.
    Please tell me how to synthesize it. Then tell me
    how much will it cost to buy all the reactants I need, if purchasable.
    """

runner = ChemCrowTaskRunner()
runner.run_task(task, save_name="safinamide_task1")
