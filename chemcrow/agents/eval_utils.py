import openai
import os

class NoToolsAgent:
    def __init__(self, model="gpt-3.5-turbo", temp=0.1, api_key=None):
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temp = temp

    def run(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temp,
            messages=[
                {"role": "system", "content": "You are a helpful chemistry assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()


class Evaluator:
    def __init__(self, model="gpt-3.5-turbo", temp=0.1, api_key=None):
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temp = temp

    def run(self, task, student1_answer, student2_answer):
        eval_prompt = f"""
You are a chemistry professor. Evaluate the two responses below to the same task.

Please follow the exact format below and be concise.

TASK:
{task}

STUDENT 1 (ChemCrow):
{student1_answer}

STUDENT 2 (GPT-3.5 NoTools):
{student2_answer}

Please give a grade (0–10) for each student, list strengths and weaknesses, and explain your justification.

Format:
Student 1's Grade: x
Strengths: ...
Weaknesses: ...
Justification: ...

Student 2's Grade: y
Strengths: ...
Weaknesses: ...
Justification: ...
"""
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temp,
            messages=[
                {"role": "system", "content": "You are a strict but fair chemistry teacher."},
                {"role": "user", "content": eval_prompt}
            ]
        )
        return response.choices[0].message.content.strip()