import json

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "valid_traces = [r for r in group if r['JudgeScore'] >= self.cfg.judge_score_threshold]" in source:
            print("Found loop replacement!")
