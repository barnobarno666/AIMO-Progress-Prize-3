import json

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # fix the incorrectly indented w = self.cfg... and total = ...
        source = source.replace(
        "                    w = self.cfg.rubric_weights\n            total = 7.5 * w[0] + 7.5 * w[1] + 7.5 * w[2] + code_score * w[3]\n",
        "                    w = self.cfg.rubric_weights\n                    total = 7.5 * w[0] + 7.5 * w[1] + 7.5 * w[2] + code_score * w[3]\n"
        )
        
        # reconstruct source into lines (the previous one was slightly flawed)
        lines = [s + "\n" for s in source.split("\n")]
        # remove trailing newline on last line
        if lines and lines[-1] == "\n":
            lines.pop()
        elif lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell["source"] = lines

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Indentation Patched!")
