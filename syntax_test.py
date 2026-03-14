import json, ast
with open('/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"Checking cell {i}")
        if source.startswith("!") or source.startswith("%"):
            print(f"Skipping cell {i} cause magic")
            continue
        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error in cell {i}: {e}")
            
print("Syntax check finished.")
