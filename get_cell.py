import json
with open('/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb') as f:
    nb = json.load(f)
print("".join(nb['cells'][10]['source']))
