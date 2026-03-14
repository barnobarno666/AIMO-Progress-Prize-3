import json

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. CFG changes
        if "judge_temperature = 0.3" in source:
            source = source.replace(
                "    judge_temperature = 0.3          # Low temperature for consistent evaluation\n    \n    # Output directories (local validation only)",
                "    judge_temperature = 0.3          # Low temperature for consistent evaluation\n    \n    rubric_weights = [0.25, 0.3, 0.25, 0.2]  # [Relevance, Logic, Repetition, Code] sum = 1.0\n    judge_score_threshold = 7.5      # Minimum trace score to be considered\n    \n    # Output directories (local validation only)"
            )
            
        # 2. _judge_single_trace replace total
        source = source.replace(
            "total = (7.5 + 7.5 + 7.5 + code_score) / 4.0",
            "w = self.cfg.rubric_weights\n            total = 7.5 * w[0] + 7.5 * w[1] + 7.5 * w[2] + code_score * w[3]"
        )
        source = source.replace(
            "total = (rel + logic + rep + code_score) / 4.0",
            "w = self.cfg.rubric_weights\n            total = rel * w[0] + logic * w[1] + rep * w[2] + code_score * w[3]"
        )
        
        # 3. _select_answer_with_judge logic
        old_loop = """        for answer, group in answer_groups.items():
            scores = [r['JudgeScore'] for r in group]
            avg_score = sum(scores) / len(scores)
            count = len(group)
            group_weight = avg_score * (math.log(count + 1) / math.log(log_base))
            
            # Per-rubric averages for display
            avg_rel = sum(r.get('relevance', 5) for r in group) / count
            avg_logic = sum(r.get('logical_correctness', 5) for r in group) / count
            avg_code = sum(r.get('code_correctness', 7) for r in group) / count
            avg_rep = sum(r.get('repetition', 5) for r in group) / count"""
            
        new_loop = """        for answer, group in answer_groups.items():
            valid_traces = [r for r in group if r['JudgeScore'] >= self.cfg.judge_score_threshold]
            
            if not valid_traces:
                # If all are rejected, keep the one with the highest score
                best_trace = max(group, key=lambda x: x['JudgeScore'])
                valid_traces = [best_trace]
                
            scores = [r['JudgeScore'] for r in valid_traces]
            avg_score = sum(scores) / len(scores)
            count = len(valid_traces)
            group_weight = avg_score * (math.log(count + 1) / math.log(log_base))
            
            # Per-rubric averages for display
            avg_rel = sum(r.get('relevance', 5) for r in valid_traces) / count
            avg_logic = sum(r.get('logical_correctness', 5) for r in valid_traces) / count
            avg_code = sum(r.get('code_correctness', 7) for r in valid_traces) / count
            avg_rep = sum(r.get('repetition', 5) for r in valid_traces) / count"""
            
        source = source.replace(old_loop, new_loop)
        
        # reconstruct source into lines
        cell["source"] = [s + "\n" for s in source.split("\n")]
        # remove trailing newline on last line
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()
        elif cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

    elif cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        if "This combines the scores to pick a final winner" in source:
            old_md = "This combines the scores to pick a final winner. It groups matching answers, averages all of their scores together, and adds a weight for how frequently it was answered by using the math formula: `Average Score * log8(Vote Count + 1)`. The answer with the highest final weight is chosen."
            new_md = "This combines the scores to pick a final winner. First, traces with an average score below the threshold (`judge_score_threshold`) are thrown out. However, if all traces in an answer group are below the threshold, the highest scoring one is kept so the group is not completely empty. Then, it groups matching answers, calculates the weighted average of the rubrics (`rubric_weights`), and adds a bonus for how frequently it was answered using: `Average Score * log8(Vote Count + 1)`. The answer with the highest final weight is chosen."
            source = source.replace(old_md, new_md)
            
        if "Smart Grouping" in source:
            old_sg = "We take the *average* of their judge scores, and give"
            new_sg = "First, traces scoring below a threshold are filtered out (unless all traces for an answer fail, in which case the best one is kept). We compute a *weighted average* of their rubrics (Relevance, Logic, Repetition, Code), and give"
            source = source.replace(old_sg, new_sg)

        # reconstruct source into lines
        lines = [s + "\n" for s in source.split("\n")]
        # remove trailing newline on last line
        if lines and lines[-1] == "\n":
            lines.pop()
        elif lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell["source"] = lines

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Patched!")
