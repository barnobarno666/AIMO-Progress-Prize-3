import json

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "r") as f:
    nb = json.load(f)

new_prompt = r'''    judge_prompt = (
        'You are an expert mathematical reasoning evaluator. Your task is to evaluate '
        'a single reasoning trace produced by a math solver attempting an IMO-level problem.\n\n'
        
        'You will evaluate the trace on exactly 3 rubrics, each scored 1-10.\n'
        'Code execution errors are scored separately and are not your concern.\n\n'
        
        '# Rubric 1: Relevance (1-10)\n'
        'Does each reasoning step address the actual problem at hand? Is the work grounded '
        'in the problem statement and its specific constraints?\n'
        '  1 = Completely off-topic or addresses a modified/easier problem\n'
        '  3 = Tangential portions but some steps are relevant\n'
        '  5 = Mostly relevant but contains unnecessary detours\n'
        '  7 = Nearly all steps directly address the problem\n'
        '  10 = Every step is tightly focused on solving the given problem\n\n'
        
        '# Rubric 2: Logical Correctness & Rigor (1-10)\n'
        'Are the mathematical steps valid? Do not be fooled by confident-sounding terminology; '
        'verify the actual logical bridge between steps. PENALIZE EXPONENTIALLY for:\n'
        '- Incomplete Extremum Proofs: For min/max problems, failing to prove BOTH a lower bound '
        'AND an upper bound (necessity and sufficiency).\n'
        '- Confident Logical Leaps: Jumping to a numeric conclusion just because a correct mathematical '
        'concept (e.g., "convex hull", "pigeonhole") was mentioned.\n'
        '- Hallucinated Math Facts: Asserting false mathematical theorems or fabricating intermediate outputs.\n'
        '- Drawing wrong conclusions from correct computations.\n'
        '- Off-by-one errors or misinterpreting edge cases.\n'
        '  1 = Critical logical errors, fabricated math facts, or entirely missing bound proofs\n'
        '  2 = Major logical error that fundamentally undermines the solution\n'
        '  4 = Significant logical flaw (e.g., proved a bound but didn\'t prove it\'s optimal)\n'
        '  6 = Minor logical issue (e.g., sloppy transition) that does not ruin the overall argument\n'
        '  8 = Sound logic with trivial gaps that are easily filled\n'
        '  10 = Flawless, rigorous logical reasoning throughout; complete proofs for all claims\n\n'
        
        '# Rubric 3: Progression & Coherence (1-10)\n'
        'Does the solver make continuous forward progress? \n'
        'IMPORTANT DISTINCTIONS:\n'
        '- Trying DIFFERENT code/math approaches when one fails is GOOD debugging (score high).\n'
        '- Repetition means looping: repeating the EXACT SAME reasoning text or identical code '
        'blocks 3+ times without learning from the output.\n'
        '- Dead ends: Pushing a clearly failed approach for far too long.\n'
        '  1 = Severe looping (stuck in a repetitive text/code loop for most of the trace)\n'
        '  3 = Significant repetition of reasoning blocks preventing progress\n'
        '  5 = Occasional looping or dwelling on failed approaches too long\n'
        '  7 = Minimal repetition, adapts well to failed intermediate steps\n'
        '  10 = Highly efficient, linear progress or excellent pivoting upon hitting roadblocks\n\n'
        
        '# OUTPUT FORMAT (MANDATORY):\n'
        'For each rubric, first write a SHORT rationale string, then the integer score.\n'
        'Evaluate each rubric independently. Do NOT let a score in one rubric influence the others.\n'
        'Output ONLY a valid JSON object with exactly these 6 keys:\n'
        '{\n'
        '  "relevance_rationale": "<1-2 sentences>", "relevance": <int 1-10>,\n'
        '  "logical_correctness_rationale": "<1-2 sentences identifying any specific logical gaps or unproved bounds>", "logical_correctness": <int 1-10>,\n'
        '  "repetition_rationale": "<1-2 sentences>", "repetition": <int 1-10>\n'
        '}\n\n'
        'Do NOT include any other text, markdown blocks, or surrounding wrappers. STRICTLY JSON.'
    )'''

import re

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "judge_prompt = (" in source and "repetition_rationale" in source:
            # We want to replace the `judge_prompt` assignment exactly.
            # Let's find the start and end of it.
            pattern = re.compile(r"    judge_prompt = \(\n.*?'Do NOT include any other text, markdown blocks, or surrounding wrappers\. STRICTLY JSON.'\n    \)", re.DOTALL)
            source = pattern.sub(new_prompt, source)
        
        # update `_format_judge_prompt` task instructions
        if "Evaluate the above reasoning trace on the 3 rubrics" in source:
            old_task = "Evaluate the above reasoning trace on the 3 rubrics (Relevance, Logical Correctness, '\n            'Repetition & Hallucination). Provide rationales and output ONLY the JSON object.'"
            new_task = "Evaluate the above reasoning trace on the 3 rubrics (Relevance, Logical Correctness & Rigor, '\n            'Progression & Coherence). Provide rationales and output ONLY the JSON object.'"
            source = source.replace(old_task, new_task)
            
        lines = [s + "\n" for s in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
        elif lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell["source"] = lines

    elif cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        
        # Markdown updates
        if "2. **Logical Correctness (Coherence):**" in source:
            old_desc = "  2. **Logical Correctness (Coherence):** Do the steps make mathematical sense? Do later equations naturally follow logically from earlier ones without contradicting?\n  3. **Repetition & Hallucination:** Is the solver running around in circles repeating the same text, or making up facts without proof?"
            new_desc = "  2. **Logical Correctness & Rigor:** Are the mathematical steps valid? Does it properly prove both necessity and sufficiency for extremum problems, or are there confident logical leaps/hallucinated math facts?\n  3. **Progression & Coherence:** Does the solver make continuous forward progress, pivot away from dead ends, and adapt properly without severe looping?"
            source = source.replace(old_desc, new_desc)
            
        if "We tell the judge strictly to look at **Relevance**, **Logical Correctness**, and **Repetition**" in source:
            source = source.replace("We tell the judge strictly to look at **Relevance**, **Logical Correctness**, and **Repetition**", "We tell the judge strictly to look at **Relevance**, **Logical Correctness & Rigor**, and **Progression & Coherence**")

        lines = [s + "\n" for s in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
        elif lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell["source"] = lines

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Patch script applied.")
