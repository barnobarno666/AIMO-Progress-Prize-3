import json

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "r") as f:
    nb = json.load(f)

# Re-write the correct exact cell source.
new_prompt_str = (
    "    judge_prompt = (\n"
    "        'You are an expert mathematical reasoning evaluator. Your task is to evaluate '\\n"
    "        'a single reasoning trace produced by a math solver attempting an IMO-level problem.\\n\\n'\\n"
    "        \\n"
    "        'You will evaluate the trace on exactly 3 rubrics, each scored 1-10.\\n'\\n"
    "        'Code execution errors are scored separately and are not your concern.\\n\\n'\\n"
    "        \\n"
    "        '# Rubric 1: Relevance (1-10)\\n'\\n"
    "        'Does each reasoning step address the actual problem at hand? Is the work grounded '\\n"
    "        'in the problem statement and its specific constraints?\\n'\\n"
    "        '  1 = Completely off-topic or addresses a modified/easier problem\\n'\\n"
    "        '  3 = Tangential portions but some steps are relevant\\n'\\n"
    "        '  5 = Mostly relevant but contains unnecessary detours\\n'\\n"
    "        '  7 = Nearly all steps directly address the problem\\n'\\n"
    "        '  10 = Every step is tightly focused on solving the given problem\\n\\n'\\n"
    "        \\n"
    "        '# Rubric 2: Logical Correctness & Rigor (1-10)\\n'\\n"
    "        'Are the mathematical steps valid? Do not be fooled by confident-sounding terminology; '\\n"
    "        'verify the actual logical bridge between steps. PENALIZE EXPONENTIALLY for:\\n'\\n"
    "        '- Incomplete Extremum Proofs: For min/max problems, failing to prove BOTH a lower bound '\\n"
    "        'AND an upper bound (necessity and sufficiency).\\n'\\n"
    "        '- Confident Logical Leaps: Jumping to a numeric conclusion just because a correct mathematical '\\n"
    "        'concept (e.g., \"convex hull\", \"pigeonhole\") was mentioned.\\n'\\n"
    "        '- Hallucinated Math Facts: Asserting false mathematical theorems or fabricating intermediate outputs.\\n'\\n"
    "        '- Drawing wrong conclusions from correct computations.\\n'\\n"
    "        '- Off-by-one errors or misinterpreting edge cases.\\n'\\n"
    "        '  1 = Critical logical errors, fabricated math facts, or entirely missing bound proofs\\n'\\n"
    "        '  2 = Major logical error that fundamentally undermines the solution\\n'\\n"
    "        '  4 = Significant logical flaw (e.g., proved a bound but didn\\'t prove it\\'s optimal)\\n'\\n"
    "        '  6 = Minor logical issue (e.g., sloppy transition) that does not ruin the overall argument\\n'\\n"
    "        '  8 = Sound logic with trivial gaps that are easily filled\\n'\\n"
    "        '  10 = Flawless, rigorous logical reasoning throughout; complete proofs for all claims\\n\\n'\\n"
    "        \\n"
    "        '# Rubric 3: Progression & Coherence (1-10)\\n'\\n"
    "        'Does the solver make continuous forward progress? \\n'\\n"
    "        'IMPORTANT DISTINCTIONS:\\n'\\n"
    "        '- Trying DIFFERENT code/math approaches when one fails is GOOD debugging (score high).\\n'\\n"
    "        '- Repetition means looping: repeating the EXACT SAME reasoning text or identical code '\\n"
    "        'blocks 3+ times without learning from the output.\\n'\\n"
    "        '- Dead ends: Pushing a clearly failed approach for far too long.\\n'\\n"
    "        '  1 = Severe looping (stuck in a repetitive text/code loop for most of the trace)\\n'\\n"
    "        '  3 = Significant repetition of reasoning blocks preventing progress\\n'\\n"
    "        '  5 = Occasional looping or dwelling on failed approaches too long\\n'\\n"
    "        '  7 = Minimal repetition, adapts well to failed intermediate steps\\n'\\n"
    "        '  10 = Highly efficient, linear progress or excellent pivoting upon hitting roadblocks\\n\\n'\\n"
    "        \\n"
    "        '# OUTPUT FORMAT (MANDATORY):\\n'\\n"
    "        'For each rubric, first write a SHORT rationale string, then the integer score.\\n'\\n"
    "        'Evaluate each rubric independently. Do NOT let a score in one rubric influence the others.\\n'\\n"
    "        'Output ONLY a valid JSON object with exactly these 6 keys:\\n'\\n"
    "        '{\\n'\\n"
    "        '  \"relevance_rationale\": \"<1-2 sentences>\", \"relevance\": <int 1-10>,\\n'\\n"
    "        '  \"logical_correctness_rationale\": \"<1-2 sentences identifying any specific logical gaps or unproved bounds>\", \"logical_correctness\": <int 1-10>,\\n'\\n"
    "        '  \"repetition_rationale\": \"<1-2 sentences>\", \"repetition\": <int 1-10>\\n'\\n"
    "        '}\\n\\n'\\n"
    "        'Do NOT include any other text, markdown blocks, or surrounding wrappers. STRICTLY JSON.'\n"
    "    )"
)

# wait actually I can format it so we define lines properly for Jupyter
real_lines = [
    "    judge_prompt = (\n",
    "        'You are an expert mathematical reasoning evaluator. Your task is to evaluate '\n",
    "        'a single reasoning trace produced by a math solver attempting an IMO-level problem.\\n\\n'\n",
    "        \n",
    "        'You will evaluate the trace on exactly 3 rubrics, each scored 1-10.\\n'\n",
    "        'Code execution errors are scored separately and are not your concern.\\n\\n'\n",
    "        \n",
    "        '# Rubric 1: Relevance (1-10)\\n'\n",
    "        'Does each reasoning step address the actual problem at hand? Is the work grounded '\n",
    "        'in the problem statement and its specific constraints?\\n'\n",
    "        '  1 = Completely off-topic or addresses a modified/easier problem\\n'\n",
    "        '  3 = Tangential portions but some steps are relevant\\n'\n",
    "        '  5 = Mostly relevant but contains unnecessary detours\\n'\n",
    "        '  7 = Nearly all steps directly address the problem\\n'\n",
    "        '  10 = Every step is tightly focused on solving the given problem\\n\\n'\n",
    "        \n",
    "        '# Rubric 2: Logical Correctness & Rigor (1-10)\\n'\n",
    "        'Are the mathematical steps valid? Do not be fooled by confident-sounding terminology; '\n",
    "        'verify the actual logical bridge between steps. PENALIZE EXPONENTIALLY for:\\n'\n",
    "        '- Incomplete Extremum Proofs: For min/max problems, failing to prove BOTH a lower bound '\n",
    "        'AND an upper bound (necessity and sufficiency).\\n'\n",
    "        '- Confident Logical Leaps: Jumping to a numeric conclusion just because a correct mathematical '\n",
    "        'concept (e.g., \"convex hull\", \"pigeonhole\") was mentioned.\\n'\n",
    "        '- Hallucinated Math Facts: Asserting false mathematical theorems or fabricating intermediate outputs.\\n'\n",
    "        '- Drawing wrong conclusions from correct computations.\\n'\n",
    "        '- Off-by-one errors or misinterpreting edge cases.\\n'\n",
    "        '  1 = Critical logical errors, fabricated math facts, or entirely missing bound proofs\\n'\n",
    "        '  2 = Major logical error that fundamentally undermines the solution\\n'\n",
    "        '  4 = Significant logical flaw (e.g., proved a bound but didn\\'t prove it\\'s optimal)\\n'\n",
    "        '  6 = Minor logical issue (e.g., sloppy transition) that does not ruin the overall argument\\n'\n",
    "        '  8 = Sound logic with trivial gaps that are easily filled\\n'\n",
    "        '  10 = Flawless, rigorous logical reasoning throughout; complete proofs for all claims\\n\\n'\n",
    "        \n",
    "        '# Rubric 3: Progression & Coherence (1-10)\\n'\n",
    "        'Does the solver make continuous forward progress? \\n'\n",
    "        'IMPORTANT DISTINCTIONS:\\n'\n",
    "        '- Trying DIFFERENT code/math approaches when one fails is GOOD debugging (score high).\\n'\n",
    "        '- Repetition means looping: repeating the EXACT SAME reasoning text or identical code '\n",
    "        'blocks 3+ times without learning from the output.\\n'\n",
    "        '- Dead ends: Pushing a clearly failed approach for far too long.\\n'\n",
    "        '  1 = Severe looping (stuck in a repetitive text/code loop for most of the trace)\\n'\n",
    "        '  3 = Significant repetition of reasoning blocks preventing progress\\n'\n",
    "        '  5 = Occasional looping or dwelling on failed approaches too long\\n'\n",
    "        '  7 = Minimal repetition, adapts well to failed intermediate steps\\n'\n",
    "        '  10 = Highly efficient, linear progress or excellent pivoting upon hitting roadblocks\\n\\n'\n",
    "        \n",
    "        '# OUTPUT FORMAT (MANDATORY):\\n'\n",
    "        'For each rubric, first write a SHORT rationale string, then the integer score.\\n'\n",
    "        'Evaluate each rubric independently. Do NOT let a score in one rubric influence the others.\\n'\n",
    "        'Output ONLY a valid JSON object with exactly these 6 keys:\\n'\n",
    "        '{\\n'\n",
    "        '  \"relevance_rationale\": \"<1-2 sentences>\", \"relevance\": <int 1-10>,\\n'\n",
    "        '  \"logical_correctness_rationale\": \"<1-2 sentences identifying any specific logical gaps or unproved bounds>\", \"logical_correctness\": <int 1-10>,\\n'\n",
    "        '  \"repetition_rationale\": \"<1-2 sentences>\", \"repetition\": <int 1-10>\\n'\n",
    "        '}\\n\\n'\n",
    "        'Do NOT include any other text, markdown blocks, or surrounding wrappers. STRICTLY JSON.'\n",
    "    )\n"
]

import re
cell = nb['cells'][10]
source_str = "".join(cell['source'])

# The corrupted part starts around judge_prompt = (\n
start_idx = source_str.find("judge_prompt = (\n")
end_idx = source_str.find("    served_model_name = 'gpt-oss'")

new_source_str = source_str[:start_idx] + "".join(real_lines) + "\n" + source_str[end_idx:]
lines = [s + "\n" for s in new_source_str.split("\n")]
if lines and lines[-1] == "\n":
    lines.pop()
elif lines:
    lines[-1] = lines[-1].rstrip("\n")

cell['source'] = lines

with open("/home/nhr13/AIMO-Combined/AIMO-Progress-Prize-3/NHR/llm-as-judge.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Prompt properly escaped and replaced.")
