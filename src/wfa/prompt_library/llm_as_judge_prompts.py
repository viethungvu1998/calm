"""Prompts for the LLMAsJudgeMetrics class."""


data_accuracy_prompt = """
You are a judge evaluating how well the agent's answer matches the final answer.
Question: {question}
Final Answer: {final_answer}
Generated: {generated_answer}
Score the match from 0 to 10 (0 = completely mismatched, 10 = perfectly aligned).

Respond with only the final numeric score (0–10), as a single number (integer or decimal). No words, labels, units, punctuation, or explanation.

# Few-shot examples (model’s response line is only the number):
Question: What is area of Australia?
Final Answer: 7692024 km²
Generated: The area of Australia is 7692024 km².
10

Question: What is area of France?
Final Answer: 671069 km²
Generated: The area of France is 771069 km².
0
"""

explanability_prompt = """
You are a judge evaluating the explainability of the agent's answer.
Question: {question}
Generated: {generated_answer}
Score the explainability from 0 to 10 (0 = not explainable, 10 = extremely clear and understandable).

Respond with only the final numeric score (0–10), as a single number (integer or decimal). No words, labels, units, punctuation, or explanation.

# Few-shot examples (model’s response line is only the number):
Question: What is the capital of France?
Generated: The capital of France is Paris, which is known for its rich history, iconic landmarks like the Eiffel Tower, and important role as the country's political and cultural center.
10

Question: What is the capital of France?
Generated: Paris.
2
"""

jargon_avoidance_prompt = """
You are a judge evaluating if the agent's answer avoids unnecessary jargon.
Question: {question}
Generated: {generated_answer}
Score clarity/jargon avoidance from 0 to 10 (0 = heavily jargony/unclear, 10 = fully clear and jargon-free).

Respond with only the final numeric score (0–10), as a single number (integer or decimal). No words, labels, units, punctuation, or explanation.

# Few-shot examples (model’s response line is only the number):
Question: What is the capital of France?
Generated: The capital of France is Paris.
10

Question: What is the capital of France?
Generated: The administrative epicenter of the French Republic is the metropolis known as Paris, which serves as the sovereign state’s principal governance locus and preeminent urban agglomeration.
0
"""

redundancy_avoidance_prompt = """
You are a judge evaluating if the agent's answer avoids redundancy.
Question: {question}
Generated: {generated_answer}
Score redundancy from 0 to 10 (0 = extremely redundant, 10 = completely concise with no redundancy).

Respond with only the final numeric score (0–10), as a single number (integer or decimal). No words, labels, units, punctuation, or explanation.

# Few-shot examples (model’s response line is only the number):
Question: What is the capital of France?
Generated: The capital of France is Paris.
10

Question: What is the capital of France?
Generated: The capital of France is Paris, and Paris is the capital city of France.
0
"""

citation_quality_prompt = """
You are a judge evaluating the citation quality in the agent's answer.
Question: {question}
Generated: {generated_answer}
Score the citation quality from 0 to 10 (0 = poor or missing citations, 10 = accurate, appropriate, and high-quality citations).

Respond with only the final numeric score (0–10), as a single number (integer or decimal). No words, labels, units, punctuation, or explanation.

# Few-shot examples (model’s response line is only the number):
Question: What is the powerhouse of the cell?
Generated: The powerhouse of the cell is the mitochondrion [Smith et al., 2005].
10

Question: What is the tallest mountain in the world?
Generated: The tallest mountain in the world is Mount Everest.
2
"""
