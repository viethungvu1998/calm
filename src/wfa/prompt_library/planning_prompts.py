planner_prompt = """
You have been given a problem and must formulate a step-by-step plan to solve it.

Consider the complexity of the task and assign an appropriate number of steps.
Each step should be a well-defined task that can be implemented and evaluated.
For each step, specify:

1. A descriptive name for the step
2. A detailed description of what needs to be done
3. Whether the step requires generating and executing code
4. Expected outputs of the step
5. How to evaluate whether the step was successful

Consider a diverse range of appropriate steps such as:
- Data gathering or generation
- Data preprocessing and cleaning
- Analysis and modeling
- Hypothesis testing
- Visualization
- Evaluation and validation

Only allocate the steps that are needed to solve the problem.       

** For question-answering tasks:**
    - Focus on gathering information with the tools available (e.g. web search, database search, etc.) and synthesizing findings
    - Include reasoning steps that explain how conclusions were reached
    - Avoid data analysis and modeling unless explicitly required
    - Avoid file generation unless explicitly required
    - Avoid data visualization unless explicitly required
    - Avoid evaluation and validation unless explicitly required
    - Output should be clear, concise text suitable for console display
"""

detailed_planner_prompt = """
You are contributing to a larger solution.
You have been given one sub-task from this larger effort. Your objective is to:

1. Identify and outline the specific steps needed to complete the sub-task successfully.
2. Provide each step as a numbered list, ensuring each step is a well-defined action that is feasible to implement and evaluate.
3. Offer a short rationale explaining why each step is necessary.
4. Include only as many steps as are needed to accomplish this sub-task effectively; do not add unnecessary complexity.

Please keep your plan concise yet sufficiently detailed so that it can be executed without additional clarification.

** For question-answering tasks:**
    - Focus on gathering information with the tools available (e.g. web search, database search, etc.) and synthesizing findings
    - Include reasoning steps that explain how conclusions were reached
    - Avoid data analysis and modeling unless explicitly required
    - Avoid file generation unless explicitly required
    - Avoid data visualization unless explicitly required
    - Avoid evaluation and validation unless explicitly required
    - Output should be clear, concise text suitable for console display

"""

# reflection_prompt = '''
# You are a critical reviewer being given a series of steps to solve a problem.

# Provide detailed recommendations, including adding missing steps or removing
# superfluous steps. Ensure the proposed effort is appropriate for the problem.

# In the end, decide if the current proposal should be approved or revised.
# Include [APPROVED] in your response if the proposal should be approved with no changes.
# '''

reflection_prompt = """
You are acting as a critical reviewer evaluating a series of steps proposed to solve a specific problem.

Carefully review the proposed steps and provide detailed feedback based on the following criteria:

- **Clarity:** Is each step clearly and specifically described?
- **Completeness:** Are any important steps missing?
- **Relevance:** Are all steps necessary, or are there steps that should be removed because they do not directly contribute to solving the problem?
- **Feasibility:** Is each step realistic and achievable with available resources?
- **Efficiency:** Could the steps be combined or simplified for greater efficiency without sacrificing clarity or completeness?

Provide your recommendations clearly, listing any additional steps that should be included or identifying specific steps to remove or adjust.  

At the end of your feedback, clearly state your decision:

- If the current proposal requires no changes, include "[APPROVED]" at the end of your response.
- If revisions are necessary, summarize your reasoning clearly and briefly describe the main revisions needed.
"""

formalize_prompt = """
Now that the step-by-step plan is finalized, format it into a series of steps in the form of a JSON array with objects having the following structure:
[
    {{
        "id": "unique_identifier",
        "name": "Step name",
        "description": "Detailed description of the step",
        "requires_code": true/false,
        "expected_outputs": ["Output 1", "Output 2", ...],
        "success_criteria": ["Criterion 1", "Criterion 2", ...]
    }},
    ...
]
"""

# planner_prompt = """
# You have been given a problem and must formulate a step-by-step plan to solve it.

# Consider the complexity of the task and assign an appropriate number of steps.
# Each step should be a well-defined task that can be implemented and evaluated.
# For each step, specify:

# 1. A descriptive name for the step
# 2. A detailed description of what needs to be done
# 3. Whether the step requires generating and executing code
# 4. Expected outputs of the step
# 5. How to evaluate whether the step was successful

# Consider a diverse range of appropriate steps such as:
# - Check available data
# - Extract location from the user's query
# - Extract date range from the user's query
# - Check available models
# - Check available tools
# - Data gathering or generation
# - Data preprocessing and cleaning
# - Analysis and modeling
# - Hypothesis testing
# - Reasoning and explanation about the results
# - Synthesize findings into coherent answer
# - Present final answer with supporting reasoning

# Only allocate the steps that are needed to solve the problem.        

# ** For question-answering tasks:**
# - Focus on gathering information with the tools available (e.g. web search, database search, etc.) and synthesizing findings
# - Final step should be "Final Answer" that outputs to console
# - Include reasoning steps that explain how conclusions were reached
# - Avoid data analysis and modeling unless explicitly required
# - Avoid file generation unless explicitly required
# - Avoid data visualization unless explicitly required
# - Output should be clear, concise text suitable for console display
# """

# detailed_planner_prompt = """
# You are contributing to a larger solution.
# You have been given one sub-task from this larger effort. Your objective is to:

# 1. Identify and outline the specific steps needed to complete the sub-task successfully.
# 2. Provide each step as a numbered list, ensuring each step is a well-defined action that is feasible to implement and evaluate.
# 3. Offer a short rationale explaining why each step is necessary.
# 4. Include only as many steps as are needed to accomplish this sub-task effectively; do not add unnecessary complexity.

# Please keep your plan concise yet sufficiently detailed so that it can be executed without additional clarification.
# """

# # reflection_prompt = '''
# # You are a critical reviewer being given a series of steps to solve a problem.

# # Provide detailed recommendations, including adding missing steps or removing
# # superfluous steps. Ensure the proposed effort is appropriate for the problem.

# # In the end, decide if the current proposal should be approved or revised.
# # Include [APPROVED] in your response if the proposal should be approved with no changes.
# # '''

# reflection_prompt = """
# You are acting as a critical reviewer evaluating a series of steps proposed to solve a specific problem.

# Carefully review the proposed steps and provide detailed feedback based on the following criteria:

# - **Clarity:** Is each step clearly and specifically described?
# - **Completeness:** Are any important steps missing?
# - **Relevance:** Are all steps necessary, or are there steps that should be removed because they do not directly contribute to solving the problem?
# - **Feasibility:** Is each step realistic and achievable with available resources?
# - **Efficiency:** Could the steps be combined or simplified for greater efficiency without sacrificing clarity or completeness?
# - **Output Format:** For question-answering tasks, does the plan lead to a clear console output with reasoning? Avoid unnecessary file generation.

# Provide your recommendations clearly, listing any additional steps that should be included or identifying specific steps to remove or adjust.  

# At the end of your feedback, clearly state your decision:

# - If the current proposal requires no changes, include "[APPROVED]" at the end of your response.
# - If revisions are necessary, summarize your reasoning clearly and briefly describe the main revisions needed.
# """

# formalize_prompt = """
# Now that the step-by-step plan is finalized, format it into a series of steps in the form of a JSON array with objects having the following structure:
# [
#     {
#         "id": "unique_identifier",
#         "name": "Step name",
#         "description": "Detailed description of the step",
#         "requires_code": true/false,
#         "expected_outputs": [
#             {"type": "text|data|analysis", "content": "Description of output"}
#         ],
#         "success_criteria": [
#             {"criteria": "Success criterion description", "weightage": 1-10}
#         ]
#     },
#     ...
# ]

# **For question-answering tasks:**
# - The final step should typically output console text with reasoning
# - Use expected_outputs type "text" for textual answers
# - Avoid file paths in expected_outputs - use content descriptions instead
# - Success criteria should validate answer quality and reasoning completeness
# """
