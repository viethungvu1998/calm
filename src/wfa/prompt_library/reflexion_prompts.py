# Actor prompt template following notebook pattern
actor_prompt_template = """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.

<system>Reflect on the user's original question and the actions taken thus far. Respond using the {function_name} function.</system>"""

# Initial answer instruction
initial_answer_instruction = "Provide a detailed ~250 word answer."

# Revision instruction
revision_instruction = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""
