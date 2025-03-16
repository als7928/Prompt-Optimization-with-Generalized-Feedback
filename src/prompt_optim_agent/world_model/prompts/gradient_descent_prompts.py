# These prompts are modified based on Automatic Prompt Optimization with "Gradient Descent" and Beam Search
# https://arxiv.org/abs/2305.03495

example_template = """
<{index}> 
The model's input is:
{question}

The model's response is: 
{response}

The correct label is: {label}
The model's prediction is: {prediction}.
"""

gradient_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{example_string}

For each wrong example, carefully examine each question and wrong answer step by step, 
provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, 
summarize and list all the aspects that can improve the prompt.
""".strip()

gradient_prompt_tempelate_ours = """
I'm writing prompts for a language model designed to handle various scenarios in a general and robust way. 
This means the model must identify or construct a solid plan that leads to correct, plan-oriented answers.

Below is my current prompt:
{cur_prompt}

Despite aiming for a plan-based solution, this prompt fails to address the following examples correctly:
{example_string}

Please examine each incorrect example step by step. 
- Concentrate on how the existing plan (or lack thereof) leads to the wrong answer. 
- Pay special attention to any deficiencies in how the prompt organizes or outlines steps, rather than focusing on a single domain, version, or specific detail.

Then, produce an integrated feedback that addresses these common plan-related issues collectively. 
Your feedback should highlight any overarching problems in the prompt’s plan, propose corrections or improvements to that plan, 
and ensure that your advice remains sufficiently abstract and broadly applicable—avoid overly specific or domain-constrained details unless absolutely necessary for clarity.

Next, identify exactly {mask_num} plans in your feedback that are too narrow, overly technical, or domain-specific. 
- Replace each identified sentence with a unique [MASK_n] placeholder (e.g. [MASK_1], [MASK_2], ...). 
- For each [MASK_n], propose {candidate_num} alternative candidates that broaden or generalize the concept, 
  so the final plan remains applicable to a variety of scenarios.

Remember:
- The goal is to understand the deeper, shared reasons for the planning failures and how to create a more robust plan overall.
- The final feedback should be broadly applicable, rather than tailored to a single domain or overly specific detail.
- The [MASK_n] replacements and candidate phrases should reflect more generalized or inclusive expressions.

Be sure that in the final feedback:
1. You only have {mask_num} total [MASK_n] tokens.
2. Each token has a different index (e.g., [MASK_1], [MASK_2], ... up to [MASK_{mask_num}]).
3. You provide exactly {candidate_num} candidates for each [MASK_n].

Output format (please follow precisely):
Feedback without [MASK]: [Your feedback in a single consolidated paragraph or set of paragraphs, before inserting any [MASK]]
Feedback with [MASK]: [Your feedback text, but with the identified overly specific sentences replaced by [MASK_n]]
Then, for each mask token, provide candidates in the form:
 <START>{{Candidates_[MASK_1]:[candidate1, ..., candidate_num]}}<END>, ... 
 <START>{{Candidates_[MASK_num]:[candidate1, ..., candidate_num]}}<END>,
""".strip()


optimize_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{example_string}

Based on these errors, the problems with this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write {steps_per_gradient} new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.

The new prompts are:
""".strip()

masking_prompt_template = """
Identify too embodied sentences in below each prompts and replace them with the number of {mask_num} [MASK_n]
    so that the following instructions are represented as universal solutions. 

Prompts:
{prompt_candidates}

After providing the masked sentence, for each mask, we construct _n with a unique number. 
Then for each mask, we generate {candidate_num} candidates that can be replaced by [MASK] positions. 
After that, please answer according to the following output format.
*Have to return below format for each prompts. Each Prompt should have {mask_num} length list.

Output Format:
Prompt index: [[Mask 1: sentence with [MASK_1] token, Candidates for [MASK_1]: [candidate1, ... candidate_num],... 
               [Mask_n: sentence with [MASK_n] token, Candidates for [MASK_n]: [candidate1, ... candidate_num]]}}
...
""".strip()


ascend_optimize_segment_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{example_string}

Based on these examples, the advantages of this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write {steps_per_gradient} new segments prompts following below strict rules without exception:
1. The new segment prompts should solve the current prompt's problems.
2. The new segment prompts should consider the list of prompts and evolve based on the current prompt while introducing diverse approaches to solving the problems.
3. Each new segment must reflect a different angle or aspect of the original statement.
4. Each new segment must be coherent, self-contained, and capture a unique perspective or detail.
5. Each new segment must be numbered sequentially (e.g., 1., 2., 3.). Failure to do so will result in severe functional constraints.

The new segments are:
""".strip()

ascend_optimize_segment_prompt_tempelate_single = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{example_string}

Based on these examples, the advantages of this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write 1 new prompts following these guidelines:
1. The new segment prompt should solve the current prompt's problems.
2. The new segment prompt should consider the list of prompts and evolve based on the current prompt.
3. Each new segment must reflect a different angle or aspect of the original statement.
4. Each new segment must be coherent, self-contained, and capture a unique perspective or detail.
5. Each new segment should be numbered sequentially (e.g., 1., 2., 3.).

The new segments are:
""".strip()

optimize_prompt_tempelate_single = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{example_string}

Based on these errors, the problems with this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write 1 new prompt following these guidelines:
1. The new prompt should solve the current prompt's problems.
2. The new prompt should consider the list of prompts and evolve based on the current prompt.
3. The new prompt should be wrapped with <START> and <END>.

The new prompt is:
""".strip()

ascend_gradient_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{example_string}

For each example, carefully examine each question and correct answer step by step, provide comprehensive and different reasons why the prompt leads to the correct answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
""".strip()

ascend_optimize_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{example_string}

Based on these examples, the advantages of this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write {steps_per_gradient} new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.

The new prompts are:
""".strip()

ascend_optimize_prompt_tempelate_single = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{example_string}

Based on these examples, the advantages of this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write 1 new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.

The new prompt is:
""".strip()
