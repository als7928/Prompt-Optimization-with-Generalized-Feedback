# The code is modified based on Automatic Prompt Optimization with "Gradient Descent" and Beam Search
# https://arxiv.org/abs/2305.03495

from .prompts.log_prompt_templates import *
from .prompts.gradient_descent_prompts import example_template, optimize_prompt_tempelate_single, \
    optimize_prompt_tempelate, gradient_prompt_tempelate, gradient_prompt_tempelate_ours, masking_prompt_template,\
    ascend_gradient_prompt_tempelate, ascend_optimize_prompt_tempelate, ascend_optimize_prompt_tempelate_single
from ..utils import *
import re
import numpy as np
import networkx as nx
import time

class GradientDescent():
    def __init__(self, 
                 task, 
                 base_model, 
                 optim_model,
                 print_log = True,
                 logger = None,
                 num_new_prompts = 1,
                 mode = 'mode',
                 segments_combine_mode = 'mode',
                 **kwargs): 
        
        assert mode in ['baseline', 'proposed'], f'mode must be one of baseline or proposed, but got {mode}.'
        assert segments_combine_mode in ['random', 'proposed'], f"segments_combine_mode must be one of random or proposed, but got {segments_combine_mode}."

        self.mode = mode
        self.segments_combine_mode = segments_combine_mode
        self.task = task
        self.base_model = base_model
        self.optim_model = optim_model
        

        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.num_new_prompts = num_new_prompts
        # self.num_segments_to_combine = num_segments_to_combine
        # self.num_combinations_to_select = num_combinations_to_select
        
        self.use_correct_examples = False
        
        if self.mode in ['proposed']:
            self.gradient_prompt_tempelate_ours = gradient_prompt_tempelate_ours
            # self.centrality_algorithm = get_centrality_algo(centrality_algo)
        self.gradient_prompt_tempelate = gradient_prompt_tempelate
            
        self.optimize_prompt_tempelate = optimize_prompt_tempelate_single \
            if num_new_prompts == 1 else optimize_prompt_tempelate
        self.ascend_optimize_prompt_tempelate = ascend_optimize_prompt_tempelate_single \
            if num_new_prompts == 1 else ascend_optimize_prompt_tempelate
    
        self.ascend_gradient_prompt_template = ascend_gradient_prompt_tempelate
        self.example_template = example_template
        
        self._build_forward_prompts_func = task.build_forward_prompts_completion
        self._batch_forward_func = self.base_model.batch_forward_func
        
        # self.mask_num = 2
        # self.candidate_num = 25
        self.mask_num = 2
        self.candidate_num = 50
        
        self.error_flag = False
        
    def forward(self, batch, cur_prompt): 
        batch_size = len(batch['question'])
        batch_prompts =self._build_forward_prompts_func(batch['question'], cur_prompt) #prompts+questions+ans_format
        responses = self._batch_forward_func(batch_prompts)
        
        for p, r in zip(batch_prompts, responses):
            self.logger.info(f"Input:\n{p}")
            self.logger.info(f"Output:\n{r}")
            
        preds = self.task.batch_clean_responses(responses)
        labels = self.task.clean_labels(batch['answer'])
        correct = self.task.cal_correct(preds, labels)
        
        batch_logs = []
        for i in range(batch_size):
            batch_logs.append({
                'cur_prompt': cur_prompt,
                'question': batch['question'][i],
                'model_input': batch_prompts[i],
                'gt_answer':batch['answer'][i],
                'model_response': responses[i],
                'label':labels[i],
                'pred':preds[i],
                })
        
        forward_output = {
            'cur_prompt': cur_prompt,
            'correct':correct, 
            'examples':batch_logs,
            'acc':np.mean(correct)
            }
        
        if self.print_log:
            log_str = forward_log_tempelate.format(
                cur_prompt=cur_prompt,
                batch_prompts=batch_prompts,
                responses=responses,
                preds=preds,
                labels=labels,
                correct=forward_output['correct'],
                acc=forward_output['acc'])

            self.logger.info(log_str)
        return forward_output
    
    def _clean_self_eval_score(self, response):
        return re.findall(r'\d+', response)[-1]
    
    def _split_error_and_correct_examples(self, forward_output): 
        error_examples = []
        correct_examples = []
        count = 0
        for i, example in enumerate(forward_output['examples']):
            if forward_output['correct'][i]==0:   #cur_prompt, question, model_input, gt_answer, model_respone, label, pred
                count += 1
                error_examples.append(self.example_template.format(
                    index=count, 
                    question=example['model_input'],
                    label=example['label'], 
                    response=example['model_response'],
                    prediction=example['pred']))
            elif forward_output['correct'][i]==1:
                count += 1
                correct_examples.append(self.example_template.format(
                    index=count, 
                    question=example['model_input'],
                    label=example['label'], 
                    response=example['model_response'],
                    prediction=example['pred']))
            else:
                raise ValueError(f'_get_error_examples: invalid correct number {i} {forward_output}.')
        error_string = ''.join(error_examples)
        correct_string = ''.join(correct_examples)
        return error_string, correct_string

    def _build_prompt_trajectory_str(self, prompts):
        prompt_path_str = ""
        prompt_path_str_tempelate = "({index}) {prompt}\n"
        for i, prompt in enumerate(prompts):
            prompt_path_str += prompt_path_str_tempelate.format(index=i,prompt=prompt)
        return prompt_path_str
        
        
    ##############################################################################
    #                                                                            #
    #                                                                            #
    #                                                                            #
    ##############################################################################
    

    def extract_feedback_and_candidates(self, text):
        # 1. Feedback without [MASK]
        feedback_without_mask_match = re.search(
            r'Feedback without\s*\[MASK\]:.*?\s*(.*?)(?=\s*Feedback.*\s*)',
            text,
            re.DOTALL
        )
        if feedback_without_mask_match:
            feedback_without_mask = feedback_without_mask_match.group(1).strip()
        else:
            feedback_without_mask_match_alt = re.search(
                r'without.{0,8}:(.*?)(?=\s*Feedback.*\s*)',
                text,
                re.DOTALL
            )
            feedback_without_mask = feedback_without_mask_match_alt.group(1).strip() if feedback_without_mask_match_alt else None

        # 2. Feedback with [MASK]
        feedback_with_mask_match = re.search(
            r'Feedback with\s*\[MASK\]:.*?\s*(.*?)(?=\s*<START>.*\s*)',
            text,
            re.DOTALL
        )
        if feedback_with_mask_match:
            feedback_with_mask = feedback_with_mask_match.group(1).strip()
        else:
            feedback_with_mask_match_alt = re.search(
                r'with.{0,8}:.{0,1}(.*?)(?=\s*<START>.*\s*)',
                text,
                re.DOTALL
            )
            feedback_with_mask = feedback_with_mask_match_alt.group(1).strip() if feedback_with_mask_match_alt else None

        if feedback_without_mask:
            feedback_without_mask = re.sub(r'\s*\n+\[.*$', '', feedback_without_mask).strip()
        if feedback_with_mask:
            feedback_with_mask = re.sub(r'\s*\n+\[.*$', '', feedback_with_mask).strip()

        candidates_matches = re.findall(
            r'<START>.{0,3}Candidates.{0,8}(MASK_\d+).{0,8}[\[\{](.*?)[\]\}].{0,3}<END>',
            text,
            re.DOTALL
        )
        if not candidates_matches:  
            candidates_matches = re.findall(
                r'START.{0,14}(MASK_\d+).{0,8}[\[\{](.*?)[\]\}].{0,3}END',
                text,
                re.DOTALL
            )

        candidates_dict = {
            mask: [candidate.strip().strip('"') for candidate in re.split(r',|\n', candidates)]
            for mask, candidates in candidates_matches
        }

        # candidates_dict = {
        #     mask: [candidate.strip().strip('"') for candidate in re.split(r',|\n', candidates) if candidate.strip()]
        #     for mask, candidates in candidates_matches
        # }
        
        filtered_candidates_dict = {
            mask: [candidate.strip() for candidate in candidates if candidate.strip()]
            for mask, candidates in candidates_dict.items()
        }
        
        return feedback_without_mask, feedback_with_mask, filtered_candidates_dict

    
    def fill_masks(self, feedback_with_mask, candidates_dict):
        masks = [f"MASK_{i}" for i in range(1, len(candidates_dict) + 1)] #MASK_1, MASK_2...

        replacements = [candidates_dict[mask] for mask in masks]
        from itertools import product
        all_combinations = list(product(*replacements)) 
        
        filled_sentences = []
        for combination in all_combinations:
            filled_sentence = feedback_with_mask
            for mask, replacement in zip(masks, combination): 
                filled_sentence = filled_sentence.replace(f"[{mask}]", replacement)  # Replace one instance at a time
            filled_sentences.append(filled_sentence)
        
        return filled_sentences

    def cal_gradient_correct(self, cur_prompt, example_string, gradient_prompt_tempelate):
        gradient_prompt = gradient_prompt_tempelate.format(cur_prompt=cur_prompt, 
                                                                example_string=example_string)
        gradient = self.optim_model.generate(gradient_prompt)
        
        if self.print_log:
            log_str = gradient_log_tempelate.format(gradient_prompt=gradient_prompt,
                                                    gradient=gradient)

            self.logger.info(log_str)

        return gradient
    
    ########################################################################################
    def cal_gradient_ours(self, cur_prompt, example_string, gradient_prompt_tempelate_ours):    
        retry_count = 0  
        max_retries = 3
        backoff_time = 1
        while retry_count < max_retries:
            try:
                gradient_prompt = gradient_prompt_tempelate_ours.format(
                    cur_prompt=cur_prompt, 
                    example_string=example_string,
                    mask_num=self.mask_num,
                    candidate_num=self.candidate_num
                )
                
                gradient = self.optim_model.generate(gradient_prompt)
                feedback_without_mask, feedback_with_mask, candidates_dict = self.extract_feedback_and_candidates(gradient)
                gradient_candidates = self.fill_masks(feedback_with_mask, candidates_dict)
                
                self.logger.info(f"gradient_candidates : {gradient_candidates}")
                if gradient_candidates in [None, False, "", []]:
                    raise ValueError("Invalid gradient: feedback_with_mask or candidates_dict is None")
                
                self.logger.info(f"feedback_without_mask : {feedback_without_mask}")
                self.logger.info(f"feedback_with_mask : {feedback_with_mask}")
                self.logger.info(f"candidates_dict : {candidates_dict}")
                
                if self.segments_combine_mode == "random":
                    gradient = self.select_general_prompt(gradient_candidates, self.optim_model.get_text_embeddings)
                elif self.segments_combine_mode == "proposed":
                    gradient = self.select_general_prompt(gradient_candidates, self.optim_model.get_text_embeddings)

                if self.print_log:
                    log_str = gradient_log_tempelate.format(
                        gradient_prompt=gradient_prompt,
                        gradient=gradient
                    )
                    self.logger.info(log_str)

                return gradient
                        
            except Exception as e:
                self.logger.error(f"Error occurred: {e}")
                self.logger.warning(f"Retrying with the gradient prompt... (Attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
                # self.logger.info(e, f'Sleeping {backoff_time} seconds...')
                print(f"Error: {e}. Sleeping {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
                
        self.logger.error("Max retrries exceeded. Using fallback gradient prompt.")
        gradient_prompt = self.gradient_prompt_tempelate.format(
            cur_prompt=cur_prompt, 
            example_string=example_string
        )
        gradient = self.optim_model.generate(gradient_prompt)

        if self.print_log:
            log_str = gradient_log_tempelate.format(
                gradient_prompt=gradient_prompt,
                gradient=gradient
            )
            self.logger.info(log_str)
        return gradient
            
    def cal_gradient(self, cur_prompt, example_string, gradient_prompt_tempelate):
        gradient_prompt = gradient_prompt_tempelate.format(cur_prompt=cur_prompt, 
                                                    example_string=example_string)
        gradient = self.optim_model.generate(gradient_prompt)

        if self.print_log:
            log_str = gradient_log_tempelate.format(gradient_prompt=gradient_prompt,
                                                    gradient=gradient)
            self.logger.info(log_str)
            
        return gradient

    def _clean_optim_response(self, optim_response):
        pattern = r'<START>(.*?)<END>'
        matches = re.findall(pattern=pattern, string=optim_response, flags=re.DOTALL)
        for i, m in enumerate(matches):
            matches[i] = m.strip()
            
        return matches

    def optimize(self, cur_prompt, example_string, gradient, trajectory_prompts, 
                 steps_per_gradient, optimize_prompt_tempelate, max_length=0):
        optimize_prompt = optimize_prompt_tempelate.format(
            cur_prompt=cur_prompt, 
            example_string=example_string, 
            gradient=gradient, 
            max_length = max_length,
            trajectory_prompts=trajectory_prompts,
            steps_per_gradient=steps_per_gradient)
        
        response = self.optim_model.generate(optimize_prompt) #generate = self.chat_completion
        
        optimized_prompt = self._clean_optim_response(response)
        if self.print_log:
            log_str = optimize_log_tempelate.format(optimize_prompt=optimize_prompt,
                                                    response=response,
                                                    optimized_prompt=optimized_prompt)
            self.logger.info(log_str)

        return optimized_prompt
        
    def _all_correct_exception(self, cur_prompt, forward_output, correct_string, helper_data):
        gradient = self.cal_gradient_correct(
            cur_prompt=cur_prompt, 
            example_string=correct_string,
            gradient_prompt_tempelate=self.ascend_gradient_prompt_template)
        
        trajectory_prompts = helper_data['trajectory_prompts']
        trajectory_prompts = self._build_prompt_trajectory_str(trajectory_prompts)
        
        gradient_descent_output = forward_output
        optimized_prompts = self.optimize(
            cur_prompt=cur_prompt, 
            example_string=correct_string, 
            gradient=gradient, 
            trajectory_prompts=trajectory_prompts,
            steps_per_gradient=self.num_new_prompts,
            optimize_prompt_tempelate=self.ascend_optimize_prompt_tempelate)
        
        gradient_descent_output['example_string'] = correct_string
        gradient_descent_output['gradient'] = gradient
        gradient_descent_output['optimized_prompts'] = optimized_prompts
        return gradient_descent_output
    
    def gradient_descent_step(self, cur_prompt, batch, helper_data):
        self.logger.info(f'cur_prompt: {cur_prompt}')

        forward_output = self.forward(batch=batch, cur_prompt=cur_prompt)
        error_string, correct_string = self._split_error_and_correct_examples(forward_output=forward_output) 
        
        if forward_output['acc']==1:
            gradient_descent_output = self._all_correct_exception(
                cur_prompt=cur_prompt, 
                forward_output=forward_output,
                correct_string=correct_string,
                helper_data=helper_data)
            return gradient_descent_output
        
        if self.mode == 'proposed':
            gradient = self.cal_gradient_ours( #create error feedback 
                cur_prompt=cur_prompt, 
                example_string=error_string,
                gradient_prompt_tempelate_ours=self.gradient_prompt_tempelate_ours)
                
        elif self.mode == 'baseline':
            gradient = self.cal_gradient( 
                cur_prompt=cur_prompt, 
                example_string=error_string,
                gradient_prompt_tempelate=self.gradient_prompt_tempelate)
                  
        trajectory_prompts = helper_data['trajectory_prompts']
        trajectory_prompts = self._build_prompt_trajectory_str(trajectory_prompts) 
        
        optimized_prompts = self.optimize( 
        cur_prompt=cur_prompt, 
        example_string=error_string, 
        gradient=gradient, 
        trajectory_prompts=trajectory_prompts,
        steps_per_gradient=self.num_new_prompts,
        optimize_prompt_tempelate=self.optimize_prompt_tempelate) 
                
        gradient_descent_output = forward_output
        gradient_descent_output['example_string'] = error_string
        gradient_descent_output['gradient'] = gradient
        gradient_descent_output['optimized_prompts'] = optimized_prompts
        return gradient_descent_output
    
    def __call__(self, batch, cur_prompt, helper_data=None):
        gradient_descent_output = self.gradient_descent_step(cur_prompt=cur_prompt, batch=batch, helper_data=helper_data)
        return gradient_descent_output
        
    def select_general_prompt(self, candidates, get_text_embeddings, print_log=True):
            """
            Generate all possible combinations of segments and select the desired number of combinations.
            
            Args:
                candidates (list): List of segmented prompts (strings). 
                num_combinations_to_select (int): Number of combinations to randomly select.

            Returns:
                list: A list of randomly selected combinations, where each combination is a list of segments.
                list: A list of combined texts for LLM input.
            """
            num_segments_to_combine = 1

            # Transform the segmented prompts into embeddings
            text_embs = np.array(get_text_embeddings(candidates))
            
            importance_score = (text_embs @ text_embs.T).sum(1)
            importance = dict({candidates[i]: j for i, j in enumerate(importance_score.argsort()[::-1])})

            selected = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:num_segments_to_combine]

            imp_texts = [i[0] for i in selected]

            selected_combination = [
                imp_texts[:]  
            ]
            
            combined_texts_for_llm = [
                "\n\n".join(f"{idx + 1}. {segment}" for idx, segment in enumerate(combination))
                for combination in selected_combination
            ]
            if print_log: 
                self.logger.info(f'Segmented prompts:')
                log_message = "\n".join([f" -> [importance: {score:.4f}] > {txt}" for txt, score in importance.items()])
                self.logger.info(log_message) 
                self.logger.info(f'\nSelected combination: {selected_combination}')
                
            return selected_combination, combined_texts_for_llm
