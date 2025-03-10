import os
import re
import time
import openai
import argparse
import traceback
import fire
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset, load_dataset
from termcolor import colored
from collections import defaultdict, OrderedDict
import sys
import ast
import timeout_decorator
import numpy as np
from sklearn.metrics import f1_score
import concurrent.futures


from vllm.lora.request import LoRARequest
from vllm import LLM, SamplingParams
import json
import subprocess
from transformers import AutoTokenizer
import torch
import random
random.seed(0)
from multiprocessing import Array, Value

from utils import make_signature, extract_code, unsafe_execute, code_generation_prompt, eval_exact_match, IMPORT_HELPER, extract_output, trusted_exec, HF_DATASET, HF_MODEL_NAMES
from utils import llm_call, adv_input_completion_prompt, extract_arguments, input_acceptance, STOP_WORDS, separate_unit_output_prompt, joint_adv_prompt, joint_completion_prompt
sys.set_int_max_str_digits(0)

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


from vllm import LLM, SamplingParams
import torch
from collections import defaultdict


def build_problem_lookup(example):
    global problem_lookup
    entry = {}
    entry['test'] = example['test']
    entry['canonical_solution'] = example['canonical_solution']
    entry['signature'] = example['signature']
    entry['entry_point'] = example['entry_point']
    entry['prompt'] = example['prompt']
    entry['code'] = ''
    problem_lookup[example['task_id']] = entry
    return

def extract_test(example):
    code_str = example['test']
    code_lines = code_str.split('\n')
    end = 0
    for l, line in enumerate(code_lines):
        if 'for i, (inp, exp) in enumerate(zip(inputs, results)):' in line:
            end = l
            break
    segment = '\n'.join(code_lines[:end])
    # print(segment)
    exec_globals = {}
    # print(exec_globals)
    try:
        exec(segment, exec_globals)
    except:
        # import pdb; pdb.set_trace()
        print('Error extracting input and output')
        return example
    
    # example['test_inputs'] = exec_globals['inputs']
    # example['test_outputs'] = exec_globals['results']
    try:
        assert len(exec_globals['inputs']) == len(exec_globals['results']), print('Assertion Error, unequal lengths')
    except:
        return example
    global test_lookup
    use_set = ' set(' in example['test']
    test_lookup[example['task_id']] = {'input': exec_globals['inputs'], 'output': exec_globals['results'], 'use_set': use_set}
    return example

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mbpp_plus_fix_hard', choices=['mbpp_plus_fix', 'mbpp_plus_fix_hard'])
    parser.add_argument('--model', default="llama3", type=str)
    parser.add_argument('--max-turns', default=10, type=int)
    parser.add_argument('--ckpt-dir', default=None, type=str)
    return parser.parse_args()





### Computing scaling using gold unit tests
task_setup = "\n".join(IMPORT_HELPER["python"])
simple_regenerate_prompt = 'Your output code is incorrect. Please fix it and retry generating the correct code solution for the provided problem.'
unit_regenerate_prompt = '''The above code is incorrect and does not pass the testcase.
Input: {wrong_testcase_input}
Output: {wrong_testcase_output}
Expected: {wrong_testcase_expected}

'''

feedback_generation_prompt ='''
# Your Task

## Task
{prompt}

## Code:
```
{code}
```

Based on given task and code, generate feedback that decides whether the code is correct or wrong in the format ```Feedback: <your feedback>```.
Always end your feedback with the line "The above code is correct." if it is correct, otherwise the feedback should end with "The above code is wrong, please fix it."
'''

old_feedback_generation_prompt = '''
# Example 1

## Task
Write a function to find all words which are at least 4 characters long in a string by using regex.

## Code:
```
def find_char_long(text):
    return (re.findall(r"\b\w{{4,}}\b", text))
```

Based on given task and code, first generate line-by-line explanation of the code. Then finally, generate feedback that decides whether the code is correct or wrong.
Always end your feedback with the line "The above code is correct." if it is correct, otherwise the feedback should end with "The above code is wrong, please fix it."

# Response for Example 1

Here is a line-by-line explanation of the code:
`def find_char_long(text):`: This line defines a function named `find_char_long` that takes a single argument, `text`. `text` represents the string whose words are to be extracted.
`return (re.findall(r"\b\w{{4,}}\b", text))`: This line uses the `re.findall()` function to extract all words from the input string that are at least 4 characters long. The regular expression `r"\b\w{{4,}}\b"` matches all words that are at least 4 characters long. The `\b` matches the boundary between a word character and a non-word character. The `\w` matches any word character (a letter, digit, or underscore). The `{{4,}}` matches the preceding element at least 4 times. The `\b` matches the boundary between a word character and a non-word character.

Feedback: The above code returns the following error:
"""
NameError: name `re` is not defined
"""
The above code is wrong, please fix it.

# Example 2

## Task
Write a function to find all words which are at least 4 characters long in a string by using regex.

## Code:
```
import re
def find_char_long(text):
return (re.findall(r"\b\w{{4,}}\b", text))
```

Based on given task and code, first generate line-by-line explanation of the code. Then finally, generate feedback that decides whether the code is correct or wrong.
Always end your feedback with the line "The above code is correct." if it is correct, otherwise the feedback should end with "The above code is wrong, please fix it."

# Response for Example 2

Here is a line-by-line explanation of the code:
`import re`: This line imports the `re` module.
`def find_char_long(text):`: This line defines a function named `find_char_long` that takes a single argument, `text`. `text` represents the string whose words are to be extracted.
`return (re.findall(r"\b\w{{4,}}\b", text))`: This line uses the `re.findall()` function to extract all words from the input string that are at least 4 characters long. The regular expression `r"\b\w{{4,}}\b"` matches all words that are at least 4 characters long. The `\b` matches the boundary between a word character and a non-word character. The `\w` matches any word character (a letter, digit, or underscore). The `{{4,}}` matches the preceding element at least 4 times. The `\b` matches the boundary between a word character and a non-word character.

Feedback: This code functions as intended by the proble description. The above code is correct.

# Example 3

## Task
Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.

## Code:
```
def count_ways(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n == 2:
        return 3
    return count_ways(n-1) + count_ways(n-2)
```

# Response for Example 3



# Your Task

## Task
{prompt}

## Code:
```
{code}
```

Based on given task and code, first generate line-by-line explanation of the code. Then finally, generate feedback that decides whether the code is correct or wrong.
Always end your feedback with the line "The above code is correct." if it is correct, otherwise the feedback should end with "The above code is wrong, please fix it."
'''



def all_unit_regenerate_prompt(input, true_output, gen_output):
    init_prompt = 'Your output code is incorrect as it fails the following unit test(s):\n\n'
    for i, inp, out, exp in zip(range(len(input)), input, true_output, gen_output):
        init_prompt += f"### Unit Test {i+1}\nInput: {inp}\nOutput: {exp}\nExpected: {out}\n\n"
    init_prompt += 'Please retry generating the correct code solution for the provided problem.'
    return init_prompt

def unsafe_execute_wrapper(entry_point, code, unit_inputs, unit_outputs, verbose, use_set):
    try:  
        # Replace this with your actual check_attack_success function  
        return unsafe_execute('mbpp', entry_point, code, unit_inputs, unit_outputs, verbose=verbose, use_set = use_set) 
    except Exception as e:  
        # Optionally log the exception traceback  
        print(f"Exception in unsafe_execute for task {entry_point}")  
        return None

def run_unsafe_execute(entry_point, entry_code, unit_inputs, unit_outputs, verbose, use_set, timeout=60):
    stat = Value("i", _UNKNOWN)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:  
        future = executor.submit(unsafe_execute_wrapper, entry_point, entry_code, unit_inputs, unit_outputs, verbose, use_set)  
        try:  
            result = future.result(timeout=timeout)
            return result  
        except concurrent.futures.TimeoutError:  
            executor.shutdown(wait=False, cancel_futures=True)  
            print(f"unsafe_execute timed out for: {entry_point}")  
            return None  
        except Exception as e:  
            # This should capture exceptions from the wrapper, if any  
            print(f"unsafe_execute crashed for: {entry_point}")  
            return None



if __name__ == "__main__":
    args = parse_args()
    args.backtrack = True

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, n=1, max_tokens=8*1024, stop=STOP_WORDS[args.model])
    model_name = HF_MODEL_NAMES[args.model]
    if args.ckpt_dir is not None:
        llm = LLM(model_name,
              trust_remote_code=True,
              tensor_parallel_size=torch.cuda.device_count(),
              tokenizer=model_name,
              enable_lora=True, 
              max_lora_rank=32,)
    else:
        llm = LLM(model_name,
                    trust_remote_code=True,
                    tensor_parallel_size=torch.cuda.device_count(),
                    tokenizer=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_lookup, gen_test_lookup, problem_lookup = {}, {}, {}

    problems = load_dataset('json', data_files=HF_DATASET[args.dataset], cache_dir=".cache/datasets")['train'] 

    problems = problems.map(extract_test, keep_in_memory=True)
    problems = problems.map(build_problem_lookup, keep_in_memory=True)
    problems = problems.filter(lambda x: x['task_id'] in test_lookup.keys())
    
    problems = problems.map(lambda x: {'gen_input': code_generation_prompt.format(signature = x['signature'], prompt = x['prompt'] + f"\n>>> {x['test_list'][0]}")}, keep_in_memory=True)
    


    total = len(problems['task_id'])
    print(len(problems))



    problems = problems.filter(lambda x: x['task_id'] in test_lookup.keys())


    entry_points  = {}
    for inst in problems:
        entry_points[inst['task_id']] = inst['entry_point']


    ### Unit Test Feedback
    print(colored('Building initial chat history ...', 'yellow'))
    problem_chat_history = OrderedDict()
    starting_outputs = []


    for inst in tqdm(problems):
        problem_chat_history[inst['task_id']] = [{'role': 'user', 'content': inst['gen_input']} , {'role':'assistant', 'content': f"```\n{inst['code']}\n```"}]
        starting_outputs.append(f"```\n{inst['code']}\n```")

    turn_tracker = []

    init_problem_ids = list(problem_chat_history.keys())
    still_incorrect = list(problem_chat_history.keys())
    todo_inputs = []
    round_tracker = {task: defaultdict(list) for task in init_problem_ids}

   
    for task in still_incorrect:
        todo_inputs.append(tokenizer.apply_chat_template(problem_chat_history[task], tokenize=False, add_generation_prompt=True))


    turns_remaining = args.max_turns 

    print(total)

    prev_acc = 0
    prev_count = 0
    prev_pass = 0

    prev_true_outcome = []
    prev_pred_outcome = []

    overall_acc_history, overall_pass_history, sanity_history = [], [], []
    unit_acc_history, unit_f1_history = [], []
    debug_success_history = []
    debugging_tracker = {}

    edited_tasks = set(still_incorrect)

    while turns_remaining > 0 and len(todo_inputs) > 0:

        if turns_remaining == args.max_turns:
            outputs = starting_outputs
        else:
            outputs = llm.generate(todo_inputs, sampling_params)
            outputs = [output.outputs[0].text for output in outputs]
        
        codes = [extract_code(out) for out in outputs]

        if 'code' in problems.column_names:
            problems = problems.remove_columns(['code'])
            problems = problems.filter(lambda x: x['task_id'] in still_incorrect)

        problems = problems.add_column('code', codes)
        

        turn_id = args.max_turns - turns_remaining

        print(colored(f'Evaluating for turn: {turn_id}', 'yellow'))
        new_still_incorrect = []
        overall_acc = []
        overall_pass = []
        turn_correctness = []
        turn_pass = []
        local_debug_efficacy = []


        print(f'FYI: {len(still_incorrect)} instances examined in this run...')
        no_unit_left = 0
        round_correct_lookup = {}

        printed_first = False

        feedback_prompts = [feedback_generation_prompt.format(prompt = x['prompt'], code = x['code']) for x in problems]
        feedback_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": feed}], tokenize=False, add_generation_prompt=True) for feed in feedback_prompts]

        feedbacks = llm.generate(feedback_prompts, sampling_params)
        feedbacks = [output.outputs[0].text for output in feedbacks]

        print(colored(feedbacks[0], 'yellow'))

        for task, op, feedback in tqdm(zip(still_incorrect, outputs, feedbacks)):
            code = extract_code(op)
            to_backtrack = False

            ### Checking debugging success based on feedback

            decide_str = feedback.split('Feedback: ', 1)[-1]
            unit_correct = 'code is correct' in decide_str or 'code is wrong' not in decide_str

            
            problem_lookup[task]['code'] = code
            round_tracker[task]['code'].append(code)

           
           ## determine if UT is correct


            if not unit_correct:
                new_still_incorrect.append(task)
                problem_chat_history[task].append({'role': 'assistant', 'content': op})
                round_tracker[task]['if_debugged'].append(False)
                
                problem_chat_history[task].append({'role': 'user', 'content': feedback}) 
                    
                

       
        print(f'FYI: Next round with {len(new_still_incorrect)} problems.')  

        ### True Testing this round
        acc_this_turn, pass_this_turn = [], []
        true_acc_lookup = {}
        for task in tqdm(init_problem_ids):
            num_eval_units = len(test_lookup[task]['input'])
            stat = Value("i", _UNKNOWN)

            eval_result = run_unsafe_execute(entry_points[task], task_setup + "\n" + problem_lookup[task]['code'],
                                    test_lookup[task]['input'],
                                    test_lookup[task]['output'],
                                    verbose = True,
                                    use_set=test_lookup[task]['use_set'])  
            if eval_result is None:  
                print(f"unsafe_execute crashed for: {task}") 
                # Handle the timeout or crash case  
                eval_pass = [TIMEOUT]*num_eval_units
                exec_out = [f"Execution of {entry_points[task]} crashed"]*num_eval_units
            
            # Unpack the result as per your original logic  
            try:  
                eval_pass, exec_out = eval_result  
                # Proceed with your logic using attack_succ, unit_output, func_out  
            except Exception as e: 
                print(f"unsafe_execute crashed for: {task}") 
                eval_pass = [TIMEOUT]*num_eval_units
                exec_out = [f"Execution of {entry_points[task]} crashed"]*num_eval_units

            
            eval_correct = int(eval_pass.count(PASS) == num_eval_units)
            true_acc_lookup[task] = eval_correct
            eval_pass_rate = eval_pass.count(PASS)/len(eval_pass)
            acc_this_turn.append(eval_correct)
            pass_this_turn.append(eval_pass_rate)
            
            round_tracker[task]['correct'].append(eval_correct)
            round_tracker[task]['pass_rate'].append(100*eval_pass_rate)


        
        overall_acc_history.append(100*np.mean(acc_this_turn))
        overall_pass_history.append(100*np.mean(pass_this_turn))
        print(colored(f'This turn: Acc={overall_acc_history[-1]}, Pass Rate={overall_pass_history[-1]}' ,'yellow'))


    
        turns_remaining += -1
        still_incorrect = new_still_incorrect
        todo_inputs = [tokenizer.apply_chat_template(problem_chat_history[task], tokenize=False, add_generation_prompt=True) for task in still_incorrect]
    
    
    print('')
    print(f'Source: {args.source}')
    print(f'For{args.dataset}, Model: {args.model}, Unit Mode: No UT, self-feedback.')

    for i, turn_acc, turn_pass in zip(range(args.max_turns), overall_acc_history, overall_pass_history):
        print(colored(f'\tRound {i}: Acc={round(turn_acc, 2)}, Pass Rate={round(turn_pass, 2)}.', 'green'))
    
    
    import pdb; pdb.set_trace()

