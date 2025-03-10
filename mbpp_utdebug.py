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


def select_idx(status, num):
    # Find first failing UT
    fail_count = list(range(len(status)))
    return random.sample(fail_count, min(num, len(fail_count)))

def get_unit(pass_units):
    for p in range(len(pass_units)):
        if pass_units[p] != PASS: return p
        
    return -1

def get_all_wrong_units(lookup_dict, pass_units, exec_out):
    out_dict = {'input':[], 'output':[], 'exec': []}
    for p, pass_result in enumerate(pass_units):
        if pass_result != PASS:
            out_dict['input'].append(lookup_dict['input'][p])
            out_dict['output'].append(lookup_dict['output'][p])
            out_dict['exec'].append(exec_out[p])
    return out_dict

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
    exec_globals = {}
    try:
        exec(segment, exec_globals)
    except:
        print('Error extracting input and output')
        return example
    
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
    parser.add_argument('--source', default='finetune_data/sft/all_reduced_test.jsonl', type=str)
    parser.add_argument('--model', default="llama3", type=str)
    parser.add_argument('--max-turns', default=4, type=int)
    parser.add_argument('--units', default=1,  type=int)
    parser.add_argument('--unit-mode', default='joint_oracle', choices=['joint_oracle',  'joint', 'joint_sc', 'train_joint', 'train_joint_sc', 'train_joint_oracle', 'random_joint_sc', 'random_joint_oracle', 'train_random_joint_sc', 'random_joint'])
    parser.add_argument('--filter-vote', type=float, default=0.5)
    parser.add_argument('--num-votes', type=int, default=8)
    parser.add_argument('--backtrack', action='store_true')
    parser.add_argument('--ckpt-dir', default=None, type=str)
    return parser.parse_args()



def build_test(examples, mode):
    global gen_test_lookup
    ids = examples['task_id']
    for idx in ids:
        if idx not in gen_test_lookup:
            gen_test_lookup[idx] = {'input': [], 'output': []}
    print(colored(f'Generating unit tests in mode: {mode} for {len(examples["task_id"])} problems.', 'yellow'))
    unit_sampling_params = SamplingParams(temperature=0.7, top_p=0.9, n=1, max_tokens=8*1024, stop=STOP_WORDS[args.model])
    if not "joint" in mode:
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": adv_input_completion_prompt.format(signature=inst['signature'], description=inst['prompt'], code=inst['code'], entry_point=inst['entry_point'])}], tokenize=False, add_generation_prompt=True) for inst in examples]
    elif "joint" in mode and "random" in mode:
        print('sampling UT inputs at random ...')
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": joint_completion_prompt.format(signature=inst['signature'], description=inst['prompt'], entry_point=inst['entry_point'])}], tokenize=False, add_generation_prompt=True) for inst in examples]
    else:
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": joint_adv_prompt.format(signature=inst['signature'], description=inst['prompt'], code=inst['code'], entry_point=inst['entry_point'])}], tokenize=False, add_generation_prompt=True) for inst in examples]
    remove_tasks = ids
    output_acc = []

    if 'random' in mode:
        filter_vote = round(args.filter_vote*args.num_votes)
    else:
        filter_vote = round(args.filter_vote*args.num_votes)


    tasks_fullfilled = set()
    last_round_tasks_fullfilled = 0
    patience = 3

    for n in range(4*args.units):
        if patience <= 0: 
            print(colored('Fullfilled UT generation for all tasks, stopping now....', 'yellow'))
            break
        
        last_round_tasks_fullfilled = len(tasks_fullfilled)
        unit_input, unit_out = None, None
        if 'train' in args.unit_mode:
            outputs = llm.generate(prompts, unit_sampling_params, lora_request=LoRARequest("finetined_adapter", 1, args.ckpt_dir))
        else:
            outputs = llm.generate(prompts, unit_sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        if mode == 'joint_oracle' or mode == 'train_joint_oracle' or mode == 'random_joint_oracle':
            for response, inst in zip(outputs, examples):
                unit_input = extract_arguments(response, inst['entry_point'])
                if unit_input is None:
                    continue
                decision = input_acceptance(inst['canonical_solution'], [unit_input], inst['entry_point'])
                if decision:
                    unit_out = trusted_exec(inst["canonical_solution"], [unit_input], inst['entry_point'])
                if unit_out is not None and unit_input is not None and (unit_input not in gen_test_lookup[inst["task_id"]]['input']):
                    gen_test_lookup[inst["task_id"]]['input'].append(unit_input)
                    gen_test_lookup[inst["task_id"]]['output'].append(unit_out[0])
                    if inst["task_id"] in remove_tasks: remove_tasks.remove(inst["task_id"])

        if mode == "joint" or mode == 'train_joint' or mode == 'random_joint':
            print(colored(outputs[0], 'yellow'))
            for response, inst in zip(outputs, examples):
                unit_input, unit_str = extract_arguments(response, inst['entry_point'], return_str = True)
                if unit_input is not None:
                    try:
                        unit_output = eval(extract_output(response))
                    except:
                        unit_output = None

                if unit_input is None or unit_output is None or (unit_input not in gen_test_lookup[inst["task_id"]]['input']): continue
                try:
                    decision = input_acceptance(inst['canonical_solution'], [unit_input], inst['entry_point'])
                    if decision:
                        gold_out = trusted_exec(inst["canonical_solution"], [unit_input], inst['entry_point'])
                        output_acc.append(eval_exact_match(gold_out[0], unit_output, use_set=' set(' in inst['test']))
                except: 
                    output_acc.append(False)

                if unit_out is not None and unit_input is not None and (unit_input not in gen_test_lookup[inst["task_id"]]['input']):

                    if len(gen_test_lookup[inst["task_id"]]['input']) >= args.units:
                            tasks_fullfilled.add(inst['task_id'])
                    else:
                        gen_test_lookup[inst["task_id"]]['input'].append(unit_input)
                        gen_test_lookup[inst["task_id"]]['output'].append(unit_output)
                        if inst["task_id"] in remove_tasks: remove_tasks.remove(inst["task_id"])
            print(colored(f'Output Acc: {100*np.sum(output_acc)/len(output_acc)} over {len(output_acc)} instances.', 'yellow'))  

        if mode == 'joint_sc' or mode == 'train_joint_sc' or mode == 'random_joint_sc' or mode == 'train_random_joint_sc':
            unit_lookup = defaultdict(OrderedDict)
            output_tracker = {}
            vote_tracker = {}
            for outt, inst, prompt in zip(outputs, examples, prompts):
                unit_input, unit_str = extract_arguments(outt, inst['entry_point'], return_str = True)
                if unit_input is not None and len(outt) :
                    unit_lookup[inst['task_id']]['input'] = unit_input
                    unit_lookup[inst['task_id']]['prompt'] = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': outt.split('### Output', 1)[0]}]
                    unit_lookup[inst['task_id']]['out_list'] = []


            out_prompts = [tokenizer.apply_chat_template(x['prompt'], tokenize=False, continue_final_message=True) for k,x in unit_lookup.items()]
            sc_sampling_params = SamplingParams(temperature=0.8, top_p=0.9, n=1, max_tokens=2048, stop=STOP_WORDS[args.model])
            unit_examples = examples.filter(lambda x: x['task_id'] in unit_lookup.keys())

            for t in range(args.num_votes):
                if 'train' in args.unit_mode:
                    outputs = llm.generate(out_prompts, sc_sampling_params, lora_request=LoRARequest("finetined_adapter", 1, args.ckpt_dir))
                else:
                    outputs = llm.generate(out_prompts, sc_sampling_params)
                outputs = [output.outputs[0].text for output in outputs]
                for unit_response, task_id in zip(outputs, unit_lookup.keys()):
                    try:
                        unit_output = eval(extract_output(unit_response))
                        unit_lookup[task_id]['out_list'].append(unit_output)
                    except: 
                        pass
            
            for inst in unit_examples:
                # determine the most consistent output
                unit_input = unit_lookup[inst['task_id']]['input']
                unit_outs = unit_lookup[inst['task_id']]['out_list']

                vote = defaultdict(int)
                vote_lookup = defaultdict(list)
                for ps in unit_outs: 
                    vote[str(ps)] += 1
                    vote_lookup[str(ps)].append(ps)

                if len(vote) == 0: 
                    continue

                results_sort = list(zip(vote.keys(), vote.values()))
                results_sort.sort(key=lambda x: x[1], reverse=True)

                vote_tracker[inst['task_id']] = results_sort[0][1]

                if results_sort[0][1] >= filter_vote:
                    unit_output = vote_lookup[results_sort[0][0]][0]
                else: 
                    unit_output = None
                    continue

                unit_lookup[inst['task_id']]['output'] = unit_output
                decision = input_acceptance(task_setup + '\n' + inst['canonical_solution'], [unit_input], inst['entry_point'])
                if decision:
                    gold_out = trusted_exec(task_setup + '\n' + inst["canonical_solution"], [unit_input], inst['entry_point'])[0]
                    unit_lookup[inst['task_id']]['gold_out'] = gold_out
                    try:
                        acc_result = eval_exact_match(gold_out, unit_output, use_set=' set(' in inst['test'])
                    except:
                        acc_result = False
                    if not isinstance(acc_result, bool): continue
                    output_acc.append(acc_result)
                
               

                if unit_output is not None and unit_input is not None:
                    if len(gen_test_lookup[inst["task_id"]]['input']) >= args.units:
                        tasks_fullfilled.add(inst['task_id'])
                    else:
                        try:
                            if (unit_input not in gen_test_lookup[inst["task_id"]]['input']):
                                gen_test_lookup[inst["task_id"]]['input'].append(unit_input)
                                gen_test_lookup[inst["task_id"]]['output'].append(unit_output)
                        except:
                            gen_test_lookup[inst["task_id"]]['input'].append(unit_input)
                            gen_test_lookup[inst["task_id"]]['output'].append(unit_output)
                        if inst["task_id"] in remove_tasks: remove_tasks.remove(inst["task_id"]) 
            
            try:
                print(colored(f'Output Acc: {100*np.sum(output_acc)/len(output_acc)} over {len(output_acc)} instances.', 'yellow'))  
            except:
                import pdb; pdb.set_trace()

        examples = examples.filter(lambda x: x['task_id'] not in tasks_fullfilled)
        if len(tasks_fullfilled) > 0 and len(tasks_fullfilled) == last_round_tasks_fullfilled:
            patience -= 1
        print(f"Notice: {len(remove_tasks)} do not have any unit tests")
    
    for idx in remove_tasks:
        gen_test_lookup.pop(idx)
    print(colored(f'Filtered out {len(remove_tasks)} tasks due to lack of any acceptable unit test', 'yellow'))
    return


### Computing scaling using gold unit tests
task_setup = "\n".join(IMPORT_HELPER["python"])
simple_regenerate_prompt = 'Your output code is incorrect as it fails some unit tests. Please retry generating the correct code solution for the provided problem.'
unit_regenerate_prompt = '''The above code is incorrect and does not pass the testcase.
Input: {wrong_testcase_input}
Output: {wrong_testcase_output}
Expected: {wrong_testcase_expected}

'''



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

    if 'train' in args.unit_mode: assert args.ckpt_dir is not None, "ckpt dir cannot be empty for this mode"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, n=1, max_tokens=8*1024, stop=STOP_WORDS[args.model])
    model_name = HF_MODEL_NAMES[args.model]
    if 'train' in args.unit_mode:
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


    print(colored(f"Found {len(code_src)} instances...", 'yellow'))
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
        
        print(len(gen_test_lookup))
        edited_problems = problems.filter(lambda x: x['task_id'] in edited_tasks)
        if turns_remaining != args.max_turns:
            print(colored(f"Last turned yielded {len(edited_problems['task_id'])} edited codes, deleting those UTs", 'yellow'))
        for temp in edited_problems['task_id']:
            if temp in gen_test_lookup: gen_test_lookup.pop(temp)
        build_test(edited_problems, args.unit_mode)

        print(colored('Done building unit tests for this round ...', 'yellow'))

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

        for task, op in tqdm(zip(still_incorrect, outputs)):
            code = extract_code(op)
            to_backtrack = False

            ### Checking debugging success based on feedback
            if task in debugging_tracker:
                num_debug_units = len(debugging_tracker[task]['input'])
                stat = Value("i", _UNKNOWN)

                result = run_unsafe_execute(entry_points[task], task_setup + "\n" + code,
                                    debugging_tracker[task]['input'],
                                    debugging_tracker[task]['output'],
                                    verbose = True,
                                    use_set=test_lookup[task]['use_set'])  
                if result is None:  
                    print(f"unsafe_execute crashed for: {task}") 
                    # Handle the timeout or crash case  
                    debug_pass = [TIMEOUT]*num_debug_units # Skip to the next iteration as you did before  
                
                # Unpack the result as per your original logic  
                try:  
                    debug_pass, _ = result  
                    # Proceed with your logic using attack_succ, unit_output, func_out  
                except Exception as e: 
                    print(f"unsafe_execute crashed for: {task}") 
                    debug_pass = [TIMEOUT]*num_debug_units

                debug_correct = int(debug_pass.count(PASS) == num_debug_units)
                local_debug_efficacy.append(debug_correct)
                round_tracker[task]['unit_test'].append({'input': str(debugging_tracker[task]['input'][0]), 'output': str(debugging_tracker[task]['output'][0])})
                round_tracker[task]['debug_acc'].append(debug_correct)
                round_tracker[task]['debug_pass'].append(100*debug_pass.count(PASS)/num_debug_units)

                if args.backtrack:
                    ### The unit test in the feedback was not improved
                    ### switching to code from the last round
                    if debug_pass.count(PASS)/num_debug_units < debugging_tracker[task]['pass_rate']:
                        code = problem_lookup[task]['code']
                        to_backtrack = True
                        if task in edited_tasks: edited_tasks.remove(task)

                    else:
                        edited_tasks.add(task)
                else:
                    edited_tasks.add(task)
            else:
                round_tracker[task]['unit_test'].append({'input': None, 'output': None})
                round_tracker[task]['debug_acc'].append(None)
                round_tracker[task]['debug_pass'].append(None)

            
            problem_lookup[task]['code'] = code
            round_tracker[task]['code'].append(code)

            ### Generated Testing
            if task not in gen_test_lookup: 
                no_unit_left += 1
                # new change, go to the next round in attempt to debug
                new_still_incorrect.append(task)
                if task in debugging_tracker: 
                    debugging_tracker.pop(task, None)
                continue


            if not printed_first: 
                print(gen_test_lookup[task])
                printed_first = True

            
            idxs = list(range(len(gen_test_lookup[task]['input'])))
            num_gen_units = len(gen_test_lookup[task]['input'])
            stat = Value("i", _UNKNOWN)

            unit_result = run_unsafe_execute(entry_points[task], task_setup + "\n" + code,
                                    gen_test_lookup[task]['input'],
                                    gen_test_lookup[task]['output'],
                                    verbose = True,
                                    use_set=test_lookup[task]['use_set'])  
            if unit_result is None:  
                print(f"unsafe_execute crashed for: {task}") 
                unit_pass = [TIMEOUT]*num_gen_units
                exec_out = [f"Execution of {entry_points[task]} crashed"]*num_gen_units
                # Handle the timeout or crash case  
            
            # Unpack the result as per your original logic  
            try:  
                unit_pass, exec_out = unit_result  
                # Proceed with your logic using attack_succ, unit_output, func_out  
            except Exception as e: 
                print(f"unsafe_execute crashed for: {task}") 
                # print(f"Error unpacking result for task {inst['task_id']}:\n{traceback.format_exc()}")  
                unit_pass = [TIMEOUT]*num_gen_units
                exec_out = [f"Execution of {entry_points[task]} crashed"]*num_gen_units

            
            unit_correct = int(unit_pass.count(PASS) == num_gen_units)
            if not unit_correct:
                new_still_incorrect.append(task)
                all_incorrect = get_all_wrong_units(gen_test_lookup[task], unit_pass, exec_out) 
                id = idxs[get_unit(unit_pass)]
                unit_inp = gen_test_lookup[task]['input'][id]
                unit_out = gen_test_lookup[task]['output'][id]
                unit_exec = exec_out[id]
                problem_chat_history[task].append({'role': 'assistant', 'content': op})
                round_tracker[task]['if_debugged'].append(False)
                
            
                if not to_backtrack:
                    problem_chat_history[task].append({'role': 'user', 'content': unit_regenerate_prompt.format(wrong_testcase_input=unit_inp, wrong_testcase_expected=unit_out, wrong_testcase_output=unit_exec)}) 
                
                debugging_tracker[task] = {'input': gen_test_lookup[task]['input'], 'output': gen_test_lookup[task]['output'], 'pass_rate': unit_pass.count(PASS)/num_gen_units}
                

            else:
                round_tracker[task]['if_debugged'].append(True)

                if task in debugging_tracker: 
                    debugging_tracker.pop(task, None)
                pass
                

        if len(local_debug_efficacy):
            debug_rate = 100*sum(local_debug_efficacy)/len(local_debug_efficacy)
            debug_success_history.append(debug_rate)
            print(colored(f'Debugging Efficacy: {round(debug_rate, 2)} over {len(local_debug_efficacy)} instances','yellow'))
        else:
            debug_success_history.append(None)

        
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
    

    print(f'For {args.dataset}, Model: {args.model}, Unit Mode: {args.unit_mode}, Units: {args.units}, Backtracking: {args.backtrack}.')

    for i, turn_acc, turn_pass in zip(range(args.max_turns), overall_acc_history, overall_pass_history):
        print(colored(f'\tRound {i}: Acc={round(turn_acc, 2)}, Pass Rate={round(turn_pass, 2)}.', 'green'))
        

    
