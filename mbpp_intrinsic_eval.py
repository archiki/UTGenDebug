import os
import re
import time
import openai
import argparse
import traceback
import fire
import random
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset, load_dataset
from termcolor import colored
import sys
import ast
import timeout_decorator
import numpy as np
import json
import subprocess

from collections import defaultdict
import torch
from transformers import AutoTokenizer
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
from multiprocessing import Array, Value

from utils import  HF_MODEL_NAMES, STOP_WORDS, joint_adv_prompt, joint_completion_prompt,  check_attack_success, deepseek_call, llm_call
from utils import adv_input_completion_prompt, extract_arguments, separate_unit_output_prompt, eval_exact_match, extract_output, HF_DATASET


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="llama3", type=str)
    parser.add_argument('--eval-base', action='store_true')
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--ckpt-dir', default='finetune_data/')
    parser.add_argument('--base-ckpt', default=None)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--use-temp', action='store_true')
    parser.add_argument('--num-units', default=1, type=int)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    args.joint = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
     #
    test_lookup = {}
    if args.hard:
        problems = load_dataset('json', data_files=HF_DATASET["mbpp_plus_fix_hard"])['train']
    else:
        problems = load_dataset('json', data_files=HF_DATASET["mbpp_plus_fix"])['train']

    if 'deepseek' in args.model:
        from openai import OpenAI


    if 'gpt' not in args.model and 'deepseek' not in args.model:
        from vllm.lora.request import LoRARequest
        from vllm import LLM, SamplingParams

        if args.use_temp:
            temp = 0.8
            top_p = 0.9
        else:
            temp = 0
            top_p = 1.0

        model_name = HF_MODEL_NAMES[args.model]
        sampling_params = SamplingParams(temperature=temp, top_p=top_p, n=1, max_tokens=2048 + 512, stop=STOP_WORDS[args.model])

        if args.base_ckpt is not None:
            llm = LLM(args.base_ckpt,
                trust_remote_code=True,
                tensor_parallel_size=torch.cuda.device_count(),
                enable_lora=True, 
                max_lora_rank=32, 
                download_dir=".cache/hub/",
                tokenizer=model_name,
                max_model_len=1024*8)
        else:

            llm = LLM(model_name,
                    trust_remote_code=True,
                    tensor_parallel_size=torch.cuda.device_count(),
                    enable_lora=True, 
                    max_lora_rank=32, 
                    download_dir=".cache/hub/",
                    tokenizer=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    else:
        model_name = args.model

    if args.joint:
        random_pre_adv_prompts = [joint_completion_prompt.format(signature=inst['signature'], description=inst['prompt'], entry_point=inst['entry_point']) for inst in problems]

        if not args.random:
            pre_adv_prompts = [joint_adv_prompt.format(signature=inst['signature'], description=inst['prompt'], code=inst['code'], entry_point=inst['entry_point']) for inst in problems]
        else:
            pre_adv_prompts = random_pre_adv_prompts

        list_outputs = []


        if 'gpt' in model_name:
            for i in range(args.num_units):
                outputs = []
                for prompt in tqdm(pre_adv_prompts):
                    outputs.append(llm_call(model_name, prompt))
                list_outputs.append(outputs)
        elif 'deepseek' in model_name:
            for i in range(args.num_units):
                outputs = []
                for prompt in tqdm(pre_adv_prompts):
                    outputs.append(deepseek_call(prompt))
                list_outputs.append(outputs)
        else:
            adv_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in pre_adv_prompts]
            print(colored(adv_prompts[0], 'yellow'))

            for i in range(args.num_units): #4
                if not args.eval_base:
                    outputs = llm.generate(adv_prompts, sampling_params, lora_request=LoRARequest("finetined_adapter", 1, args.ckpt_dir))
                else:
                    outputs = llm.generate(adv_prompts, sampling_params)
                outputs = [output.outputs[0].text for output in outputs]
                list_outputs.append(outputs)

        list_outputs = zip(*list_outputs)
        list_outputs = [list(elem) for elem in list_outputs] 
        attack_tracker = []
        output_tracker = []
        output_acc = []
        overall_rate = []

        
        print(colored(list_outputs[0][0], 'yellow'))

        for inst, outs in zip(problems, list_outputs):
            for r, response in enumerate(outs):
                if 'Output: ```' not in response:
                    response = response.replace('Output: ``', 'Output: ```')
                unit_input, unit_str = extract_arguments(response, inst['entry_point'], return_str=True)
                if unit_input is None:
                    attack_tracker.append(False)
                    output_acc.append(False)
                    overall_rate.append(False)
                    continue
                attack_succ, unit_output, func_out = check_attack_success(inst, inst['code'], unit_input)
                attack_tracker.append(attack_succ)
            
                try:
                    response_output = extract_output(response)
                    try:
                        response_output = eval(response_output)
                        verdict = eval_exact_match(unit_output, response_output, use_set=' set(' in inst['test'])

                    except: 
                        response_output = None
                        verdict = False
                    output_acc.append(verdict)
                except Exception as e:
                    print(e)
                    # pass
                    verdict = False
                    output_acc.append(False)
                if attack_succ : overall_rate.append(verdict)
                else: overall_rate.append(False)

        
        print(colored(f'Attack Rate: {round(100*sum(attack_tracker)/len(attack_tracker), 2)} | Output Acc: {round(100*sum(output_acc)/len(output_acc), 2)}', 'yellow'))
        print(colored(f'Acc ∩ Attack: {round(100*sum(overall_rate)/len(overall_rate), 2)}'))       
       
