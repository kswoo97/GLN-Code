## Basic packages
import os
import pickle
import argparse
import numpy as np
import time
from tqdm import tqdm

## API packages

from openai import OpenAI # GPT
import anthropic # Claude

## Source code of GLN
from GLN_src import zero_shot_node_classification, zero_shot_link_prediction

## This is for the arXiv dataset

arxiv_mapping_function = {"arxiv cs ai": "artificial intelligence",
    "arxiv cs ar": "hardware architecture",
    "arxiv cs cc": "computational complexity",
    "arxiv cs ce": "computational engineering, finance, and science",

    "arxiv cs cg": "computational geometry",
    "arxiv cs cl": "computation and language",
    "arxiv cs cr": "cryptography and security",
    "arxiv cs cv": "computer vision and pattern recognition",

    "arxiv cs cy": "computers and society",
    "arxiv cs db": "databases",
    "arxiv cs dc": "distributed, parallel, and cluster computing",
    "arxiv cs dl": "digital libraries",

    "arxiv cs dm": "discrete mathematics",
    "arxiv cs ds": "data structures and algorithms",
    "arxiv cs et": "emerging technologies",
    "arxiv cs fl": "formal languages and automata theory",

    "arxiv cs gl": "general literature",
    "arxiv cs gr": "graphics",
    "arxiv cs gt": "computer science and game theory",
    "arxiv cs hc": "human-computer interaction",

    "arxiv cs ir": "information retrieval",
    "arxiv cs it": "information theory",
    "arxiv cs lg": "machine learning",
    "arxiv cs lo": "logic in computer science",

    "arxiv cs ma": "multiagent systems",
    "arxiv cs mm": "multimedia",
    "arxiv cs ms": "mathematical software",
    "arxiv cs na": "numerical analysis", 
                   
    "arxiv cs ne": "neural and evolutionary computing",
    "arxiv cs ni": "networking and internet architecture",
    "arxiv cs oh": "other computer science",
    "arxiv cs os": "operating systems",

    "arxiv cs pf": "performance",
    "arxiv cs pl": "programming language",
    "arxiv cs ro": "robotics",
    "arxiv cs sc": "symbolic computation",

    "arxiv cs sd": "sound",
    "arxiv cs se": "software engineering",
    "arxiv cs si": "social and information networks",
    "arxiv cs sy": "systems and control"}


parser = argparse.ArgumentParser(description='GLN Downstream.')
parser.add_argument('-data', "--data", type=str, default = "arxiv")
parser.add_argument('-task', "--task", type=str, default = "node")
parser.add_argument('-llm', "--llm", type=str, default = "gpt")
parser.add_argument('-api_key', "--api_key", type=str, default = "your_api_key")
args = parser.parse_args()

data_name = args.data
task_name = args.task
llm_type = args.llm
api_key = args.api_key

if __name__ == "__main__" : 
    
    ## LLM setting
    
    if llm_type == "gpt": 
        client = OpenAI(api_key = api_key)
        
    elif llm_type == "claude": 
        client = anthropic.Anthropic(api_key = api_key)
    
    else : 
        raise TypeError("This code supports GPT and Claude.")
        
    ## Dataset loading

    try : 
        with open("./dataset/{0}_{1}.pickle".format(data_name, task_name), "rb") as f : 
            data = pickle.load(f)
        
    except : 
        print("Data should be located on the folder named './dataset'.")
        
    try : 
        with open("./gen_results/{0}_{1}_representations.pickle".format(data_name, task_name), "rb") as f : 
            representations = pickle.load(f)
        
    except : 
        print("Representations should be located on the folder named './gen_results'.")
        
    if task_name == "node" : 
        labels = data["node_labels"]
        if data_name == "arxiv" : 
            labels = [arxiv_mapping_function[v] for v in labels] # Remapping
        target_nodes = np.array(list(representations.keys()))
        predictions = dict()
        
    elif task_name == "edge" : 
        target_nodes = np.array(data["ground_truth_pairs"][0])
        answer_nodes = np.array(data["ground_truth_pairs"][1])
        predictions = []
        answers = []
        
    for seed, vidx in tqdm(enumerate(target_nodes)) : 
        
        if task_name == "node" :

            cur_prompt = zero_shot_node_classification(representations[vidx], data_type = data_name)
            

        elif task_name == "edge" : 
            
            cur_answer = answer_nodes[seed]
            total_candidates = list(target_nodes[:seed]) + list(target_nodes[seed+1:]) + list(answer_nodes[:seed]) + list(answer_nodes[seed+1:])
            
            np.random.seed(seed) # Fixing seed for random sampling
            tmp_candidates = np.array([cur_answer] + list(np.random.choice(total_candidates, size = 4, replace = False)))
            np.random.shuffle(tmp_candidates)
            target_pos = np.where(tmp_candidates == cur_answer)[0] + 1 # Add 1 due to Python's 0-starting indexing
            answers.append(target_pos)
            candid_text_sets = [representations[vv] for vv in tmp_candidates]
            cur_prompt = zero_shot_link_prediction(representations[vidx], candid_text_sets, data_type = data_name)
            
        cond = True
        
        while cond : 
            
            try : 
        
                if llm_type == "gpt": 
                
                    messages = [{"role": "system", "content": ""},
                        {"role": "user", "content": cur_prompt}]
                    
                    response = client.chat.completions.create(
                                model="gpt-4.1-mini",
                                messages=messages,
                                temperature=0.0,
                                max_tokens=512)
                    
                    cur_result = response.choices[0].message.content

                elif llm_type == "claude": 
                    
                    messages = [{"role": "user", "content": cur_prompt}]
                    
                    response = client.messages.create(model="claude-3-5-haiku-20241022",
                                                 max_tokens=512,
                                                 messages=messages)
                    
                    cur_result = response.content[0].text
                    
                cond = False # Escape while loop
                    
            except : 
                time.sleep(5)
                
        if task_name == "node" : 
            cur_result = f"<{cur_result.strip('<> ') }>"
            predictions[vidx] = cur_result
            
        elif task_name == "edge" : 
            predictions.append(cur_result)
        
    ## Final evaluation
    
    if task_name == "node" : 
        
        ACC = 0.0
        
        for v in predictions : 
            
            cur_pred = predictions[v]
            cur_answer = "<{0}>".format(labels[v])
            
            if cur_pred == cur_answer : 
                ACC += 1.0
                
    elif task_name == "edge" : 
        
        ACC = 0.0
        
        for cur_pred, cur_answer in zip(predictions, answers) : 
            
            try : 
                
                cur_res = int(cur_pred[1:2])
                
                if cur_res == cur_answer : 
                    ACC += 1.0
                    
            except : 
                print("Wrong formatted output. Human evaluation is required.")
                print()
                print("Answer is {0}.".format(cur_answer))
                print()
                print("Prediction is {0}.".format(cur_pred))
                
            
                
    print("Number of correct predictions are: {0}/{1}".format(ACC, target_nodes.shape[0]))
    print("Note. Consider mis-formatted outputs as well.")