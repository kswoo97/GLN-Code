
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
from GLN_src import generate_paper_prompt, generate_book_prompt, generate_page_prompt


parser = argparse.ArgumentParser(description='GLN Encoding.')
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
        raise TypeError("Data should be located on the folder named './dataset'.")
        
    ## Preparation
        
    target_nodes = np.array(data["target_nodes"])
    edges = data["sampled_edges"]
    initial_text = data["initial_node_attributes"]
    one_hop_nodes = []
    
    for v in target_nodes : ## 1st-hop nodes
        
        one_hop_nodes.append(v)
        one_hop_nodes.extend(edges[v])
        
    one_hop_nodes = np.unique(np.array(one_hop_nodes))
        
    first_layer_outputs = {v : "" for v in one_hop_nodes}
    second_layer_outputs = {v : "" for v in target_nodes}
    
    ## 1st layer encoding
    
    for vidx1 in tqdm(range(one_hop_nodes.shape[0])) : # This enables clean progress-bar printing
        
        tmp_node = one_hop_nodes[vidx1]
        
        if data_name == "arxiv" : 
            cur_prompt = generate_paper_prompt(target_node = tmp_node, raw_text = initial_text, first_layer = None, neighbors = edges, 
                         layer_number = 1, use_attention = True, use_skip_connection = True)
            
        elif data_name == "book" : 
            cur_prompt = generate_book_prompt(target_node = tmp_node, raw_text = initial_text, first_layer = None, neighbors = edges, 
                         layer_number = 1, use_attention = True, use_skip_connection = True)
            
        elif data_name == "page" : 
            cur_prompt = generate_page_prompt(target_node = tmp_node, raw_text = initial_text, first_layer = None, neighbors = edges, 
                         layer_number = 1, use_attention = True, use_skip_connection = True)
            
        else : 
            raise TypeError("This code supports (arxiv, book, page).")
            
            
        ## To avoid error caused by API time limit
        
        cond = True
        
        while cond : 
            
            try : 
        
                if llm_type == "gpt": 
                
                    messages = [{"role": "system", "content": ""},
                        {"role": "user", "content": cur_prompt}]
                    
                    response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=messages,
                                temperature=0.1,
                                max_tokens=512)
                    
                    cur_result = response.choices[0].message.content

                elif llm_type == "claude": 
                    
                    messages = [{"role": "user", "content": cur_prompt}]
                    
                    response = client.messages.create(model="claude-3-haiku-20240307",
                                                 max_tokens=512,
                                                 messages=messages)
                    
                    cur_result = response.content[0].text
                    
                cond = False # Escape while loop
                    
            except : 
                time.sleep(5)
                
        first_layer_outputs[tmp_node] = cur_result
        
    ## 2nd lyaer
    
    for vidx2 in tqdm(range(target_nodes.shape[0])) : # This enables clean progress-bar printing
        
        tmp_node = target_nodes[vidx2]
        
        if data_name == "arxiv" : 
            cur_prompt = generate_paper_prompt(target_node = tmp_node, raw_text = initial_text, first_layer = first_layer_outputs, neighbors = edges, 
                         layer_number = 2, use_attention = True, use_skip_connection = True)
            
        elif data_name == "book" : 
            cur_prompt = generate_book_prompt(target_node = tmp_node, raw_text = initial_text, first_layer = first_layer_outputs, neighbors = edges, 
                         layer_number = 2, use_attention = True, use_skip_connection = True)
            
        elif data_name == "page" : 
            cur_prompt = generate_page_prompt(target_node = tmp_node, raw_text = initial_text, first_layer = first_layer_outputs, neighbors = edges, 
                         layer_number = 2, use_attention = True, use_skip_connection = True)
            
        else : 
            raise TypeError("This code supports (arxiv, book, page).")
            
        cond = True
        
        while cond : 
            
            try : 
        
                if llm_type == "gpt": 
                
                    messages = [{"role": "system", "content": ""},
                        {"role": "user", "content": cur_prompt}]
                    
                    response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=messages,
                                temperature=0.1,
                                max_tokens=512)
                    
                    cur_result = response.choices[0].message.content

                elif llm_type == "claude": 
                    
                    messages = [{"role": "user", "content": cur_prompt}]
                    
                    response = client.messages.create(model="claude-3-haiku-20240307",
                                                 max_tokens=512,
                                                 temperature=0.1,
                                                 messages=messages)
                    
                    cur_result = response.content[0].text
                    
                cond = False # Escape while loop
                    
            except : 
                time.sleep(5)
                
        second_layer_outputs[tmp_node] = cur_result
        
    final_descriptions = dict()
    
    ## Final merging
    
    for v in target_nodes : 
        
        cur_prompt = "- [Detailed description]: <{0}>\n- [General description]: <{1}>\n- [Highly general description]: <{2}>".format(initial_text[v], 
                                                                                                                        first_layer_outputs[v],
                                                                                                                        second_layer_outputs[v])
        
        final_descriptions[v] = cur_prompt
        
    ## Save
    
    os.makedirs("./gen_results", exist_ok=True)
        
    with open("./gen_results/{0}_{1}_representations.pickle".format(data_name, task_name), "wb") as f : 
        pickle.dump(final_descriptions, f)