import numpy as np
import pickle
import pandas as pd
import torch
import copy

def generate_paper_prompt(target_node = 0, raw_text = None, first_layer = None, neighbors = None, 
                         layer_number = 1, use_attention = True, use_skip_connection = True) : 
    
    task_prompt1 = "[Task] Refine the target paper's description in [Point 1] by incorporating the papers from [Point 2].\n"
    
    if layer_number == 1 : # First layer
        
        point1 = "[Point 1] Target paper: <{0}>.\n".format(raw_text[target_node])
        point2 = "[Point 2] Papers that cite or are cited by the target paper: [\n"

        for idx, v in enumerate(neighbors[target_node]) : 
            
            point2 += "- Paper {0}: <{1}>\n".format(idx+1, raw_text[v])
    
    else : 
        
        if use_skip_connection : 
            
            point1 = "[Point 1] Target paper: [Detailed description: <{0}>\nGeneral description: <{1}>]\n".format(raw_text[target_node], first_layer[target_node])
            point2 = "[Point 2] Papers that cite or are cited by the target paper: [\n"
            
            for idx, v in enumerate(neighbors[target_node]) : 
            
                point2 += "- Paper {0}: [Detailed description: <{1}>\nGeneral description: <{2}>]\n".format(idx+1, raw_text[v], first_layer[v])
                
        else : 
            
            point1 = "[Point 1] Target paper: <{0}>.\n".format(first_layer[target_node])
            point2 = "[Point 2] Papers that cite or are cited by the target paper: [\n"

            for idx, v in enumerate(neighbors[target_node]) : 

                point2 += "- Paper {0}: <{1}>\n".format(idx+1, first_layer[v])
    
    point2 += "]\n"
        
    task_prompt2 = "[Instructions] Your summary must:\n"
    if use_attention : 
        attention_prompt = "- In summarizing the papers in [Point 2], give more emphasis to those more relevant to the target paper in [Point 1].\n"
    else : 
        attention_prompt = ""
    
    other_instruction1 = "- Return 2 paragraphs at most.\n"
    other_instruction2 = "- Do not introduce external facts; only use the given data.\n"
    other_instruction3 = "- Do not mention specific papers by name; focus on content.\n"
    other_instruction4 = "- Output only the refined description (no extra commentary).\n"
    
    total_prompt = task_prompt1 + point1 + point2 + task_prompt2 + attention_prompt + other_instruction1 + other_instruction2 + other_instruction3 + other_instruction4
    
    return total_prompt

def generate_book_prompt(target_node = 0, raw_text = None, first_layer = None, neighbors = None, 
                         layer_number = 1, use_attention = True, use_skip_connection = True) : 
    
    task_prompt1 = "[Task] Refine the target book's description in [Point 1] by incorporating the books from [Point 2].\n"
    
    if layer_number == 1: # First layer
        
        point1 = "[Point 1] Target book: <{0}>.\n".format(raw_text[target_node])
        point2 = "[Point 2] Books that are frequently co-purchased with the target book: [\n"

        for idx, v in enumerate(neighbors[target_node]) : 
            
            point2 += "- Book {0}: <{1}>\n".format(idx+1, raw_text[v])
    
    else : 
        
        if use_skip_connection : 
            
            point1 = "[Point 1] Target book: [Detailed description: <{0}>\nGeneral description: <{1}>]\n".format(raw_text[target_node], first_layer[target_node])
            point2 = "[Point 2] Books that are frequently co-purchased with the target book: [\n"
            
            for idx, v in enumerate(neighbors[target_node]) : 
            
                point2 += "- Book {0}: [Detailed description: <{1}>\nGeneral description: <{2}>]\n".format(idx+1, raw_text[v], first_layer[v])
                
        else : 
            
            point1 = "[Point 1] Target book: <{0}>.\n".format(first_layer[target_node])
            point2 = "[Point 2] Books that are frequently co-purchased with the target book: [\n"

            for idx, v in enumerate(neighbors[target_node]) : 

                point2 += "- Book {0}: <{1}>\n".format(idx+1, first_layer[v])
    
    point2 += "]\n"
        
    task_prompt2 = "[Instructions] Your summary must:\n"
    if use_attention : 
        attention_prompt = "- In summarizing the books in [Point 2], give more emphasis to those more relevant to the target book in [Point 1].\n"
    else : 
        attention_prompt = ""
    
    other_instruction1 = "- Return 2 paragraphs at most.\n"
    other_instruction2 = "- Do not introduce external facts; only use the given data.\n"
    other_instruction3 = "- Do not mention specific books by name; focus on content.\n"
    other_instruction4 = "- Output only the refined description (no extra commentary).\n"
    
    total_prompt = task_prompt1 + point1 + point2 + task_prompt2 + attention_prompt + other_instruction1 + other_instruction2 + other_instruction3 + other_instruction4
    
    return total_prompt

def generate_page_prompt(target_node = 0, raw_text = None, first_layer = None, neighbors = None, 
                         layer_number = 1, use_attention = True, use_skip_connection = True) : 
    
    task_prompt1 = "[Task] Refine the target web page's description in [Point 1] by incorporating the web pages from [Point 2].\n"
    
    if layer_number == 1: # First layer
        
        point1 = "[Point 1] Target web page: <{0}>.\n".format(raw_text[target_node])
        point2 = "[Point 2] Web pages that are hyperlinked to or from the target web page: [\n"

        for idx, v in enumerate(neighbors[target_node]) : 
            
            point2 += "- Web page {0}: <{1}>\n".format(idx+1, raw_text[v])
    
    else : 
        
        if use_skip_connection : 
            
            point1 = "[Point 1] Target web page: [Detailed description: <{0}>\nGeneral description: <{1}>]\n".format(raw_text[target_node], first_layer[target_node])
            point2 = "[Point 2] Web pages that are hyperlinked to or from the target web page: [\n"
            
            for idx, v in enumerate(neighbors[target_node]) : 
            
                point2 += "- Web page {0}: [Detailed description: <{1}>\nGeneral description: <{2}>]\n".format(idx+1, raw_text[v], first_layer[v])
                
        else : 
            
            point1 = "[Point 1] Target web page: <{0}>.\n".format(first_layer[target_node])
            point2 = "[Point 2] Web pages that are hyperlinked to or from the target web page: [\n"

            for idx, v in enumerate(neighbors[target_node]) : 

                point2 += "- Web page {0}: <{1}>\n".format(idx+1, first_layer[v])
    
    point2 += "]\n"
        
    task_prompt2 = "[Instructions] Your summary must:\n"
    if use_attention : 
        attention_prompt = "- In summarizing the web pages in [Point 2], give more emphasis to those more relevant to the target web page in [Point 1].\n"
    else : 
        attention_prompt = ""
    
    other_instruction1 = "- Return 2 paragraphs at most.\n"
    other_instruction2 = "- Do not introduce external facts; only use the given data.\n"
    other_instruction3 = "- Do not mention specific web page by name; focus on content.\n"
    other_instruction4 = "- Output only the refined description (no extra commentary).\n"
    
    total_prompt = task_prompt1 + point1 + point2 + task_prompt2 + attention_prompt + other_instruction1 + other_instruction2 + other_instruction3 + other_instruction4
    
    return total_prompt

def zero_shot_node_classification(target_node_description, data_type = "arxiv") : 
    
    if data_type == "arxiv" : 
    
        prompt = "[Data description] You have data describing a single paper: [{0}].\n[Task]: Choose the most suitable field this paper belongs to from the followings: [<artificial intelligence>, <hardware architecture>, <computational complexity>, <computational engineering, finance, and science>, <computational geometry>, <computation and language>, <cryptography and security>, <computer vision and pattern recognition>, <computers and society>, <databases>, <distributed, parallel, and cluster computing>, <digital libraries>, <discrete mathematics>, <data structures and algorithms>, <emerging technologies>, <formal languages and automata theory>, <general literature>, <graphics>, <computer science and game theory>, <human-computer interaction>, <information retrieval>, <information theory>, <machine learning>, <logic in computer science>, <multiagent systems>, <multimedia>, <mathematical software>, <numerical analysis>, <neural and evolutionary computing>, <networking and internet architecture>, <other computer science>, <operating systems>, <performance>, <programming language>, <robotics>, <symbolic computation>, <sound>, <software engineering>, <social and information networks>, <systems and control>].\n[Instruction] Only return a single predicted field in the format of <field name>. DO NOT include any other words.".format(target_node_description)
        
    elif data_type == "book" : 
        
        prompt = "[Data description] You have data describing a single book: [{0}].\n[Task]: Choose the most suitable category this book belongs to from the followings: [<Africa>, <Americas>, <Ancient Civilizations>, <Arctic & Antarctica>, <Asia>, <Australia & Oceania>, <Europe>, <Historical Study & Educational Resources>, <Middle East>, <Military>, <Russia>, <World>].\n[Instruction] Only return a single predicted category in the format of <category name>. DO NOT include any other words.".format(target_node_description)
        
    elif data_type == "page" : 
        
        prompt = "[Data description] You have data describing a single web page: [{0}].\n[Task]: Choose the most suitable category this web page belongs to from the followings: [<computational linguistics>, <databases>, <operating systems>, <computer architecture>, <computer security>, <internet protocols>, <computer file systems>, <distributed computing architecture>, <web technology>, <programming language topics>].\n[Instruction] Only return a single predicted category in the format of <category name>. DO NOT include any other words.".format(target_node_description)
        
    return prompt

def zero_shot_link_prediction(target_node_description, candidate_sets, data_type = "arxiv") : 
    
    
    ## Paper
    if data_type == "arxiv" : 
        
        prompt = "[Task]: Among the 10 candidate papers, choose the one that is most likely to cite or be cited by the target paper.\n"
        prompt += "Target paper: [{0}].\n".format(target_node_description)
        prompt += "Candidate papers: [\n"
        
        for vidx, cur_data in enumerate(candidate_sets) : 
            
            prompt += "- Paper {0}: [{1}]\n".format(vidx+1, cur_data)
            
        prompt += "[Instruction] Only return the predicted paper number in the format of [Paper k]. DO NOT include any other words."
        
        
    ## Book
    elif data_type == "book" : 
        
        prompt = "[Task]: Among the 10 candidate books, choose the one that is most likely to be co-purchased with the target book.\n"
        prompt += "Target book: [{0}].\n".format(target_node_description)
        prompt += "Candidate books: [\n"
        
        for vidx, cur_data in enumerate(candidate_sets) : 
            
            prompt += "- Book {0}: [{1}]\n".format(vidx+1, cur_data)
            
        prompt += "[Instruction] Only return the predicted book number in the format of [Book k]. DO NOT include any other words."
        
        
    ## Page
    elif data_type == "page" : 
        
        prompt = "[Task]: Among the 10 candidate web pages, choose the one that is most likely to be hyperlinked to or from the target web page.\n"
        prompt += "Target web page: [{0}].\n".format(target_node_description)
        prompt += "Candidate web pages: [\n"
        
        for vidx, cur_data in enumerate(candidate_sets) : 
            
            prompt += "- Web page {0}: [{1}]\n".format(vidx+1, cur_data)
            
        prompt += "[Instruction] Only return the predicted web page number in the format of [Page k]. DO NOT include any other words."
        
    return prompt

def zero_shot_node_classification_with_reasoning(target_node_description) : 

    
    prompt = "[Data description] You have data describing a single paper: [{0}].\n[Task]: Choose the most suitable field this paper belongs to from the followings: [<artificial intelligence>, <hardware architecture>, <computational complexity>, <computational engineering, finance, and science>, <computational geometry>, <computation and language>, <cryptography and security>, <computer vision and pattern recognition>, <computers and society>, <databases>, <distributed, parallel, and cluster computing>, <digital libraries>, <discrete mathematics>, <data structures and algorithms>, <emerging technologies>, <formal languages and automata theory>, <general literature>, <graphics>, <computer science and game theory>, <human-computer interaction>, <information retrieval>, <information theory>, <machine learning>, <logic in computer science>, <multiagent systems>, <multimedia>, <mathematical software>, <numerical analysis>, <neural and evolutionary computing>, <networking and internet architecture>, <other computer science>, <operating systems>, <performance>, <programming language>, <robotics>, <symbolic computation>, <sound>, <software engineering>, <social and information networks>, <systems and control>].\n[Instruction] Provide the predicted category and explain your reasoning.".format(target_node_description)

        
    return prompt