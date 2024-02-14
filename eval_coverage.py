import argparse
import os
import glob
from collections import defaultdict
import json 
from transformers import pipeline, AutoTokenizer
import pandas as pd
import torch
import time
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import re
import numpy as np
import json
import traceback

tokenizer = None
llm_pipeline = None

def run_model(model):
    global tokenizer, llm_pipeline
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm_pipeline = pipeline("text-generation", model=model, torch_dtype=torch.float16, device_map="auto")


def get_label_name(line):  
    label = line.split("[")[0]
    name = (line.split("[")[1]).split("]")[0]
    return label, name

def get_label_name(line):  
    label = line.split("[")[0]
    name = (line.split("[")[1]).split("]")[0]
    return label, name


def parse_graphs(schema_g, low_level_labels, curr_node, visited_g, curr_start):
    tos = schema_g[curr_node]
    if len(tos) == 1 and tos[0] not in schema_g["end_nodes"]:
        to = tos[0]
        if curr_start in low_level_labels:
            low_level_labels[curr_start].append(curr_node)
            visited_g[tos[0]] = 1
            parse_graphs(schema_g, low_level_labels, tos[0], visited_g, curr_start)
        else:
            low_level_labels[curr_node].append(curr_node)
            visited_g[curr_node]=1
            parse_graphs(schema_g, low_level_labels, to, visited_g, curr_node)
        
    elif len(tos) > 1 and tos[0] not in schema_g["end_node"]:
        for to in tos:
            if visited_g[to] == 0:
                visited_g[to] = 1
                parse_graphs(schema_g, low_level_labels, to, visited_g, to)

def parse_file(schema_g, nodes_g, visited_g, schema_code_file) :
    to_list = []
    fro_list = []
    with open(schema_code_file, 'r') as f:
        for line in f:
            data = line.strip()
            
            if "-->" in data:
                fro = data.split("-->")[0].strip()
                if "[" in fro:
                    fro, name = get_label_name(fro)
                    nodes_g[fro] = name
                to = data.split("-->")[1].strip()
                if "[" in to:
                    to, name = get_label_name(to)
                    nodes_g[to] = name
                to_list.append(to)
                fro_list.append(fro)
                schema_g[fro].append(to)
                visited_g[to] = 0
            else:
                if "[" in data:
                    data, name = get_label_name(data)
                    nodes_g[data] = name

    fro= ""
    for fros in fro_list:
        if fros not in to_list:
            fro = fros
    tos = []
    for to in to_list:
        if to not in fro_list:
            tos.append(to)
    return fro,  tos

def create_graph_from_edges(schema_code_file):
    schema_g = defaultdict(list)
    nodes_g = defaultdict(str)
    visited_g = defaultdict(int)
    user_node_names = dict()
    bot_node_names = dict()

    start_node, end_nodes = parse_file(schema_g, nodes_g, visited_g, schema_code_file)
    schema_g["end_node"] = end_nodes

    for node, label in nodes_g.items():
        if node[0] == "u" or node[0] == "U":
            user_node_names[node]  = label 
        elif node[0] == "b" or node[0] == "B":
            bot_node_names[node]  = label

    node_name_to_no = {}
    for no, name in nodes_g.items():
        node_name_to_no[name] = no
    return nodes_g, node_name_to_no, user_node_names, bot_node_names, schema_g

def load_conversations(conversation_file):
    conversations = []
    with open(conversation_file, 'r') as f:
        for line in f:
            conversations.append(json.loads(line.strip()))
    conversations_df = pd.DataFrame(conversations)
    convos = []
    for i, row in conversations_df.iterrows():
        convos.append(row["utterances"])

    return convos

def chat_model(context, candidates, utterance, temperature=1.0, gen_length=4):
    generate_prompt = format_prompt(context, candidates, utterance)

    sampling = False
    if temperature > 0.0:
        sampling = True

    sequences = llm_pipeline(
        generate_prompt,
        do_sample=sampling,
        top_k=10,
        return_full_text=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=gen_length,
    )

    generated_text = sequences[0]["generated_text"].strip()
    
    return generated_text
    # response = generated_text[
    #     len(generate_prompt) :
    # ]  ##Here we are removing the query that we pass onto our llm.
    # return generated_text.strip()

def format_prompt(context, candidates, utterance):
    system_prompt = """You will be provided with a dialog history and a task-oriented bot utterance. Given a set of candidate dialog bot actions, your task is to identify the most appropriate dialog action that the bot has taken in the provided utterance. You SHOULD NOT generate the entire dialog action, ONLY output the action ID. Below are some examples provided some example examples below:
    
    ### Candidate Actions: [1: Greeting and Inquiry, 2: Asks for Clarification, 3: Provides Specific Movie Info, 4: Handles Errors and Misunderstandings, 5: Provides Showtimes for Specific Movie, 6: Asks for Feedback/Satisfaction, 7: Provides Specific Movie Info from List, 8: Lists Movies at Local Theater, 9: Explains reservation limits, 10: Provides Additional Info, 11: Informs About Ticket Prices]
    
    ### Utterance: The Day of the Big Ants will be playing at 7:15 and 10:45
    ### Action: 5

    ### Utterance: Yes, the AMC near you is playing Weekend at Bernies 2, a Star Wars Marathon, and Godfather III.
    ### Action: 8

    ### Utterance: Tickets for A Quiet Place at AMC are $13.95.
    ### Action: 11

    Now, you are provided with the following dialog history and the current task-oriented bot utterance. You need to classify the given bot utterance into a dialog action from the given candidate actions. You should use the dialog history as additional context when deciding the most appropriate dialog action. Generate ONLY the action ID corresponding to the dialog action. Do NOT generate anything else.

    ### Context: {context}

    ### Candidate Actions: {candidates}

    ### Utterance: {utterance}

    ### Action:"""

    candidates_list = [str(i+1) + ": " + candidate for i,candidate in enumerate(candidates)]
    candidates_str = "["+", ".join(candidates_list) + "]"

    inp_prompt = system_prompt.format(context=context, candidates=candidates_str, utterance=utterance)
    messages = [
                    {"role": "user", "content": inp_prompt}
                ]
    starting_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return starting_prompt   


def pick_closest_match(query, candidates, context, attempts=3):
    
    num_retries = 0
    temperature = 0.0
    gen_length = 4
    while num_retries < attempts:
        generated_text = chat_model(query, candidates, context, temperature=temperature, gen_length=gen_length)
        pattern = r"\d+"
        match = re.search(pattern, generated_text)
        if match:
            if int(match.group(0)) <= len(candidates):
                return candidates[int(match.group(0))-1]
            else:
                temperature = 1.0
                gen_length = 50
                num_retries +=1
        else:
            temperature = 1.0
            gen_length = 50
            num_retries +=1        

    return candidates[0]

def calculate_evaluation_score(convo_nodes, induced_schema):
    score = 0
    for i in range(len(convo_nodes) - 1):
        current_node = convo_nodes[i]
        next_node = convo_nodes[i + 1]
        
        next_user_actions = induced_schema[current_node]
        node_found = False
        for user_action in next_user_actions:

            if next_node in induced_schema.get(user_action, []) and not node_found:
                # Increase the score if the transition is valid in the expected schema
                score += 1
                node_found = True

            else:
                next_to_next_user_actions = []
                bot_actions =  induced_schema.get(user_action, [])
                for bot_action in bot_actions:
                    next_to_next_user_actions += induced_schema.get(bot_action, [])
                    for next_user_action in next_to_next_user_actions:
                        if next_node in induced_schema.get(next_user_action,[]) and not node_found:
                            score += 1
                            node_found = True  
    
    # total_possible_transitions = sum(len(connected_nodes) for connected_nodes in induced_schema.values())
    return score / (len(convo_nodes)-1)  # Normalize the score to be between 0 and 1


def match_utterances_to_labels(dataset, conversations, user_node_names, bot_node_names, nodes_name_to_no):
    bot_candidates = {}
    for bot_node in bot_node_names.values(): 
        bname = bot_node
        if bname[0] == '"':
            bname = bname[1:-1]
        if bname.startswith("Bot:"):
            bname = bname[4:]
        bot_candidates[bname] = bot_node       

    convos_nodes = []
    for conversation in tqdm(conversations):
        convo_nodes = []
        context = []
        for i, utterance in enumerate(conversation):
            if len(context) == 3:
                context.pop(0)
            if (i % 2 == 0 and dataset == "metawoz") or (i % 2 != 0 and (dataset == "multiwoz" or "simulated_dialogs" in dataset)):
                context.append("Bot: "+ utterance)
            elif (i % 2 != 0 and dataset == "metawoz") or (i % 2 == 0 and (dataset == "multiwoz" or  "simulated_dialogs" in dataset)):
                context.append("User: " + utterance)

            if i % 2 == 0:
                best_matches = pick_closest_match(utterance, list(bot_candidates.keys()), context)
                # print(best_matches)
                if best_matches:
                    best_match = best_matches  # Take the top-ranked match
                    node_no = nodes_name_to_no[bot_candidates[best_match]]
                    convo_nodes.append(node_no)

        convos_nodes.append(convo_nodes)
    return convos_nodes
        
def find_domains_to_use(dataset, conversations):
    if dataset == "metawoz":
        if conversations.split("/")[2] == "dev":
            return "metawoz_dev_domains"
        elif conversations.split("/")[2] == "test":
            return "metawoz_test_domains" 
    elif dataset == "multiwoz":
        return "multiwoz_domains"
    else:
        return dataset
    
def main(dataset, model, conversations, schemas, results):
    domains = {
        "metawoz_test_domains" : ["alarm_set", "apartment_finder", "appointment_reminder","bank_bot", "bus_schedule_bot","catalogue_bot", "city_info", "edit_playlist", "event_reserve","guiness_check", "insurance", "library_request", "look_up_info", "music_suggester", "name_suggester", "pet_advice", "scam_lookup", "shopping", "ski_bot", "sports_info", "store_details", "update_calendar", "update_contact", "wedding_planner"],
        "metawoz_dev_domains" : ["phone_plan", "order_pizza", "movie_listings", "restaurant_picker", "weather_check"],
        "multiwoz_domains" : ["attraction", "hotel", "restaurant", "taxi", "train"],
        "simulated_dialogs_test" : ["alarm_set", "apartment_finder", "appointment_reminder","bank_bot", "bus_schedule_bot","catalogue_bot", "city_info", "edit_playlist", "event_reserve","guiness_check", "insurance", "library_request", "look_up_info", "music_suggester", "name_suggester", "pet_advice", "scam_lookup", "shopping", "ski_bot", "sports_info", "store_details", "update_calendar", "update_contact", "wedding_planner"],
        "simulated_dialogs_dev" : ["movie_listings", "order_pizza", "phone_plan", "restaurant_picker", "weather_check"]
    }
    

    # # Check if the provided paths exist
    # if not all(map(os.path.exists, [conversations, schemas, results])):
    #     raise ValueError("One or more paths do not exist.")
    if not os.path.exists(results):
        os.makedirs(results)

    # run the model
    run_model(model)

    domains_to_use = find_domains_to_use(dataset, conversations)

    methods = ["llm", "data", "merged"]
    for domain in domains[domains_to_use]:
        print("running for", domain)
        domain_results = {}
        try:
            for method in methods:
                schema_file = os.path.join(schemas, method, f"{domain}_code.txt")
                if "simulated_dialogs" in dataset:
                    
                else:
                    conversation_file = os.path.join(conversations, f"{dataset}_test_{domain}.txt")
                nodes_g, node_name_to_no, user_node_names, bot_node_names, schema_g = create_graph_from_edges(schema_file)
                convos = load_conversations(conversation_file)
                convos_nodes = match_utterances_to_labels(dataset, convos, user_node_names, bot_node_names, node_name_to_no)

                scores = [calculate_evaluation_score(convo_nodes, schema_g) for convo_nodes in convos_nodes]
                domain_results[method] = {
                    'max': np.max(scores),
                    'min': np.min(scores),
                    'average': np.mean(scores)
                }
                results_file_path = os.path.join(results, f"{dataset}_{domain}.json")
                with open(results_file_path, 'w') as results_file:
                    json.dump(domain_results, results_file, indent=4)
        except Exception as e:
            print(f"An error occurred during setup: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some files.')
    
    parser.add_argument('dataset', type=str, help='Path to the dataset file')
    parser.add_argument('model', type=str, help='Path to the model file')
    parser.add_argument('conversations', type=str, help='Path to the folder containing conversation files')
    parser.add_argument('schemas', type=str, help='Path to the folder containing schema files')
    parser.add_argument('results', type=str, help='Path to the folder where results will be saved')

    args = parser.parse_args()
    print("args parsed")
    
    main(args.dataset, args.model, args.conversations, args.schemas, args.results)

