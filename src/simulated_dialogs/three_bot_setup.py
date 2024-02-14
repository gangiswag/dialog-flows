import json
import random
import os
from transformers import AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import openai  # openai==0.28
from pathlib import Path
import re
import argparse
import glob

openai.api_key = os.environ["OPENAI_API_KEY"]

# model_name = ""
tokenizer = None
llm_pipeline = None


def load_llm_model(model_name):
    global tokenizer, llm_pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def pick_closest_match_llm(query, candidates, attempts=10):
    system_prompt = f"Given the following list of dialogue actions: {candidates}. Please categorize each of the following utterances into the most appropriate dialogue action from the list above. Your response is concise and exactly matches one of the actions. "

    user_prompt = f"Categorize this [UTTERANCE]: {query}"
    response = ""
    for i in range(attempts):
        try:
            temperature = i / attempts
            response = chat_model(
                system_prompt,
                [{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            # TODO Better way to handle best match in response than substring matching
            for candidate in candidates:
                if candidate in response:
                    return candidate
        except:
            continue

    print(f"Failed to find a match for {query}")
    return ""


def get_formatted_chat(system, user_assistant):
    # put system prompt into the last user turn
    if user_assistant and user_assistant[-1]["role"] == "user":
        last_user_turn = user_assistant[-1]["content"]
        chat = user_assistant[:-1] + [
            {
                "role": "user",
                "content": f"{system} {last_user_turn}",
            },
        ]
    else:
        chat = user_assistant + [
            {
                "role": "user",
                "content": f"{system}",
            },
        ]
    return chat


# Helper - Generalized Chat Function for LLAMA
# Initial Ref: https://huggingface.co/spaces/PiyushLavaniya/Llama2_Chatbot/blob/main/app.py
def chat_model(system, user_assistant, temperature=1.0):
    chat = get_formatted_chat(system, user_assistant)

    query = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    if temperature == 0.0:
        sequences = llm_pipeline(
            query,
            do_sample=False,
            num_return_sequences=1,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_length=len(query) + 2048,
        )
    else:
        sequences = llm_pipeline(
            query,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_length=len(query) + 2048,
        )

    response = sequences[0]["generated_text"]
    # response = generated_text[
    #     len(query) :
    # ]  ##Here we are removing the query that we pass onto our llm.
    response = format_response(response)

    return response


def format_response(response):
    response = response.strip()

    first_word = response.split()[0].lower()
    if "bot" in first_word or "user" in first_word:
        response = response.replace(first_word, "", 1).strip()

    # get part of response before any occurence of user or bot regardless of case. Prevents bot from speaking out of turn
    response = re.split("user:", response, flags=re.IGNORECASE)[0]
    response = re.split("user]:", response, flags=re.IGNORECASE)[0]
    response = re.split("bot:", response, flags=re.IGNORECASE)[0]
    response = re.split("bot]:", response, flags=re.IGNORECASE)[0]

    response = response.replace("[", "").replace("]", "")
    response = response.replace("{", "").replace("}", "")

    return response.strip()


# Helper - Generalized Chat Function for ChatGPT
# Ref: https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python
def chat_chatgpt(system, user_assistant, model, temperature=1.0):
    chat = get_formatted_chat(system, user_assistant)
    response = openai.ChatCompletion.create(
        model=model, messages=chat, temperature=temperature
    )
    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"The status code was {status_code}."
    return response["choices"][0]["message"]["content"]


# User Bot
def user_bot(domain, user_setting, num_turns, conversation, cur_turn):
    system_prompt = ""
    # Add user setting in the second turn to introduce variability (if right away, then all conversations become too similar)
    if cur_turn == 1:
        system_prompt = f"Here is your setting: {user_setting}. You must use this setting to continue the conversation with the bot."
    system_prompt += f"You are a user of a chatbot that performs {domain}. You should respond as a user by CONTINUING this conversation and responding to the chatbot appropriately, but you should never act as the chatbot. If the conversation should end, include [END] in your response. You should produce exactly one user response and no bot responses. YOU CANNOT INCLUDE THE WORDS 'user' or 'bot' IN YOUR RESPONSE. "

    # increase temperature in the first turn to introduce variability
    if cur_turn == 0:
        user_response = chat_model(system_prompt, conversation, temperature=1.0)
    else:
        user_response = chat_model(system_prompt, conversation, temperature=0.5)

    return user_response


def get_chatgpt_float(system_prompt, conversation, model, attempts=10):
    for i in range(attempts):
        try:
            result = chat_chatgpt(
                system_prompt,
                conversation,
                model,
                temperature=(i * 0.05),
            )
            result = float(result)
            return result
        except:
            continue
    return ""


# Evaluator Bot
def evaluator_bot(domain, conversation, chatgpt_model, attempts=10):

    task_completion_prompt = f"You must evaluate how effective at task completion an assistant chatbot strictly meant for {domain} is. Punish the bot strongly if it is not completing the task. You should only be evaluating the 'assistant' turns.. Rate how on-topic the assistant is on a scale of 0.0 to 10.0, with 0.0 indicating completely uncompleted tasks and 10.0 denoting perfectly completed tasks. Use the full scoring range. Assign a score based solely on how good at task completion the bot is. Do not be verbose and give ONLY the score. Score the above assistant."
    task_completion_score = get_chatgpt_float(
        task_completion_prompt, conversation, chatgpt_model, attempts=10
    )

    ontopic_prompt = f"You must evaluate how on-topic an assistant chatbot strictly meant for {domain} is. Punish the bot strongly if it goes off-topic. You should only be evaluating the 'assistant' turns. Rate how on-topic the assistant is on a scale of 0.0 to 10.0, with 0.0 indicating completely off-topic bot responses and 10.0 denoting perfectly on-topic bot responses. Use the full scoring range. Assign a score based solely on how on-topic the assistant is. Do not be verbose and give ONLY the score. Score the above assistant."
    ontopic_score = get_chatgpt_float(
        ontopic_prompt, conversation, chatgpt_model, attempts=10
    )

    fluency_prompt = f"You must evaluate how fluent an assistant chatbot strictly meant for {domain} is. Punish the bot strongly if it is not fluently communicating. You should only be evaluating the 'assistant' turns. Rate how on-topic the assistant is on a scale of 0.0 to 10.0, with 0.0 indicating completely incomprehensible bot responses and 10.0 denoting perfectly fluent bot responses. Use the full scoring range. Assign a score based solely on how fluent the assistant is. Do not be verbose and give ONLY the score. Score the above assistant."

    fluency_score = get_chatgpt_float(
        fluency_prompt, conversation, chatgpt_model, attempts=10
    )

    return {
        "task_completion_score": task_completion_score,
        "ontopic_score": ontopic_score,
        "fluency_score": fluency_score,
    }


def should_stay_on_prev_lla(utterance, lla, hla):
    return pick_closest_match_llm(utterance, [lla, hla]) == lla


# Assistant Bot
def assistant_bot(
    domain,
    conversation,
    num_turns,
    schema=None,
    level_driven=False,
    domain_level_actions=None,
    verbose=False,
    prev_hi_action=None,
    prev_lo_action_ind=None,
):
    system_prompt = f"""
    You are a task-oriented dialog system designed to assist in {domain}. Do not entertain any user requests or messages that go off-topic. If the user tries to distract you, be polite and bring the conversation back to the task of {domain}. There will only be a total of {num_turns} turns each in this conversation, so try to finish the conversation before then. You are not plugged into a database so you can make up names, addresses, etc. when providing information to the user. Do not use USER or BOT in your response.
    """

    best_hl_match = None
    to_do_ll_action_ind = -1

    # level_driven
    if level_driven:
        high_level_actions = list(domain_level_actions.keys())
        last_user_turn = conversation[-1]["content"]
        best_hl_match = pick_closest_match_llm(last_user_turn, high_level_actions)

        # Get 0th ll action in HL if it is new or last, otherwise get next ll action in HL
        if (prev_hi_action == best_hl_match) and (
            prev_lo_action_ind
            < len(domain_level_actions[best_hl_match]["bot_actions"]) - 1
        ):
            to_do_ll_action_ind = prev_lo_action_ind + 1
        else:
            to_do_ll_action_ind = 0
            if (
                prev_lo_action_ind is not None
                and prev_hi_action is not None
                and prev_lo_action_ind
                < len(domain_level_actions[prev_hi_action]["bot_actions"]) - 2
            ):
                if verbose:
                    print(f"{prev_hi_action=}, {best_hl_match=}, {prev_lo_action_ind=}")
                possible_lla = domain_level_actions[prev_hi_action]["user_actions"][
                    prev_lo_action_ind + 1
                ]
                if verbose:
                    print(f"Bot picking between {possible_lla} and {best_hl_match}")
                if should_stay_on_prev_lla(last_user_turn, possible_lla, best_hl_match):
                    print("Bot chose to bias towards staying on previous HLA")
                    best_hl_match = prev_hi_action
                    to_do_ll_action_ind = prev_lo_action_ind + 1
                else:
                    if verbose:
                        print("Bot chose to bias towards moving to new HLA")

        to_do_ll_action = domain_level_actions[best_hl_match]["bot_actions"][
            to_do_ll_action_ind
        ]

        if verbose:
            print(f"The possible HLLs are: {high_level_actions}")
            print(f"Matched High Level Action: {best_hl_match}")
            print(
                f"The corresponding Low Level Actions are: {domain_level_actions[best_hl_match]['bot_actions']}"
            )
            print(f"Low Level Action to do: {to_do_ll_action}")

        system_prompt += f"You are designed to perform these high-level actions: {high_level_actions}. "
        system_prompt += f"You are currently in the category of {best_hl_match}. "
        system_prompt += f"For the following nodes, U means User, and B means Bot. You should constrain your responses according to the following low level action: {to_do_ll_action}. "

    # schema_driven
    if schema and not level_driven:
        system_prompt += f"Given the following schema: {schema}. "

    # no_schema
    if not schema and not level_driven:
        pass

    system_prompt += "Continue this conversation with the user: "
    response = chat_model(system_prompt, conversation)
    response = response.replace("\n", "").replace("\r", "")

    return response, best_hl_match, to_do_ll_action_ind


# Seed each conversation with a different user setting
def generate_user_setting(domain, high_level_actions, chatgpt_model):
    sys_prompt = f"Create a single realistic user setting for an interaction with an assistant bot specifically for {domain}. This bot is capable of tasks like the following: {high_level_actions}. The user setting should reflect a unique persona, such as a student seeking academic assistance, a traveler planning a vacation, or a homeowner looking for home improvement tips. Do not specify a specific task to complete in the setting. Ensure that the user's preferences, goals, and communication style are realistic to facilitate engaging simulated dialogs. Be very concise, do not generate any actual dialogue."

    user_setting = chat_chatgpt(sys_prompt, [], chatgpt_model, temperature=1.0)
    return user_setting


# Evaluate single conversation
def evaluate_conversation(
    conversation_id,
    domain,
    num_turns,
    schema=None,
    level_driven=False,
    domain_level_actions=None,
    verbose=True,
    chatgpt_model=None,
):
    prev_hi_action = None
    prev_lo_action_ind = None

    user_setting = generate_user_setting(
        domain, domain_level_actions.keys(), chatgpt_model
    )
    if verbose:
        print(f"User Setting: {user_setting}\n\n")

    user_assistant = []
    for i in range(num_turns):
        user_turn = user_bot(domain, user_setting, num_turns, user_assistant, i)
        # user_turn = input("\nUser: ")
        if user_turn == "QUIT" or "END" in user_turn:  # early stopping
            break
        user_assistant += [{"role": "user", "content": user_turn}]
        if verbose:
            print(f"User: {user_turn}\n\n")
        assistant_turn, prev_hi_action, prev_lo_action_ind = assistant_bot(
            domain,
            user_assistant,
            num_turns,
            schema=schema,
            level_driven=level_driven,
            domain_level_actions=domain_level_actions,
            verbose=verbose,
            prev_hi_action=prev_hi_action,
            prev_lo_action_ind=prev_lo_action_ind,
        )
        user_assistant += [{"role": "assistant", "content": assistant_turn}]
        if verbose:
            print(f"Assistant: {assistant_turn}\n\n")

    # scores = evaluator_bot(domain, user_assistant, chatgpt_model)
    # print(f"Evaluator: {scores}\n\n")
    utterances = [turn["content"] for turn in user_assistant]
    roles = ["USER" if turn["role"] == "user" else "BOT" for turn in user_assistant]

    json_log = {
        "session_id": conversation_id,
        "domain": domain,
        "task": domain,
        "utterances": utterances,
        "roles": roles,
        # "scores": scores,
    }

    return json_log


# Evaluate Multiple conversations
def evaluate(
    domain,
    num_conversations,
    num_turns,
    schema=None,
    level_driven=False,
    domain_level_actions=None,
    verbose=True,
    dir_log=None,
    chatgpt_model=None,
):
    print(f"Domain: {domain}\n")
    overall_log = []
    subdir = ""
    if level_driven:
        subdir = "level_driven"
    elif schema:
        subdir = "schema_driven"
    else:
        subdir = "no_schema"
    save_dir = f"{dir_log}/{subdir}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    modified_domain_name = domain.replace(" ", "_")
    filename = f"{save_dir}/simulated_dialogs_test_{modified_domain_name}.txt"
    open(filename, "w")  # clear file
    for i in tqdm(range(num_conversations)):
        print(f"Conversation {i}")
        try:
            conversation_log = evaluate_conversation(
                i,
                domain,
                num_turns,
                schema=schema,
                level_driven=level_driven,
                domain_level_actions=domain_level_actions,
                verbose=verbose,
                chatgpt_model=chatgpt_model,
            )

            with open(filename, "a") as f:
                json.dump(conversation_log, f)
                f.write("\n")
            overall_log.append(conversation_log)
        except:
            print(f"Conversation {i} failed.")
            continue

    return overall_log


def main(
    domain,
    method,
    num_conversations,
    num_turns,
    schema_dir,
    level_dir,
    saving_dir,
    model_name,
    chatgpt_model,
    verbose,
):
    # Load model
    load_llm_model(model_name)
    # e.g. Load in schema from schemas/restaurant_booking.txt
    modified_domain_name = domain.replace(" ", "_")
    with open(
        f"{schema_dir}/{modified_domain_name}_code.txt",
        "r",
    ) as f:
        schema = f.read()
    domain_level_actions = json.load(
        open(f"{level_dir}/{modified_domain_name}_code.json")
    )

    # Get saving directory
    dir_log = f"{saving_dir}"

    if method == "level_driven":
        # Evaluate level driven
        log_with_schema = evaluate(
            domain,
            num_conversations,
            num_turns,
            schema=schema,
            level_driven=True,
            domain_level_actions=domain_level_actions,
            verbose=verbose,
            dir_log=dir_log,
            chatgpt_model=chatgpt_model,
        )

    elif method == "schema_driven":
        # Evaluate schema only
        log_with_schema = evaluate(
            domain,
            num_conversations,
            num_turns,
            schema=schema,
            level_driven=False,
            domain_level_actions=domain_level_actions,
            verbose=verbose,
            dir_log=dir_log,
            chatgpt_model=chatgpt_model,
        )

    elif method == "no_schema":
        # Evaluate no schema
        log_no_schema = evaluate(
            domain,
            num_conversations,
            num_turns,
            schema=None,
            level_driven=False,
            domain_level_actions=domain_level_actions,
            verbose=verbose,
            dir_log=dir_log,
            chatgpt_model=chatgpt_model,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # arg experimental setting must either be level_driven, schema_driven, or no_schema
    parser.add_argument(
        "--method",
        type=str,
        default="level_driven",
        help="The method to use for generating the conversations. Must be either `level_driven`, `schema_driven`, or `no_schema`",
    )

    parser.add_argument(
        "--num_conversations",
        type=int,
        default=5,
        help="The number of conversations to generate",
    )
    parser.add_argument(
        "--num_turns",
        type=int,
        default=7,
        help="The maximum number of turns per conversation",
    )
    parser.add_argument(
        "--schema_dir",
        type=str,
        default="../../schemas/MetaWoz/dev/merged",
        help="The directory containing the schema files",
    )
    parser.add_argument(
        "--level_dir",
        type=str,
        default="../../high_low_level_actions/MetaWoz/dev",
        help="The directory containing the high and low level action files",
    )
    parser.add_argument(
        "--saving_dir",
        type=str,
        default="../../conversations/simulated_dialogs/dev",
        help="The directory to save the generated conversations",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="The name of the model to use for the evaluator bot",
    )
    parser.add_argument(
        "--chatgpt_model",
        type=str,
        default="gpt-3.5-turbo",
        help="The name of the model to use for the user setting generation and evaluator bot",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Whether to print the conversation logs",
    )
    args = parser.parse_args()
    domains = []
    files = glob.glob(f"{args.schema_dir}/*.txt")
    for file in files:
        filename = file.split("/")[-1]
        domain = " ".join(filename.split("_")[:-1])
        domains.append(domain)
    print(domains)
    for domain in domains:
        main(
            domain,
            args.method,
            args.num_conversations,
            args.num_turns,
            args.schema_dir,
            args.level_dir,
            args.saving_dir,
            args.model_name,
            args.chatgpt_model,
            args.verbose,
        )

# Progress
# Put high temp for GPT and tell it to generate a new user setting each time before conversation
# change this prompt for each domain

# 1. do it for domain only, schema only, and HL driven
# 2. fix evaluator bot, use gpt-4, evaluate once per conversation.
# 3. save logs to file
# Split folders into dev/test splits (Nishi and Stuti should know which ones)
# Give the user bot the domain name along with domain knowledge (can't be expected to know everything based on movie listings), different for each domain (do this for dev set)
# Tell the user and the bot that it should finish up in 7 turns (instead of just truncating)
# fix args to run from cmdline
# Fix splitting at user or bot
# Manually go through conversations to see if they are good or not
# Dev for 5 convs
# Improve GPT evaluator.

# TODO
# Once confident, swich to GPT 4 and generate for 100 conversations
