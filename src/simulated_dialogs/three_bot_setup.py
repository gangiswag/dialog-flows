import json
import random
import os
from transformers import AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import openai  # openai==0.28
from pathlib import Path
import re

openai.api_key = os.environ["OPENAI_API_KEY"]
USE_CHATGPT = False
os.environ["CUDA_VISIBLE_DEVICES"] = "3,1"

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
llm_pipeline = pipeline(
    "text-generation", model=model_name, torch_dtype=torch.float16, device_map="auto"
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

    # print(f"Pre-templated chat: {chat}")
    query = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    if temperature == 0.0:
        sequences = llm_pipeline(
            query,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_length=2048,
        )
    else:
        sequences = llm_pipeline(
            query,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_length=2048,
        )

    generated_text = sequences[0]["generated_text"]
    response = generated_text[
        len(query) :
    ]  ##Here we are removing the query that we pass onto our llm.
    response = response.strip()

    first_word = response.split()[0].lower()
    if "bot" in first_word or "user" in first_word:
        response = response.replace(first_word, "", 1).strip()

    # get part of response before any occurence of user or bot regardless of case
    response = re.split("user", response, flags=re.IGNORECASE)[0]
    response = re.split("bot", response, flags=re.IGNORECASE)[0]

    return response.strip()


# Helper - Generalized Chat Function for ChatGPT
# Ref: https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python
def chat_chatgpt(system, user_assistant, model="gpt-3.5-turbo", temperature=1.0):
    chat = get_formatted_chat(system, user_assistant)
    response = openai.ChatCompletion.create(
        model=model, messages=chat, temperature=temperature
    )
    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"The status code was {status_code}."
    return response["choices"][0]["message"]["content"]


# User Bot
def user_bot(domain, user_setting, conversation):
    system_prompt = f"Here is your setting: {user_setting}. You are a user of a different chatbot that performs {domain}, and you must use this setting to continue the conversation with the bot. Give an example of a user response by continuing this conversation as a user. You should produce exactly one user response and no bot responses. Do not include the words 'user' or 'bot' in your response"
    is_adversarial = False
    # Use a random variable with a 0.3 probability of being true to determine whether to produce a relevant or irrelevant response.
    # if random.random() < 0.3:
    #     user_prompt += (
    #         f"Produce only a user response that is completely irrelevant to {domain}."
    #     )
    #     is_adversarial = True

    user_response = chat_model(system_prompt, conversation, temperature=0.5)

    return user_response, is_adversarial


def get_chatgpt_float(system_prompt, conversation, model, attempts=10):
    for i in range(attempts):
        try:
            result = chat_chatgpt(
                system_prompt,
                conversation,
                model=model,
                temperature=(i * 0.05),
            )
            result = float(result)
            return result
        except:
            continue
    return ""


# Evaluator Bot
def evaluator_bot(domain, conversation, attempts=10):
    model = "gpt-3.5-turbo"

    task_completion_prompt = f"You must evaluate how effective at task completion an assistant chatbot strictly meant for {domain} is. Punish the bot strongly if it is not completing the task. You should only be evaluating the 'assistant' turns.. Rate how on-topic the assistant is on a scale of 0.0 to 10.0, with 0.0 indicating completely uncompleted tasks and 10.0 denoting perfectly completed tasks. Use the full scoring range. Assign a score based solely on how good at task completion the bot is. Do not be verbose and give ONLY the score. Score the above assistant."
    task_completion_score = get_chatgpt_float(
        task_completion_prompt, conversation, model, attempts=10
    )

    ontopic_prompt = f"You must evaluate how on-topic an assistant chatbot strictly meant for {domain} is. Punish the bot strongly if it goes off-topic. You should only be evaluating the 'assistant' turns. Rate how on-topic the assistant is on a scale of 0.0 to 10.0, with 0.0 indicating completely off-topic bot responses and 10.0 denoting perfectly on-topic bot responses. Use the full scoring range. Assign a score based solely on how on-topic the assistant is. Do not be verbose and give ONLY the score. Score the above assistant."
    ontopic_score = get_chatgpt_float(ontopic_prompt, conversation, model, attempts=10)

    fluency_prompt = f"You must evaluate how fluent an assistant chatbot strictly meant for {domain} is. Punish the bot strongly if it is not fluently communicating. You should only be evaluating the 'assistant' turns. Rate how on-topic the assistant is on a scale of 0.0 to 10.0, with 0.0 indicating completely incomprehensible bot responses and 10.0 denoting perfectly fluent bot responses. Use the full scoring range. Assign a score based solely on how fluent the assistant is. Do not be verbose and give ONLY the score. Score the above assistant."

    fluency_score = get_chatgpt_float(fluency_prompt, conversation, model, attempts=10)

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
    schema=None,
    level_driven=False,
    domain_level_actions=None,
    verbose=False,
    prev_hi_action=None,
    prev_lo_action_ind=None,
):
    system_prompt = f"""
    You are a task-oriented dialog system designed to assist in {domain}. Do not entertain any user requests or messages that go off-topic. If the user tries to distract you, be polite and bring the conversation back to the task of {domain}. You are not plugged into a database so you can make up names, addresses, etc. when providing information to the user. Do not use USER or BOT in your response.
    """

    best_hl_match = None
    to_do_ll_action_ind = -1
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
                print(f"{prev_hi_action=}, {best_hl_match=}, {prev_lo_action_ind=}")
                possible_lla = domain_level_actions[prev_hi_action]["user_actions"][
                    prev_lo_action_ind + 1
                ]
                print(f"Bot picking between {possible_lla} and {best_hl_match}")
                if should_stay_on_prev_lla(last_user_turn, possible_lla, best_hl_match):
                    print("Bot chose to bias towards staying on previous HLA")
                    best_hl_match = prev_hi_action
                    to_do_ll_action_ind = prev_lo_action_ind + 1
                else:
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

    if schema and not level_driven:
        system_prompt += f"Given the following schema: {schema}. "

    if not schema and not level_driven:
        pass

    system_prompt += "Continue this conversation with the user: "
    response = chat_model(system_prompt, conversation)
    response = response.replace("\n", "").replace("\r", "")

    return response, best_hl_match, to_do_ll_action_ind


# Seed each conversation with a different user setting
def generate_user_setting(domain):
    sys_prompt = f"Create a single realistic user setting for an interaction with an assistant bot specifically for {domain}. The user setting should reflect a unique persona, such as a student seeking academic assistance, a traveler planning a vacation, or a homeowner looking for home improvement tips. Ensure that the user's preferences, goals, and communication style are realistic to facilitate engaging simulated dialogs. Be very concise, do not generate any actual dialogue."

    user_setting = chat_chatgpt(sys_prompt, [], model="gpt-3.5-turbo", temperature=1.0)
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
):
    prev_hi_action = None
    prev_lo_action_ind = None

    user_setting = generate_user_setting(domain)
    print(f"User Setting: {user_setting}\n\n")

    user_assistant = []
    for i in range(num_turns):
        user_turn, is_adversarial = user_bot(domain, user_setting, user_assistant)
        # user_turn = input("\nUser: ")
        if user_turn == "QUIT":
            break
        user_assistant += [{"role": "user", "content": user_turn}]
        print(f"User: {user_turn}\n\n")
        assistant_turn, prev_hi_action, prev_lo_action_ind = assistant_bot(
            domain,
            user_assistant,
            schema=schema,
            level_driven=level_driven,
            domain_level_actions=domain_level_actions,
            verbose=verbose,
            prev_hi_action=prev_hi_action,
            prev_lo_action_ind=prev_lo_action_ind,
        )
        user_assistant += [{"role": "assistant", "content": assistant_turn}]
        print(f"Assistant: {assistant_turn}\n\n")

    scores = evaluator_bot(domain, user_assistant)
    print(f"Evaluator: {scores}\n\n")

    json_log = {
        "session_id": conversation_id,
        "domain": domain,
        "task": domain,
        "utterances": [turn["content"] for turn in user_assistant],
        "roles": [turn["role"] for turn in user_assistant],
        "scores": scores,
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
    filename = f"{save_dir}/simulated_{modified_domain_name}.txt"
    with open(filename, "w") as f:
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
                )

                json.dump(conversation_log, f)
                f.write("\n")
                overall_log.append(conversation_log)
            except:
                print(f"Conversation {i} failed.")
                continue

    return overall_log


def main(domain):
    # e.g. Load in schema from schemas/restaurant_booking.txt
    modified_domain_name = domain.replace(" ", "_")
    with open(
        f"../../schemas/MetaWoz/merged/{modified_domain_name}_code.txt", "r"
    ) as f:
        schema = f.read()
    domain_level_actions = json.load(
        open(
            f"../../high_low_level_actions/MetaWoz/merged/{modified_domain_name}_code.json"
        )
    )

    # Get saving directory
    dir_log = f"../../conversations/simulated_dialogs"

    num_conversations = 3
    num_turns = 1

    # Evaluate level driven
    log_with_schema = evaluate(
        domain,
        num_conversations,
        num_turns,
        schema=schema,
        level_driven=True,
        domain_level_actions=domain_level_actions,
        verbose=True,
        dir_log=dir_log,
    )

    # Evaluate schema only
    log_with_schema = evaluate(
        domain,
        num_conversations,
        num_turns,
        schema=schema,
        level_driven=False,
        domain_level_actions=domain_level_actions,
        verbose=True,
        dir_log=dir_log,
    )

    # Evaluate no schema
    log_with_schema = evaluate(
        domain,
        num_conversations,
        num_turns,
        schema=False,
        level_driven=False,
        domain_level_actions=domain_level_actions,
        verbose=True,
        dir_log=dir_log,
    )


if __name__ == "__main__":
    main("movie listing")

# Progress
# Put high temp for GPT 4 and tell it to generate a new user setting each time before conversation
# change this prompt for each domain

# 1. do it for domain only, schema only, and HL driven
# 2. fix evaluator bot, use gpt-4, evaluate once per conversation.
# 3. save logs to file

# TODO
# Improve GPT evaluator. Switch to GPT 4 and find a good prompt.
# Better way to generate max length. Think that might be why "Hey there movie" keeps happening, prompt is very long
# Find a better way to handle the best match in response than substring matching
# Add in information into the overleaf
