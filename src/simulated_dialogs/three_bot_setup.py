import json
import random
import os
import pandas as pd
from transformers import AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import openai  # openai==0.28
from pathlib import Path

openai.api_key = os.environ["OPENAI_API_KEY"]
USE_CHATGPT = False
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# from huggingface_hub import login
# hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# huggingface_hub.login(token=hf_api_token, add_to_git_credential=True)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
llm_pipeline = pipeline(
    "text-generation", model=model_name, torch_dtype=torch.float16, device_map="auto"
)


def pick_closest_match_llm(query, candidates, attempts=10):
    system_prompt = f"Given the following list of dialogue actions: {candidates}. Please categorize each of the following utterances into the most appropriate dialogue action from the list above. Your response is concise and exactly matches one of the actions."

    # Few shot prompting
    # system_prompt += "For example,"
    # examples = [
    #     {
    #         "utterance": "Tell me more about Friday deals",
    #         "category": "Special Deals and Membership",
    #     },
    #     {
    #         "utterance": "I want to book a reservation for The Shining on Friday.",
    #         "category": "Reserve a Movie",
    #     },
    #     {
    #         "utterance": "Hi",
    #         "category": "Greeting and introduction",
    #     },
    # ]
    # for example in examples:
    #     system_prompt += (
    #         f"'{example['utterance']}' should exactly return '{example['category']}'."
    #     )

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


# Helper - Generalized Chat Function for LLAMA
# Initial Ref: https://huggingface.co/spaces/PiyushLavaniya/Llama2_Chatbot/blob/main/app.py
def chat_model(system, user_assistant, temperature=1.0):
    last_user_turn = user_assistant[-1]["content"] if len(user_assistant) > 0 else ""
    prev_conversation = user_assistant[:-1] if len(user_assistant) > 0 else []
    chat = user_assistant[:-1] + [
        {
            "role": "user",
            "content": f"{system} {last_user_turn}",
        },
    ]
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
    return response.strip()


# Helper - Generalized Chat Function for ChatGPT
# Ref: https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python
def chat_chatgpt(system, user_assistant, temperature=1.0):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(user_assistant, list), "`user_assistant` should be a list"
    system_msg = [{"role": "system", "content": system}]
    user_assistant_msgs = [
        (
            {"role": "assistant", "content": user_assistant[i]}
            if i % 2
            else {"role": "user", "content": user_assistant[i]}
        )
        for i in range(len(user_assistant))
    ]

    msgs = system_msg + user_assistant_msgs
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=msgs, temperature=temperature
    )
    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"The status code was {status_code}."
    return response["choices"][0]["message"]["content"]


# Helper - Tag Conversation with [USER] and [BOT]
def tag_conversation(conversation):
    tagged_conversation = []
    for i, turn in enumerate(conversation):
        if i % 2 == 0:
            tagged_conversation.append(f"[USER]: {turn}")
        else:
            tagged_conversation.append(f"[BOT]: {turn}")
    return tagged_conversation


# User Bot
def user_bot(domain, conversation):
    system_prompt = f"You are a user of a different chatbot that performs {domain}"

    system_prompt += f"Give only an example of a user response by continuing this conversation as a user."
    is_adversarial = False
    # Use a random variable with a 0.3 probability of being true to determine whether to produce a relevant or irrelevant response.
    # if random.random() < 0.3:
    #     user_prompt += (
    #         f"Produce only a user response that is completely irrelevant to {domain}."
    #     )
    #     is_adversarial = True

    user_response = chat_model(system_prompt, conversation)

    # user_response = chat_model(system_prompt, [user_prompt])
    # user_response = chat_model(
    #     system_prompt, [{"role": "user", "content": user_prompt}]
    # )

    # Remove "[USER]: " and related topics from the response
    user_response = (
        user_response.replace("[USER] ", "")
        .replace("[BOT] ", "")
        .replace("[USER]: ", "")
        .replace("[BOT]: ", "")
    )

    return user_response, is_adversarial


# TODO: Add in other metrics like fluency, interestingness, etc.
# Evaluator Bot
def evaluator_bot(domain, conversation):
    tagged_conversation = tag_conversation(conversation)
    return chat_chatgpt(
        f"You must evaluate how on-topic an assistant chatbot strictly meant for {domain} is. Punish the bot strongly if it goes off-topic. [USER] indicates a user turn and [BOT] indicates a bot turn. Rate how on-topic the current conversation is on a scale of 0.0 to 10.0, with 0.0 indicating completely off-topic bot responses and 10.0 denoting perfectly on-topic bot responses. Use the full scoring range. Assign a score based solely on how on-topic the conversation is. Do not be verbose.",
        # f"You are an evaluator of an assistant chatbot that is strictly for {domain}. Punish the bot strongly for off-topic conversations. [USER] indicates a user turn and [BOT] indicates a bot turn. Read the conversation and score it from 1.0-10.0, use the full range. Do not be verbose.",
        [f"Score the following conversation: {tagged_conversation}"],
        temperature=0.0,
    )


def should_stay_on_prev_lla(utterance, lla, hla):
    return pick_closest_match_llm(utterance, [lla, hla]) == lla


# Assistant Bot
def assistant_bot(
    domain,
    conversation,
    schema=None,
    domain_level_actions=None,
    verbose=False,
    prev_hi_action=None,
    prev_lo_action_ind=None,
):
    system_prompt = f"""
    You are a task-oriented dialog system designed to assist in {domain}. Do not entertain any user requests or messages that go off-topic. If the user tries to distract you, be polite and bring the conversation back to the task of {domain}. You are not plugged into a database so you can make up names, addresses, etc. when providing information to the user. [USER] indicates a user turn and [BOT] indicates a bot turn.  Do not use USER or BOT in your response.
    """

    best_hl_match = None
    to_do_ll_action_ind = -1
    if schema:
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

        system_prompt += f"For the following nodes, U means User, and B means Bot. You should constrain your responses according to the following low level action: {to_do_ll_action}"

    system_prompt += "Continue the conversation with the user."
    response = chat_model(system_prompt, conversation)
    response = response.replace("\n", "").replace("\r", "")

    if schema:
        bot_ll_match = pick_closest_match_llm(
            response, domain_level_actions[best_hl_match]["bot_actions"]
        )
        if verbose:
            print(f"Bot matched Low Level Action: {bot_ll_match}")

    return response, best_hl_match, to_do_ll_action_ind


# Evaluate single conversation
def evaluate_conversation(
    conversation_id,
    domain,
    num_turns,
    schema=None,
    domain_level_actions=None,
    verbose=True,
):
    prev_hi_action = None
    prev_lo_action_ind = None

    log = []
    user_assistant = []
    for i in range(num_turns):
        user_turn, is_adversarial = user_bot(domain, user_assistant)
        # user_turn = input("\nUser: ")
        if user_turn == "QUIT":
            break
        user_assistant += [{"role": "user", "content": user_turn}]
        assistant_turn, prev_hi_action, prev_lo_action_ind = assistant_bot(
            domain,
            user_assistant,
            schema=schema,
            domain_level_actions=domain_level_actions,
            verbose=verbose,
            prev_hi_action=prev_hi_action,
            prev_lo_action_ind=prev_lo_action_ind,
        )
        user_assistant += [{"role": "assistant", "content": assistant_turn}]

        # turned off evaluation bot for now
        # MAX_EVAL_ATTEMPTS = 5
        # for j in range(MAX_EVAL_ATTEMPTS):
        #     time.sleep(0.5)
        #     try:
        #         evaluator_turn = float(evaluator_bot(domain, user_assistant))
        #         break
        #     except:
        #         continue
        if verbose:
            print(f"User: {user_turn}")
            print(f"Assistant: {assistant_turn}")
            # print(f"Evaluation: {evaluator_turn}")

        log.append(
            {
                "conversation_id": conversation_id,
                "turn_id": i,
                # "is_adversarial": is_adversarial,
                "user": user_turn,
                "assistant": assistant_turn,
                # "evaluator": evaluator_turn,
            }
        )

    return log


# Evaluate Multiple conversations
def evaluate(
    domain,
    num_conversations,
    num_turns,
    schema=None,
    domain_level_actions=None,
    verbose=True,
):
    print(f"Domain: {domain}\n")
    log = []
    for i in tqdm(range(num_conversations)):
        print(f"Conversation {i}")
        # try:
        conversation_log = evaluate_conversation(
            i,
            domain,
            num_turns,
            schema=schema,
            domain_level_actions=domain_level_actions,
            verbose=verbose,
        )
        log.append(conversation_log)
        # except:
        #     print(f"Conversation {i} failed.")
        #     continue

    df_log = pd.DataFrame([turn for conversation in log for turn in conversation])
    return df_log


def main(domain):
    # e.g. Load in schema from schemas/restaurant_booking.txt
    modified_domain_name = domain.replace(" ", "_")
    with open(f"../../schemas/MetaWoz/merged/{modified_domain_name}.txt", "r") as f:
        schema = f.read()
    domain_level_actions = json.load(
        open(f"../../high_low_level_actions/MetaWoz/merged/{modified_domain_name}.json")
    )

    # Get saving directory
    formatted_model_name = format(model_name.split("/")[-1])
    dir_log = f"../../conversations/simulated_dialogs/{modified_domain_name}"
    Path(dir_log).mkdir(parents=True, exist_ok=True)

    num_conversations = 5
    num_turns = 5

    # Evaluate with schema
    df_with_schema = evaluate(
        domain,
        num_conversations,
        num_turns,
        schema=schema,
        domain_level_actions=domain_level_actions,
        verbose=True,
    )
    df_with_schema.to_csv(
        f"{dir_log}/with_schema.csv",
        index=False,
    )

    # Evaluate without schema
    df_without_schema = evaluate(
        domain, num_conversations, num_turns, schema=None, verbose=True
    )
    df_without_schema.to_csv(
        f"{dir_log}/without_schema.csv",
        index=False,
    )


if __name__ == "__main__":
    main("movie listing")

    # MDE Multi Domain Evaluation
    # domains = [
    #     "movie listing",
    #     "phone plan",
    #     "pizza order",
    #     "restaurant booking",
    #     "weather checker",
    # ]
    # for domain in domains:
    #     main(domain)

# TODO
# Put high temp for GPT 4 and tell it to generate a new user setting each time before conversation
# change this prompt for each domain
