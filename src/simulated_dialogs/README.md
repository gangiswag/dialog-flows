To run Simulated Dialogs, run the following within a new terminal (replace with your own OpenAI API Key). The default argument values are listed below.

```
export OPENAI_API_KEY=<your-api-key>
export CUDA_VISIBLE_DEVICES=0
python three_bot_setup.py \
    --num_conversations 5 \
    --num_turns 7 \
    --scoring False \
    --schema_dir "../../schemas/MetaWoz/dev/merged" \
    --level_dir "../../high_low_level_actions/MetaWoz/dev" \
    --saving_dir "../../conversations/simulated_dialogs/dev" \
    --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --chatgpt_model "gpt-3.5-turbo"
    --verbose False \
    --method schema_driven 
```

Notes:
- The method must be either `level_driven`, `schema_driven`, or `no_schema`.
- If `scoring=False`, `chatgpt_model` is not used. 
