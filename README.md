# Dialog Schema

## Running the schema coverage
```
python eval_coverage.py metawoz mistralai/Mistral-7B-Instruct-v0.2 conversations/Metawoz/dev/ schemas/Metawoz/dev/ results/Metawoz/dev/ 
```

```
python eval_coverage.py metawoz mistralai/Mistral-7B-Instruct-v0.2 conversations/Metawoz/test/ schemas/Metawoz/test/ results/Metawoz/test/ --batch {1,2,3,4,5}
```

```
python eval_coverage.py multiwoz mistralai/Mistral-7B-Instruct-v0.2 conversations/Multiwoz schemas/Multiwoz results/Multiwoz
```
