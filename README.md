# Dialog Schema

# Steps to run  schema coverage numbers mentioned in the paper

## Create a conda environment 
```
conda create --name myenv
```
## Install the requirements 
```
pip install -r requirements.txt
```

## Run the flow coverage

For running the numbers for the dev set for the MetaLWoz dataset
```
python eval_coverage.py metawoz mistralai/Mistral-7B-Instruct-v0.2 conversations/Metawoz/dev/ schemas/Metawoz/dev/ results/Metawoz/dev/ 
```

For running the numbers for the dev set for the MetaLWoz dataset in batches due to the large number of domains in MetaLWoz
```
python eval_coverage.py metawoz mistralai/Mistral-7B-Instruct-v0.2 conversations/Metawoz/test/ schemas/Metawoz/test/ results/Metawoz/test/ --batch {1,2,3,4,5}
```

For running the numbers for the dev set for the MetaLWoz dataset in batches due to the large number of domains in Multiwoz
```
python eval_coverage.py multiwoz mistralai/Mistral-7B-Instruct-v0.2 conversations/Multiwoz schemas/Multiwoz results/Multiwoz
```
