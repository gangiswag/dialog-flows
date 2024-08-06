# Dialog Flows

This paper contains the code for the SigDIAL 2024 paper: "Dialog Flow Induction for Constrainable LLM-Based Chatbots"

The induced dialog flows are provided in the folder named `flows`. The code below runs the automatic evaluation coverage for the `MetaLWoz` and `MultiWOZ` datasets.


## Installation 
```
conda create --name dialog_flow python=3.10
conda activate dialog_flow
pip install -r requirements.txt
```

## Run the flow coverage

For running the evaluation coverage on the MetaLWoz dataset
```
python eval_coverage.py --dataset metalwoz --model mistralai/Mistral-7B-Instruct-v0.2 --conversations conversations/Metalwoz/ --flows flows/Metalwoz/ --results ./results/
```

For running the evaluation coverage on the MultiWOZ dataset
```
python eval_coverage.py --dataset multiwoz --model mistralai/Mistral-7B-Instruct-v0.2 --conversations conversations/Multiwoz/ --flows flows/Multiwoz/ --results ./results/
```