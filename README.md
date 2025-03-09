### Code for Bachelor Thesis at Hochschule Darmstadt. 

The purpose of this work is to generate high-quality synthetic data to train a reasoning LLM. This is done by prompting the LLM to output its reasoning process given human-written solution and then fine-tuning on the data.

This repo provides the data generation code as well as fine-tuning script.

### Installation
First, run pip install

```pip install -r requirements.txt```

Then setup config.yml
1. Add API_KEY ```api_key: <API_KEY>```. 
2. Choose baseurl ```baseurl: <BASE_URL>```. Support OpenAI compatible endpoints, including deepinfra and openrouter
3. Download BIRD dataset and put the root path in ```bird_root_path: <bird_path>\train\train_databases``` where <bird_path> is the path to the BIRD dataset.
4. Same for BIRD dev split
5. Download Spider dataset and put root path in the repo ![img.png](img.png)


### Usage
See main.py for details.

Main functions are ```distillWrapper``` and ```evaluateModel```.

DistillWrapper starts the distillation process which uses ```model_name``` to generate distillation data for ```dataset``` and the coresponding ```split```.
Set ```batchRange``` to specify the number of batches to generate, ```None``` to distill the whole dataset.

```distillWrapper(model_name="gpt-4o", dataset="bird", split="train", batchRange=batchRange)```

This will generate a .csv file, which can then be transformed into Alpaca-Format using
```python
convertDistillationEntriesToAlpaca(
    inputFilePath="datasets/distilled_bird/bird_train_distilled.csv",
    outputFilePath="bird_train_distilled-alpaca.json",
    outputExtension="json"
)
```
The repo provides a fine-tuning scripts using LoRA and Unsloth which can be found in the Finetuning Script.ipynb.

To evaluate a model use
```python
split="minidev"
datasetName = "bird"
# batchRange=(0,10)
batchRange=None
result = evaluateModel(
    model_name,
    batchRange=batchRange,
    datasetName=datasetName,
    split=split,
    promptTemplate=config["alpaca_inference_template"],
)
```
Additionally, to add more models/custom models, modify prompt_delivery.py
add the model name to 
```python
self.modelPromptStrategyMap = {
    "gpt-4o": self._deliverAPIPrompt,
    <others models>
}
```
and the corresponding prompt strategy.
There are 3 prompt strategies: 
1. ```_deliverAPIPrompt``` for OpenAI API compatible API LLMs
2. ```_deliverOllamaPrompt``` for Ollama LLMs
3. ```_deliverTransformersTokenizerPrompt``` for Huggingface models using Transformers
