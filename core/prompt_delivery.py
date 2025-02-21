import instructor
import ollama
from icecream import ic

from models.OpenAIBatchRequestAPI import Message, Body, Request
from models.SQLQuery import SQLQuery
from openai import OpenAI

from core.utils import config
from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformerCache:
    transformerModel = None
    tokenizer = None

    @classmethod
    def get_or_create(cls, model_name):
        if cls.transformerModel is None:
            cls.transformerModel = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls.transformerModel, cls.tokenizer

#Delivers prompt to LLM over different interfaces, currently prototype, enough for the time being.
#In the future add other interfaces and ways to register model -> prompt strategy

class Prompt:
    """
    promptTemplate: str -> The message content
    """
    def __init__(
            self,
            modelName: str,
            promptTemplate: str = config["prompt_template"],
            isInstructor: bool = False,
            **kwargs
    ):
        self.modelName = modelName
        self.messageContent = promptTemplate.format(**kwargs)
        self.isInstructor = isInstructor
        self.defaultPromptStrategy = self._deliverOllamaPrompt
        self.modelPromptStrategyMap = {
            "gpt-4o": self._deliverAPIPrompt,
            "gpt-4o-mini": self._deliverAPIPrompt,
            "meta-llama/llama-3.2-3b-instruct": self._deliverAPIPrompt,
            "NyanDoggo/Qwen2.5-Coder-3B-Instruct-Spider-Vanilla": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Qwen2.5-Coder-3B-Instruct-Bird-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Qwen2.5-Coder-7B-Instruct-Spider-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Qwen2.5-Coder-7B-Instruct-Spider-Baseline": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Qwen2.5-Coder-0.5B-Instruct-Spider-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Phi-3.5-mini-instruct-Spider-Baseline": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Phi-3.5-mini-instruct-Spider-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Llama-3.2-3B-Instruct-Spider-Baseline": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Llama-3.2-3B-Instruct-Spider-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Llama-3.2-3B-Instruct-Bird-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Llama-3.2-3B-Instruct-Bird-Baseline": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Qwen2.5-Coder-7B-Instruct-Bird-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Qwen2.5-Coder-7B-Instruct-Bird-Baseline": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Meta-Llama-3.1-8B-Instruct-Spider-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Meta-Llama-3.1-8B-Instruct-Spider-Baseline": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Llama-3.1-8B-Instruct-Bird-Reasoning": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/Llama-3.1-8B-Instruct-Bird-Baseline": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/gemma-2-9b-it-Spider-Baseline": self._deliverTransformersTokenizerPrompt,
            "NyanDoggo/gemma-2-9b-it-Spider-Reasoning": self._deliverTransformersTokenizerPrompt,
            "microsoft/Phi-3.5-mini-instruct": self._deliverTransformersTokenizerPrompt,
            # "Qwen/Qwen2.5-Coder-0.5B-Instruct": self._deliverTransformersTokenizerPrompt,
            "Qwen/Qwen2.5-Coder-7B-Instruct": self._deliverTransformersTokenizerPrompt,
            "google/gemma-2-2b-it": self._deliverTransformersTokenizerPrompt,
            "google/gemma-2-9b-it": self._deliverAPIPrompt,
            "meta-llama/llama-3.1-8b-instruct": self._deliverAPIPrompt,
            "gemma2-9b-it": self._deliverGroqPrompt,
            "llama-3.1-8b-instant": self._deliverGroqPrompt,
        }
        self.baseurl, self.apiKey = self._getBaseUrlAndKey()

        self.transformerModel = None
        self.tokenizer = None

    def deliver(self):
        return self._setupPromptStrategy()()

    def _getBaseUrlAndKey(self):
        if self.modelName in ['gpt-4o', 'gpt-4o-mini']:
            return None, None #Uses base openAI url
        return config.get("baseurl", None), config.get("api_key", None)

    def _setupPromptStrategy(self):
        if self.isInstructor:
            return self._deliverPromptInstructor
        return self.modelPromptStrategyMap.get(self.modelName, self.defaultPromptStrategy)

    def _setupClient(self):
        if self.isInstructor:
            return self._setupInstructorClient()
        return OpenAI(base_url=self.baseurl, api_key=self.apiKey)

    def _setupInstructorClient(self):
        client = instructor.from_openai(
            OpenAI(
                base_url=config["default_ollama_server"],
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    def _deliverPromptInstructor(self, structuredOutputClass=SQLQuery):
        client = self._setupClient()
        response = client.chat.completions.create(
            model=self.modelName,
            messages=[
                {
                    "role": "user",
                    "content": self.messageContent,
                }
            ],
            response_model=structuredOutputClass,
        )
        return response

    def _deliverAPIPrompt(self):
        client = self._setupClient()
        ic(self.messageContent)
        if self.messageContent is None:
            return
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": self.messageContent
                }
            ], model=self.modelName,
            temperature=0.3
        )
        if response is None:
            return
        return response.choices[0].message.content

    def _deliverOllamaPrompt(self):
        return ollama.generate(model=self.modelName, prompt=self.messageContent)['response']

    def _deliverTransformersTokenizerPrompt(self):
        device = "cuda"  # the device to load the model onto

        if self.transformerModel is None:
            self.transformerModel, self.tokenizer = TransformerCache.get_or_create(self.modelName)

        messages = [{"role": "user", "content": self.messageContent}]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.transformerModel.generate(model_inputs.input_ids, max_new_tokens=4096, do_sample=True)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        return response

