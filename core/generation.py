import os
import time

import instructor
import ollama
from openai import OpenAI
from tqdm import tqdm

from core.data_loader import get_spider_db_path, getDbPath
from core.prompt_delivery import Prompt
from core.sql_tools import getDatabaseSchemaForPrompt
from core.utils import config, cleanLLMResponse
from models.SQLEvaluationEntry import SQLEvaluationEntry
from models.SQLQuery import SQLQuery
from models.SQLDataset import SQLDataset


def askAI(model, db_path, request, promptTemplate=config["alpaca_inference_template"]):
    schema = getDatabaseSchemaForPrompt(db_path)
    prompt = Prompt(
        modelName=model,
        promptTemplate=promptTemplate,
        request=request,
        schema=schema
    )
    return prompt.deliver()
# @suppress_prints
def generateSQL(model_name, promptTemplate=config["prompt_template"], db_path="", **kwargs):
    # kwargs should include arguments for the prompt template
    kwargs["db_path"] = db_path
    print("--" * 50)
    print(promptTemplate.format(**kwargs))
    print("--" * 50)
    prompt = Prompt(
        modelName=model_name,
        promptTemplate=promptTemplate,
        **kwargs
    )
    return prompt.deliver()

# @suppress_prints
def generateSQLEvaluationEntry(
        model_name: str,
        dataset_entry: SQLDataset,
        isInstructor=False,
        dataset="spider",
        split="train",
        promptTemplate=config["prompt_template"]
):
    db_path = getDbPath(dataset, dataset_entry.db_id, split=split)
    request = dataset_entry.question
    schema = getDatabaseSchemaForPrompt(db_path)

    response = generateSQL(model_name=model_name,
                           promptTemplate=promptTemplate,  #promptTemplate
                           db_path=db_path,
                           request=request,
                           schema=schema)
    print("----------RESPONSE----------")
    print(response)

    if isInstructor:
        generated_sql = response.sql_query
    else:
        generated_sql = cleanLLMResponse(response)

    return SQLEvaluationEntry(
        db_path=db_path,
        generated_sql=generated_sql,
        gold_sql=dataset_entry.query,
        question=dataset_entry.question,
        response=response
    )

def generateSQLEvaluationEntryFromLoadedDataset(model_name, datasetName, split, promptTemplate, instances):
    result = []
    for instance in tqdm(instances):
        evaluation_entry = generateSQLEvaluationEntry(model_name, instance, dataset=datasetName, split=split,
                                                      promptTemplate=promptTemplate)
        from core.evaluation import evaluateSQLGenerationEntry
        result.append(evaluateSQLGenerationEntry(evaluation_entry))
    return result