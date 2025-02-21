import textwrap
from typing import Optional

import pandas
from django.db.models.expressions import result
from tqdm import tqdm

from core.data_loader import load_spider, get_spider_db_path, load_bird, getBirdDbPath
from core.evaluation import evaluateSQLGenerationEntry
from core.prompt_delivery import Prompt
from core.sql_tools import getDatabaseSchemaForPrompt
from core.utils import cleanLLMResponse, config, objects_to_dataframe, loadToObjectsFromFile
from models.DistillationEntry import DistillationEntry
from models.OpenAIBatchRequestAPI import BatchRequestController
from models.SQLEvaluationEntry import SQLEvaluationEntry
from pathlib import Path


def generateDistillationEntry(
        modelName: str,
        question: str,
        goldSolution: str,
        schema: str,
        promptTemplate: str = config["knowledge_distillation_generation_template"],
):
    prompt = Prompt(modelName=modelName, promptTemplate=promptTemplate, problem=question, solution=goldSolution,
                    schema=schema)
    model_response = prompt.deliver()

    reasoning = cleanLLMResponse(model_response, openTag="<reasoning>", closeTag="</reasoning>")
    distillationEntry = DistillationEntry(
        teacher_model_name=modelName,
        question=question,
        schema=schema,
        gold_solution=goldSolution,
        reasoning=reasoning,
        verification_solution="",
        isVerified=None
    )

    return distillationEntry


# Load distillation entries from file
def verifyDistillationFromFile(
        db_path: str,
        verifierModelName: str,
        distillationFilePath: str,
        promptTemplate: str = config["knowledge_distillation_verification_template"],
) -> list[DistillationEntry]:
    distillationEntries = loadToObjectsFromFile(distillationFilePath, DistillationEntry)
    verificationResult = []
    for distillationEntry in tqdm(distillationEntries):
        distillationEntry = verifyDistillationEntry(
            distillationEntry=distillationEntry,
            model_name=verifierModelName,
            db_path=db_path,
            promptTemplate=promptTemplate
        )
        verificationResult.append(distillationEntry)

    return verificationResult


def verifyDistillationEntry(
        distillationEntry: DistillationEntry,
        model_name: str,
        db_path: str,
        promptTemplate: str = config["knowledge_distillation_verification_template"]) -> DistillationEntry:
    '''
    This function verifies the logical reasoning of the generated distillation by prompting another model
    given the question and the reasoning to generate sql solution which would then be evaluated.
    :param distillationEntry:
    :param model_name:
    :param db_path:
    :param promptTemplate:
    :return:
    '''

    prompt = Prompt(
        modelName=model_name,
        promptTemplate=promptTemplate,
        problem=distillationEntry.question,
        schema=distillationEntry.schema,
        reasoning=distillationEntry.reasoning
    )
    model_response = prompt.deliver()

    distillationEntry.verification_solution = cleanLLMResponse(model_response, openTag="<final answer>",
                                                               closeTag="</final answer>")
    sqlEvaluationEntry = SQLEvaluationEntry(
        db_path=db_path,
        generated_sql=distillationEntry.verification_solution,
        gold_sql=distillationEntry.gold_solution,
        question=distillationEntry.question
    )
    sqlEvaluationEntry = evaluateSQLGenerationEntry(sqlEvaluationEntry)

    distillationEntry.isVerified = sqlEvaluationEntry.isCorrect

    return distillationEntry


def distillKnowledge(
        teacher_model_name: str,
        student_model_name: Optional[str] = None,
        dataset="spider",
        split="train",
        useCache=False,
        batchRange: Optional[tuple[int, int]] = None,
):
    """

    :param teacher_model_name: Model used to generate data
    :param student_model_name: Model used to verify reasoning if set to None then will not verify
    :param dataset: dataset to distill
    :param split: split of dateset
    :param useCache:
    :param batchRange:
    :return:
    """
    cachePath = Path(f'{config["cache_path"]}/distillation/{teacher_model_name}_{dataset}_{split}.csv')
    if useCache and cachePath.exists():
        return pandas.read_csv(cachePath)

    result = []
    if dataset == "spider":
        result = distillSpider(batchRange, result, split, student_model_name, teacher_model_name)
    if dataset == "bird":
        result = distillBird(batchRange, result, split, student_model_name, teacher_model_name)

    pd = objects_to_dataframe(result)
    if useCache:
        pd.to_csv(cachePath)

    return pd

def distillSpider(batchRange, result, split, student_model_name, teacher_model_name):
    spider_instances = load_spider(split, batchRange=batchRange)
    for instance in tqdm(spider_instances):
        db_path = get_spider_db_path(instance.db_id, split=split)
        schema = getDatabaseSchemaForPrompt(db_path)
        print("Distilling: " + instance.question)
        distillationEntry = generateDistillationEntry(
            modelName=teacher_model_name,
            question=instance.question,
            goldSolution=instance.query,
            schema=schema,
        )
        print("Reasoning: " + distillationEntry.reasoning)
        if student_model_name is not None:
            distillationEntry = verifyDistillationEntry(
                distillationEntry=distillationEntry,
                model_name=student_model_name,
                db_path=db_path
            )
        result.append(distillationEntry)

    return result


def distillBird(batchRange, result, split, student_model_name, teacher_model_name):
    bird_instances = load_bird(split, batchRange=batchRange)

    for instance in tqdm(bird_instances):
        db_path = getBirdDbPath(instance.db_id, split=split)
        schema = getDatabaseSchemaForPrompt(db_path)
        print("Distilling: " + instance.question)
        print("Schema:" + str(schema))
        distillationEntry = generateDistillationEntry(
            modelName=teacher_model_name,
            question=instance.question,
            goldSolution=instance.query,
            schema=schema,
        )
        print("Reasoning: " + distillationEntry.reasoning)
        if student_model_name is not None:
            distillationEntry = verifyDistillationEntry(
                distillationEntry=distillationEntry,
                model_name=student_model_name,
                db_path=db_path
            )
        result.append(distillationEntry)
    return result


def distillUnverifiedEntries(
        filePath: str,
        teacher_model_name,
        promptTemplate=config["knowledge_distillation_generation_template"]
):
    result = []
    try:
        distillationEntries = loadToObjectsFromFile(filePath, DistillationEntry)
        for i in range(len(tqdm(distillationEntries))):
            distillationEntry = distillationEntries[i]
            if distillationEntry.isVerified is None or not distillationEntry.isVerified:
                print("Redistilling: " + distillationEntry.question)
                distillationEntry = generateDistillationEntry(
                    modelName=teacher_model_name,
                    question=distillationEntry.question,
                    goldSolution=distillationEntry.gold_solution,
                    schema=distillationEntry.schema,
                    promptTemplate=promptTemplate
                )
            result.append(distillationEntry)
        pd = objects_to_dataframe(result)
        return pd
    except Exception as e:
        print(e)
        pd = objects_to_dataframe(result)
        return pd


def redistillEntries(
        inputFilePath: str,
        outputFilePath: str,
        model_name: str,
        entriesIndex: list[int],
        promptTemplate: str = config["knowledge_distillation_generation_template"]
):
    distillationEntries = loadToObjectsFromFile(inputFilePath, DistillationEntry)
    result = []
    for i in range(len(distillationEntries)):
        if i in entriesIndex:
            distillationEntry = distillationEntries[i]
            distillationEntry = generateDistillationEntry(
                modelName=model_name,
                question=distillationEntry.question,
                goldSolution=distillationEntry.gold_solution,
                schema=distillationEntry.schema,
                promptTemplate=promptTemplate
            )
            print("Redistilling: " + distillationEntry.question)
            print(textwrap.fill("reasoning: " + distillationEntry.reasoning, 160))
            print()
            result.append(distillationEntry)
        else:
            result.append(distillationEntries[i])

    df = objects_to_dataframe(result)
    df.to_csv(outputFilePath)


def generateVanillaEntry(dataset="spider", split="train", batchRange: Optional[tuple[int, int]] = None):
    result = []
    if dataset == "spider":
        spider_instances = load_spider(split, batchRange=batchRange)
        for instance in tqdm(spider_instances):
            db_path = get_spider_db_path(instance.db_id, split=split)
            schema = getDatabaseSchemaForPrompt(db_path)
            distillationEntry = DistillationEntry(
                teacher_model_name="",
                question=instance.question,
                schema=schema,
                gold_solution=instance.query,
                reasoning="",
                verification_solution="",
                isVerified=None
            )
            result.append(distillationEntry)

    elif dataset == "bird":
        bird_instances = load_bird(split, batchRange=batchRange)
        for instance in tqdm(bird_instances):
            db_path = getBirdDbPath(instance.db_id, split=split)
            schema = getDatabaseSchemaForPrompt(db_path)
            distillationEntry = DistillationEntry(
                teacher_model_name="",
                question=instance.question,
                schema=schema,
                gold_solution=instance.query,
                reasoning="",
                verification_solution="",
                isVerified=None
            )
            result.append(distillationEntry)
    df = objects_to_dataframe(result)
    return df



