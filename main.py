import pprint
import textwrap
import time
from typing import Optional

import pandas as pd
from icecream import ic

from DatasetFormat.AlpacaFormat import AlpacaFormat
from core.data_handler import distillationToAlpaca, convertDistillationEntriesToAlpaca
from core.distillation import distillKnowledge, distillUnverifiedEntries, redistillEntries, generateVanillaEntry
from core.evaluation import evaluateModel
from core.post_processing import reevaluateCSVResult
from core.prompt_delivery import Prompt
from core.utils import merge_csv_files, load_json_to_class, loadToObjectsFromFile, config
from models.DistillationEntry import DistillationEntry
from models.SQLEvaluationEntry import SQLEvaluationEntry


def analyseEvaluation(evaluationResultDf: pd.DataFrame):
    counter = 0
    for index, row in evaluationResultDf.iterrows():
        if row["isCorrect"]:
            counter += 1

    print("Accuracy: " + str(counter / len(evaluationResultDf)))

def distillWrapper(
        model_name: str,
        student_model_name: str = None,
        dataset="spider",
        split="train",
        batchRange: Optional[tuple[int, int]] = None,
        isBatchMode=False
):
    data = distillKnowledge(model_name, student_model_name, dataset=dataset, split=split, batchRange=batchRange, isBatchMode=isBatchMode)
    if batchRange is not None:
        outputName = f"{model_name.replace(':', '-')}_distilled_data_{dataset}_{split}_{batchRange[0]}_{batchRange[1]}.csv"
    else:
        outputName = f"{model_name.replace(':', '-')}_distilled_data_{dataset}_{split}.csv"
    print(f"Output saved to {outputName}")
    data.to_csv(outputName)

def generateVanillaData(dataset, split):
    data = generateVanillaEntry(dataset=dataset, split=split)
    outputName = f"vanilla_data_{dataset}_{split}.csv"
    print(f"Output saved to {outputName}")
    data.to_csv(outputName)


def redistillWrapper(
        file_path: str,
        model_name: str,
):
    pd = distillUnverifiedEntries(file_path, model_name)
    outputName = file_path.replace(".csv", "_redistilled.csv")
    print(f"Output saved to {outputName}")
    pd.to_csv(outputName)

if __name__ == '__main__':
    pass
    """
    Generation of vanilla data
    """
    # generateVanillaData(dataset="bird", split="train")

    """
    Util function to merge csv files
    """
    # merge_csv_files("temp", "gemma2-9b-it_spider_test_result.csv")

    """
    In order to finetune the model, we need to convert the distilled data to alpaca format
    """
    # convertDistillationEntriesToAlpaca(
    #     inputFilePath="datasets/distilled_bird/bird_train_distilled.csv",
    #     outputFilePath="bird_train_distilled-alpaca.json",
    #     outputExtension="json"
    # )


    """
    Distillation code, set batchRange to None if you want to distill the whole dataset
    """
    # batchList = [
    #     # (8901, 9000),
    #     (9001, 9100),
    #     (9101, 9200),
    #     (9201, 9300),
    #     (9301, 9400),
    #     (9401, 9427),
    # ]
    # for batchRange in batchList:
    #     distillWrapper(model_name="gpt-4o", dataset="bird", split="train", batchRange=batchRange)

    """
    Evaluation code, set batchRange to None if you want to evaluate the whole dataset
    """
    # models = [
    #     "NyanDoggo/Qwen2.5-Coder-3B-Instruct-Bird-Reasoning",
    # ]
    # for model_name in models:
    #     print("Running")
    #     split="minidev"
    #     datasetName = "bird"
    #     # batchRange=(0,10)
    #     batchRange=None
    #     result = evaluateModel(
    #         model_name,
    #         batchRange=batchRange,
    #         datasetName=datasetName,
    #         split=split,
    #         promptTemplate=config["alpaca_inference_template"],
    #     )
    #     analyseEvaluation(result)
    #     print("----RESULT----")
    #     print(result)
    #     model_name = model_name.replace("/", "-")
    #     if batchRange is not None:
    #         outputName = f"{model_name.replace(':', '-')}_{datasetName}__{batchRange[0]}_{batchRange[1]}_result.csv"
    #     else:
    #         outputName = f"{model_name.replace(':', '-')}_{datasetName}__{split}_result.csv"
    #     print(f"Output saved to {outputName}")
    #     result.to_csv(outputName)

    """
    Reevaluate CSV results in case of errors/interruption
    """
    # fileList = [
    #     "Qwen-Qwen2.5-Coder-3B-Instruct_bird__minidev_result.csv",
    # ]
    # for file_path in fileList:
    #     reevaluateCSVResult(
    #         file_path=file_path,
    #         targetColumn="response",
    #         retrievalModel="meta-llama/llama-3.1-8b-instruct"
    #     )

    """
    Redistill entries, in case of error in some entries, instead of re-distilling the whole dataset
    """
    # entriesIndex = 0
    # redistillEntries(
    #     inputFilePath="datasets/distilled_spider_train/spider-train-distilled.csv",
    #     outputFilePath="datasets/distilled_spider_train/spider-train-redistilled.csv",
    #     model_name="gpt-4o",
    #     entriesIndex=entriesIndex
    # )




    # inputFilePath = "datasets/distilled_spider_train/spider-train-redistilled-alpaca.csv"
    # distillationEntries = loadToObjectsFromFile(inputFilePath, AlpacaFormat)
    # print(distillationEntries[0])