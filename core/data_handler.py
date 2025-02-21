import math
import os.path

from DatasetFormat.AlpacaFormat import AlpacaFormat
from core.utils import config, loadToObjectsFromFile, objects_to_dataframe, suppress_prints
from models.DistillationEntry import DistillationEntry

@suppress_prints
def distillationToAlpaca(
        distillationEntries: list[DistillationEntry],
        alpaca_instruction_template: str = config["alpaca_instruction_template"]
):
    alpaca = []
    for i in range(len(distillationEntries)):
        distillationEntry = distillationEntries[i]
        if type(distillationEntry.reasoning) is not str:
            print(i)
            print(distillationEntry.reasoning)

        # if distillationEntry.reasoning is None or distillationEntry.reasoning == "" or math.isnan(distillationEntry.reasoning):
        #     reasoning_with_answer = distillationEntry.gold_solution
        # else:
        reasoning_with_answer = str(distillationEntry.reasoning) + "\n<final answer>" + distillationEntry.gold_solution + "</final answer>"

        instruction = alpaca_instruction_template.format(request=distillationEntry.question, schema=distillationEntry.schema)
        alpaca.append(
            AlpacaFormat(instruction=instruction, output=reasoning_with_answer)
        )
    return alpaca

def convertDistillationEntriesToAlpaca(
        inputFilePath: str,
        outputFilePath: str,
        alpaca_instruction_template: str = config["alpaca_instruction_template"],
        outputExtension: str = "json"
):
    distillationEntries = loadToObjectsFromFile(inputFilePath, DistillationEntry)
    alpaca = distillationToAlpaca(distillationEntries, alpaca_instruction_template)
    output = objects_to_dataframe(alpaca)

    outputFilePath = os.path.splitext(outputFilePath)[0]

    if outputExtension == "json":
        outputFilePath += ".json"
        output.to_json(outputFilePath, orient="records", indent=4)
    elif outputExtension == "csv":
        outputFilePath += ".csv"
        output.to_csv(outputFilePath)
    return alpaca
