import sqlite3
import time

import groq
import pandas as pd
from icecream import ic
from sqlalchemy import except_
from tqdm import tqdm

from core.data_loader import load_spider, load_bird
from core.generation import generateSQLEvaluationEntry, generateSQLEvaluationEntryFromLoadedDataset
from core.prompt_delivery import APIKeyManager
from core.utils import objects_to_dataframe, loadToObjectsFromFile, config
from models.SQLEvaluationEntry import SQLEvaluationEntry


def evaluateSQL(predicted_sql, ground_truth, db_path):
    """
    :param predicted_sql: Predicted sql
    :param ground_truth: Ground truth sql
    :param db_path: path of the db
    :return:
    """
    conn = sqlite3.connect(db_path)
    # Connect to the database
    try:
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()

        are_equal = set(map(frozenset, predicted_res)) == set(map(frozenset, ground_truth_res))
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    return are_equal

def evaluateSQLGenerationEntry(evaluation_entry: SQLEvaluationEntry) -> SQLEvaluationEntry:
    """
    Evaluate a generated SQL query against a gold (reference) SQL query by executing both
    on the specified SQLite database and comparing their results.

    Parameters:
    - evaluation_entry (SQLEvaluationEntry): The evaluation entry to evaluate.
    - conn (sqlite3.Connection): The SQLite connection to use. If None, a new connection will be created.
    - close_conn (bool): Whether to close the connection after evaluation.
    Returns:
    - SQLEvaluationEntry: The evaluation entry containing the results of the evaluation.
    """
    print("----------RECAP----------")
    print("Question: " + evaluation_entry.question)
    print("Gold SQL: " + evaluation_entry.gold_sql)
    print("Generated SQL: " + evaluation_entry.generated_sql)

    try:
        isCorrect = evaluateSQL(evaluation_entry.generated_sql, evaluation_entry.gold_sql, evaluation_entry.db_path)

        evaluation_entry.isCorrect = isCorrect
    except Exception as e:
        print(f"An error occurred: {e}")
        evaluation_entry.isCorrect = False
        evaluation_entry.generated_sql = evaluation_entry.generated_sql + "\n" + str(e)
    return evaluation_entry


def evaluateModel(
        model_name: str,
        batchRange=None,
        datasetName="spider",
        split="train",
        promptTemplate=config["prompt_template"],
):
    print("Evaluating model: " + model_name)
    result = []

    if datasetName == "spider":
        spider_instances = load_spider(split, batchRange=batchRange)
        result = generateSQLEvaluationEntryFromLoadedDataset(model_name, datasetName, split, promptTemplate, spider_instances)
    elif datasetName == "bird":
        bird_instances = load_bird(split, batchRange=batchRange)
        result = generateSQLEvaluationEntryFromLoadedDataset(model_name, datasetName, split, promptTemplate, bird_instances)
        
    # Convert result to pandas dataframe
    df = objects_to_dataframe(result)
    return df


def evaluateFromFile(filePath: str):
    result = []
    sqlEvalEntries = loadToObjectsFromFile(filePath, SQLEvaluationEntry, file_type="csv")
    for sqlEvalEntry in tqdm(sqlEvalEntries):
        result.append(evaluateSQLGenerationEntry(sqlEvalEntry))
    # Convert result to pandas dataframe
    df = objects_to_dataframe(result)
    return df
