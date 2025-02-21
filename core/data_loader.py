from typing import Optional

from core.utils import load_json_to_class, suppress_prints, config
from models.SQLDataset import SQLDataset


@suppress_prints
def load_spider(split: str = "train", batchRange: Optional[tuple[int, int]] = None) -> list[SQLDataset]:
    batchResult = []
    splitFilePath = {
        "train": 'old/train_spider_clean.json',
        "dev": 'old/dev_spider_clean.json',
        "test": 'old/test_spider_clean.json',
        "others": 'old/train_spider_others_clean.json'
    }
    jsonFilePath = splitFilePath[split]
    instances = load_json_to_class(jsonFilePath, SQLDataset)

    if batchRange is not None:
        print("Loading batch range: " + str(batchRange))

    for i in range(len(instances)):
        instance = instances[i]
        if batchRange is None:
            batchResult.append(instance)
        elif i in range(batchRange[0], batchRange[1] + 1):
            batchResult.append(instance)
        else:
            print("Skipping instance " + str(i))
        # print(instance)

    if batchRange is not None:
        instances = batchResult
    return instances

def load_bird(split: str = "train", batchRange: Optional[tuple[int, int]] = None) -> list[SQLDataset]:
    batchResult = []
    splitFilePath = {
        "train": 'old/train_bird.json',
        "dev": 'old/dev_bird.json',
        "minidev": 'old/bird_minidev_clean.json'
    }
    jsonFilePath = splitFilePath[split]
    instances = load_json_to_class(jsonFilePath, SQLDataset, {"SQL": "query"})
    if batchRange is not None:
        print("Loading batch range: " + str(batchRange))
    for i in range(len(instances)):
        instance = instances[i]
        if batchRange is None:
            batchResult.append(instance)
        elif i in range(batchRange[0], batchRange[1] + 1):
            batchResult.append(instance)
        else:
            print("Skipping instance " + str(i))

    if batchRange is not None:
        instances = batchResult
    return instances

def get_spider_db_path(db_id, spiderRootPath=config["spider_root_path"], split="train"):
    if split == "train" or split == "others":
        return f"{spiderRootPath}/database/{db_id}/{db_id}.sqlite"
    else:
        return f"{spiderRootPath}/test_database/{db_id}/{db_id}.sqlite"

def getBirdDbPath(db_id, split="train"):
    birdRootPath = config["bird_root_path"]
    birdDevPath = config["bird_dev_path"]
    birdMinidevPath = config["bird_minidev_path"]
    dbPathMap = {
        "train": f"{birdRootPath}/{db_id}/{db_id}.sqlite",
        "dev": f"{birdDevPath}/{db_id}/{db_id}.sqlite",
        "minidev": f"{birdMinidevPath}/{db_id}/{db_id}.sqlite"
    }
    return dbPathMap[split]

def getDbPath(dataset, db_id, split):
    if dataset == "spider":
        return get_spider_db_path(db_id, split=split)
    elif dataset == "bird":
        return getBirdDbPath(db_id, split=split)