import os
from dataclasses import fields
import yaml
from icecream import ic

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

import json
import warnings
import pandas as pd
import functools
import sys
from io import StringIO
from contextlib import contextmanager


def objects_to_dataframe(objects):
    # Convert a list of objects to a list of dictionaries
    data = [vars(obj) for obj in objects]
    # Create and return a DataFrame
    return pd.DataFrame(data)


# Generalized function to create an object from a class and row data
def create_object(class_type, row):
    # Map row data to the object's fields
    field_values = {field.name: row[field.name] for field in fields(class_type)}
    # Convert the field values (e.g., strings to integers/floats)
    for key, value in field_values.items():
        field_type = next(field.type for field in fields(class_type) if field.name == key)
        if field_type == int:
            field_values[key] = int(value)
        elif field_type == float:
            field_values[key] = float(value)
    return class_type(**field_values)


# Function to load data from different sources and convert it to a DataFrame
def load_data(file_path: str, file_type: str = 'csv') -> pd.DataFrame:
    """
    This function loads data from a file and converts it to a pandas DataFrame.
    Supports CSV, JSON, and other formats supported by pandas.
    """
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# Function to load data, convert to DataFrame, and then convert DataFrame rows to objects
def loadToObjectsFromFile(file_path: str, class_type, file_type: str = 'csv') -> list:
    # Load data from the file and convert to DataFrame
    df = load_data(file_path, file_type)

    # Convert each row of the DataFrame to an object of the specified class type
    objects = [create_object(class_type, row) for _, row in df.iterrows()]
    return objects

def save_list_to_jsonl(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            json.dump(item, f)
            f.write('\n')
    return True


def load_json_to_class(json_file_path, cls, key_mapping=None):
    """
    Load data from a JSON file into instances of a given class.

    Args:
        json_file_path (str): Path to the JSON file.
        cls (type): Class to map the data to.
        key_mapping (dict, optional): Mapping of JSON keys to class attributes.

    Returns:
        list: A list of instances of the class populated with data from the JSON file.
    """
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        ic(json_file_path)
        data = json.load(f)

    # Ensure the JSON file has the correct structure (a list of objects)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("The JSON file must contain a list of objects.")

    # Extract attribute names from the class by instantiating an empty instance
    temp_instance = cls()
    class_attributes = temp_instance.__dict__.keys()

    # Initialize the mapping dictionary if not provided
    if key_mapping is None:
        key_mapping = {}

    # List to store instances of the class
    instances = []

    for obj_data in data:
        # Create a mapped version of the JSON data based on the key_mapping
        mapped_data = {
            key_mapping.get(k, k): v  # Use the mapped key if it exists, otherwise use the original key
            for k, v in obj_data.items()
        }

        # Filter mapped JSON data to only include keys matching class attributes
        filtered_data = {k: v for k, v in mapped_data.items() if k in class_attributes}

        # Check for missing attributes and set them to None
        missing_attrs = set(class_attributes) - set(filtered_data.keys())
        for missing in missing_attrs:
            filtered_data[missing] = None
            warnings.warn(f"Attribute '{missing}' is missing in JSON data and will be set to None.")

        # Create an instance of the class with the filtered data
        instance = cls(**filtered_data)
        instances.append(instance)

    return instances

def generate_class_from_json(json_file_path, class_name="GeneratedClass", to_file=False):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Ensure the JSON file has the correct structure (a list of objects)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("The JSON file must contain a list of objects.")

    # Get the keys of the first object to define the class attributes
    attributes = data[0].keys()

    # Generate class definition code
    class_code = f"class {class_name}:\n"
    class_code += "    def __init__(self, **kwargs):\n"

    # Add attributes to the class's __init__ method
    for attr in attributes:
        class_code += f"        self.{attr} = kwargs.get('{attr}', None)\n"

    if to_file:
        # Save the generated class to a .py file
        file_name = f"{class_name}.py"
        with open(file_name, "w") as file:
            file.write(class_code)
        print(f"Class code has been written to {file_name}")
    else:
        # Print the generated class code
        print(class_code)


@contextmanager
def suppress_output():
    """Context manager to temporarily suppress stdout."""
    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout

def suppress_prints(func):
    """
    Decorator that suppresses all print statements in the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with suppress_output():
            return func(*args, **kwargs)
    return wrapper

def cleanLLMResponse(response, openTag="<final answer>", closeTag="</final answer>"):
    response = response.replace("```sql", "").replace("\n", " ").replace("```", "").strip()

    if openTag in response:
        response = response.split(openTag)[1].strip()

    if closeTag in response:
        response = response.split(closeTag)[0].strip()

    #order is important
    to_remove = ["<sql>", "</sql>", "```sql", "```", "\n", "<final answer>", "</final answer>", "<final_answer>", "</final_answer>"]
    for string in to_remove:
        response = response.replace(string, "")

    return response

def merge_csv_files(input_directory, output_file):
    """
    Merges all CSV files in the specified directory into a single CSV file.

    Parameters:
        input_directory (str): Path to the directory containing CSV files.
        output_file (str): Path to save the merged CSV file.

    Returns:
        None
    """
    # List to store individual dataframes
    dataframes = []

    # Loop through all CSV files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged dataframe to a CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged CSV saved to {output_file}")

# instances = load_json_to_class('train_spider_clean.json', SpiderDataset)
# for instance in instances:
#     print(instance)