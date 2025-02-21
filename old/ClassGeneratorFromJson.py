import json
import warnings
import msgspec.json

from models.SQLDataset import SQLDataset


class ClassGenerator:
    def __init__(self, json_file_path, class_name="GeneratedClass", class_type="dataclass"):
        """
        :param json_file_path: Path to the JSON file
        :param class_name: Name of the generated class
        :param class_type: Type of class to generate; supports 'dataclass', 'pydantic', 'standard', 'attrs', and 'typeddict'
        """
        self.json_file_path = json_file_path
        self.class_name = class_name
        self.class_type = class_type.lower()  # Normalize class_type to lowercase
        self.attributes = self._extract_attributes()

        # Map class types to generation methods
        self.class_type_methods = {
            "dataclass": self._generate_dataclass,
            "pydantic": self._generate_pydantic_class,
            "standard": self._generate_standard_class,
            "attrs": self._generate_attrs_class,
            "typeddict": self._generate_typeddict_class
        }

    def _extract_attributes(self):
        # Read the JSON file and extract attributes from the first object
        with open(self.json_file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("The JSON file must contain a list of objects.")

        return data[0].keys()

    def _generate_dataclass(self):
        class_code = "from dataclasses import dataclass\n\n"
        class_code += f"@dataclass\nclass {self.class_name}:\n"
        for attr in self.attributes:
            class_code += f"    {attr}: any = None\n"
        return class_code

    def _generate_pydantic_class(self):
        class_code = "from pydantic import BaseModel\n\n"
        class_code += f"class {self.class_name}(BaseModel):\n"
        for attr in self.attributes:
            class_code += f"    {attr}: any = None\n"
        return class_code

    def _generate_standard_class(self):
        class_code = f"class {self.class_name}:\n"
        class_code += "    def __init__(self, **kwargs):\n"
        for attr in self.attributes:
            class_code += f"        self.{attr} = kwargs.get('{attr}', None)\n"
        return class_code

    def _generate_attrs_class(self):
        class_code = "import attr\n\n"
        class_code += f"@attr.s\nclass {self.class_name}:\n"
        for attr in self.attributes:
            class_code += f"    {attr} = attr.ib(default=None)\n"
        return class_code

    def _generate_typeddict_class(self):
        class_code = "from typing import TypedDict\n\n"
        class_code += f"class {self.class_name}(TypedDict, total=False):\n"
        for attr in self.attributes:
            class_code += f"    {attr}: any\n"
        return class_code

    def generate_class_code(self):
        # Select the appropriate method for the given class_type
        if self.class_type not in self.class_type_methods:
            raise ValueError(
                "Unsupported class type. Use 'dataclass', 'pydantic', 'standard', 'attrs', or 'typeddict'.")
        return self.class_type_methods[self.class_type]()

    def save_class_to_file(self, class_code):
        file_name = f"{self.class_name}.py"
        with open(file_name, "w") as file:
            file.write(class_code)
        print(f"Class code has been written to {file_name}")

    def generate(self, to_file=False):
        # Generate the class code
        class_code = self.generate_class_code()
        if to_file:
            self.save_class_to_file(class_code)
        else:
            print(class_code)

    @staticmethod
    def load_json_to_class(json_file_path, cls):
        # Load JSON data and convert it to instances of the class
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("The JSON file must contain a list of objects.")

        temp_instance = cls()
        class_attributes = temp_instance.__dict__.keys()

        instances = []
        for obj_data in data:
            filtered_data = {k: v for k, v in obj_data.items() if k in class_attributes}
            missing_attrs = set(class_attributes) - set(obj_data.keys())
            for missing in missing_attrs:
                filtered_data[missing] = None
                warnings.warn(f"Attribute '{missing}' is missing in JSON data and will be set to None.")

            instance = cls(**filtered_data)
            instances.append(instance)

        return instances

# # Example usage:
# generator = ClassGenerator('sharegpt.json', 'ShareGPTFormat', class_type='dataclass')
# generator.generate(to_file=True)

instances = ClassGenerator.load_json_to_class('../old/mini_dev_sqlite.json', SQLDataset)
# for instance in instances:
#     print(instance)

serializedJson = msgspec.json.encode(instances)
with open('old/bird_minidev_clean.json', 'wb') as json_file:  # Use 'wb' for binary writing
    json_file.write(serializedJson)

# print(len(instances))