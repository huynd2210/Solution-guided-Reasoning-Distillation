import instructor
import ollama
from openai import OpenAI

from core.data_loader import load_spider
from core.evaluation import evaluateSQLGenerationEntry, evaluateSQL
from core.generation import generateSQLEvaluationEntry
from core.prompt_delivery import Prompt
from core.sql_tools import retrieveDatabaseSchema, formatSchemaForPrompt
from core.utils import config
from models.SQLQuery import SQLQuery
from transformers import pipeline


def test():
    train_data = load_spider("train")
    generated_sql = "SELECT count(*) FROM head WHERE age  >  53"
    sqlEvaluationEntry = generateSQLEvaluationEntry(generated_sql, train_data[0])
    evaluateSQLGenerationEntry(sqlEvaluationEntry)


def testSchemaRetrieval():
    db_id = "department_management"
    sample_db_path = config["spider_root_path"] + "/database/" + db_id + "/" + db_id + ".sqlite"
    db_info = retrieveDatabaseSchema(db_path=sample_db_path, include_sample_data=True)

    print(db_info)
    print("_" * 20)

    # print_schema(db_info)
    # print("_" * 20)

    print(formatSchemaForPrompt(db_info))


def testInstructor():
    # model = "qwen2.5-coder:7b-instruct"
    model = "llama3.1:8b-instruct-q4_0"

    promptTemplate = config["prompt_template_instructor"]

    question = "How many heads of the departments are older than 56 ?"
    schema = """
    SCHEMA:
- Table: department
  - Column: Department_ID
    - Samples: [1, 2, 3, 4, 5]
  - Column: Name
    - Samples: [State, Treasury, Defense, Justice, Interior]
  - Column: Creation
    - Samples: [1789, 1789, 1947, 1870, 1849]
  - Column: Ranking
    - Samples: [1, 2, 3, 4, 5]
  - Column: Budget_in_Billions
    - Samples: [9.96, 11.1, 439.3, 23.4, 10.7]
  - Column: Num_Employees
    - Samples: [30266.0, 115897.0, 3000000.0, 112557.0, 71436.0]
- Table: head
  - Column: head_ID
    - Samples: [1, 2, 3, 4, 5]
  - Column: name
    - Samples: [Tiger Woods, Sergio García, K. J. Choi, Dudley Hart, Jeff Maggert]
  - Column: born_state
    - Samples: [Alabama, California, Alabama, California, Delaware]
  - Column: age
    - Samples: [67.0, 68.0, 69.0, 52.0, 53.0]
- Table: management
  - Column: department_ID
    - Samples: [2, 15, 2, 7, 11]
  - Column: head_ID
    - Samples: [5, 4, 6, 3, 10]
  - Column: temporary_acting
    - Samples: [Yes, Yes, Yes, No, No]
    """

    message = promptTemplate.format(request=question, schema=schema)
    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        ),
        mode=instructor.Mode.JSON,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        response_model=SQLQuery,
    )
    print(resp.model_dump_json(indent=2))
    print("_" * 20)
    print("SQL: ", resp.sql_query)
    print("Reasoning: ", resp.reasoning)


def testLLM():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained("NyanDoggo/Qwen2.5-Coder-3B-Instruct-Spider-Vanilla",
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("NyanDoggo/Qwen2.5-Coder-3B-Instruct-Spider-Vanilla")

    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    prompt = prompt.format(
        "Solve the following sql problem: How many heads of the departments are older than 56 ? Schema: SCHEMA: - Table: department - Column: Department_ID - Samples: [1, 2, 3, 4, 5] - Column: Name - Samples: [State, Treasury, Defense, Justice, Interior] - Column: Creation - Samples: [1789, 1789, 1947, 1870, 1849] - Column: Ranking - Samples: [1, 2, 3, 4, 5] - Column: Budget_in_Billions - Samples: [9.96, 11.1, 439.3, 23.4, 10.7] - Column: Num_Employees - Samples: [30266.0, 115897.0, 3000000.0, 112557.0, 71436.0] - Table: head - Column: head_ID - Samples: [1, 2, 3, 4, 5] - Column: name - Samples: [Tiger Woods, Sergio García, K. J. Choi, Dudley Hart, Jeff Maggert] - Column: born_state - Samples: [Alabama, California, Alabama, California, Delaware] - Column: age - Samples: [67.0, 68.0, 69.0, 52.0, 53.0] - Table: management - Column: department_ID - Samples: [2, 15, 2, 7, 11] - Column: head_ID - Samples: [5, 4, 6, 3, 10] - Column: temporary_acting - Samples: [Yes, Yes, Yes, No, No]",
        # instruction
        "",  # input
        "",  # output - leave this blank for generation!
    )

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def testPromptDelivery():
    model_name = "NyanDoggo/Qwen2.5-Coder-3B-Instruct-Spider-Vanilla"
    request = "How many heads of the departments are older than 56 ?"
    schema = """
     Schema: SCHEMA: - Table: department - Column: Department_ID - Samples: [1, 2, 3, 4, 5] - Column: Name - Samples: [State, Treasury, Defense, Justice, Interior] - Column: Creation - Samples: [1789, 1789, 1947, 1870, 1849] - Column: Ranking - Samples: [1, 2, 3, 4, 5] - Column: Budget_in_Billions - Samples: [9.96, 11.1, 439.3, 23.4, 10.7] - Column: Num_Employees - Samples: [30266.0, 115897.0, 3000000.0, 112557.0, 71436.0] - Table: head - Column: head_ID - Samples: [1, 2, 3, 4, 5] - Column: name - Samples: [Tiger Woods, Sergio García, K. J. Choi, Dudley Hart, Jeff Maggert] - Column: born_state - Samples: [Alabama, California, Alabama, California, Delaware] - Column: age - Samples: [67.0, 68.0, 69.0, 52.0, 53.0] - Table: management - Column: department_ID - Samples: [2, 15, 2, 7, 11] - Column: head_ID - Samples: [5, 4, 6, 3, 10] - Column: temporary_acting - Samples: [Yes, Yes, Yes, No, No]
    """

    prompt = Prompt(modelName=model_name, promptTemplate=config["alpaca_inference_template"], request=request, schema=schema)
    response = prompt.deliver()
    print(response)


def buildSCPCommand(remoteHost, remotePath):
    cmd = f"scp -i private_key.pem -r {remoteHost}:{remotePath} ./"
    print(cmd)
    return cmd

def testSQLEvaluationCorrectness():
    db_path = "spider_data/test_database/customers_and_orders/customers_and_orders.sqlite"
    predicted_sql = "SELECT avg(product_price) ,  product_type_code FROM products GROUP BY product_type_code"
    ground_truth_sql = "SELECT product_type_code ,  avg(product_price) FROM Products GROUP BY product_type_code"
    return evaluateSQL(predicted_sql, ground_truth_sql, db_path)
testSQLEvaluationCorrectness()

# remoteHost = ""
# remotePath = ""
# buildSCPCommand(remoteHost, remotePath)

# testLLM()
# testPromptDelivery()

# testInstructor()

# testSchemaRetrieval()
# load_spider("train")
# test()
# print(config['prompt_template'])
# print(ollama.generate(model="llama3.1:latest", prompt="what is the answer to life, the universe, and everything?")['response'])
# spider_instances = load_spider("test")

