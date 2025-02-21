from core.utils import save_list_to_jsonl


class Request:
    def __init__(self, custom_id, method, url, body):
        self.custom_id = custom_id
        self.method = method
        self.url = url
        self.body = body

    def __dict__(self):
        return {"custom_id": self.custom_id, "method": self.method, "url": self.url, "body": self.body.__dict__()}

    def to_dict(self):
        return {"custom_id": self.custom_id, "method": self.method, "url": self.url, "body": self.body.to_dict()}

class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __dict__(self):
        return {"role": self.role, "content": self.content}

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class Body:
    def __init__(self, model, messages, max_tokens):
        self.model = model
        self.messages = [message.__dict__() for message in messages]
        self.max_tokens = max_tokens

    def __dict__(self):
        return {"model": self.model, "messages": self.messages, "max_tokens": self.max_tokens}

    def to_dict(self):
        return {"model": self.model, "messages": self.messages, "max_tokens": self.max_tokens}

class BatchRequestController:
    @classmethod
    def _create_jsonl_requests(cls, requests, fileName="batch_requests.jsonl"):
        data_list = [req.to_dict() for req in requests]
        save_list_to_jsonl(data_list, fileName)

    @classmethod
    def upload(cls, requests):
        fileName = "batch_requests.jsonl"
        cls._create_jsonl_requests(requests, fileName=fileName)
        from openai import OpenAI
        client = OpenAI()

        batch_input_file = client.files.create(
            file=open(fileName, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )
        print("Batch request uploaded")
        print(batch)
        # Ensure batch is serializable
        if hasattr(batch, 'to_dict'):
            batch_dict = batch.to_dict()
        else:
            batch_dict = batch  # Assuming batch is already a dict
        save_list_to_jsonl([batch_dict], "batch.jsonl")