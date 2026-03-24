import json
import dspy

class DSPYOptimiser:
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def load_examples(self) -> list[dspy.Example]:
        with open(self.filepath, "r") as f:
            data = json.load(f)
        print(data)