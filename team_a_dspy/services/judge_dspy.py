from services.sandbox_es_client import SandboxESClient

class JudgeDSPY:
    """
    JudgeDSPY is a specialized client that inherits from DSPYClient and is designed to evaluate the quality of generated Elasticsearch Query DSL against a natural language query and the ES schema.
    It uses a language model to compare the generated Query DSL with the expected output based on the natural language query and the ES schema, and provides feedback on the accuracy and relevance of the generated Query DSL.
    """
    def __init__(
            self,
            sandbox_es_client: SandboxESClient,
    ):
        self.es_client = sandbox_es_client
    
    def evaluate_query_dsl(self, generated_query_dsl: dict) -> dict:
        """
        Evaluates the generated Query DSL against the natural language query and the ES schema.
        
        Args:
            query_text (str): The natural language query.
            generated_query_dsl (dict): The generated Elasticsearch Query DSL to be evaluated.
        Returns:
            dict: A dictionary containing the evaluation results, including accuracy, relevance, and feedback for improvement.
        """
        return self.es_client.validate_query_dsl(query_dsl=generated_query_dsl)