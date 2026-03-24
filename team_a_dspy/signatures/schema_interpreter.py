import dspy

from services.chroma_client import ChromaClient

class SchemaInterpreter(dspy.Signature):
    """
    Analyze a GDELT database field based on its name, type, and real sample values from the last 7 days.
    Provide a clear, concise natural language interpretation of what this field represents 
    and how it should be used in an Elasticsearch query.
    """
    field_name: str = dspy.InputField(desc="Name of the field in the GDELT schema.")
    field_type: str = dspy.InputField(desc="Elasticsearch data type of the field.")
    sample_values: str = dspy.InputField(desc="A list of real values found in this field over the last 7 days.")
    
    interpretation: str = dspy.OutputField(
        desc="A 1-2 sentence explanation of the field and its semantic meaning."
    )

class DataAwareSchemaInterpreter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.interpret = dspy.Predict(SchemaInterpreter)
        
    def forward(self, field_name, field_type, sample_values):
        prediction = self.interpret(field_name=field_name, field_type=field_type, sample_values=sample_values)
        return prediction.interpretation

class PlanSchemaRetriver(dspy.Signature):
    """
    Analyze a user's natural language data request and identify the distinct categories of database fields needed to fulfill it.
    Extract concepts like time, location, numeric metrics, or specific identifiers.
    """
    nl_query: str = dspy.InputField(desc="The user's raw request.")
    
    search_terms: str = dspy.OutputField(
        desc="A comma-separated list ofshort, distinct database terms needed."
    )

class SchemaRetriever(dspy.Module):
    def __init__(self, chroma_client: ChromaClient, k_per_requirement: int = 3):
        super().__init__()
        self.chroma_client = chroma_client
        self.k_per_requirement = k_per_requirement
        self.retrival_planner = dspy.ChainOfThought(PlanSchemaRetriver)
        
    def forward(self, nl_query: str):
        # Retrieve the interpretation from ChromaDB based on the field name
        plan = self.retrival_planner(nl_query=nl_query)
        search_terms = [c.strip() for c in plan.search_terms.split(',')]
        relevant_schema = {}

        for search_term in search_terms:
            results = self.chroma_client.query(query_text=search_term, k=self.k_per_requirement)
            flattened_results = self.flatten_chroma_results(results)
            if flattened_results:
                for item in flattened_results:
                    relevant_schema[item['field_name']] = item
        if not relevant_schema:
            return "No relevant schema information found in the database for the given query."
        
        formatted_passages = []
        for field in relevant_schema.values():
            passage = f"Field Name: {field['field_name']} | Type: {field['field_type']} | Description: {field['interpretation']}"
            formatted_passages.append(passage)
            
        return "\n---\n".join(formatted_passages)

    @staticmethod    
    def flatten_chroma_results(raw_results: dict) -> list[dict]:
        """
        Flattens a raw ChromaDB response into a clean list of schema dictionaries.
        """
        flattened_schema = []
        
        # Check if we have results in the first sub-list
        if not raw_results.get('ids') or not raw_results['ids'][0]:
            return flattened_schema

        # Extract the parallel lists (index 0 because we only passed one query)
        ids = raw_results['ids'][0]
        docs = raw_results['documents'][0]
        metadatas = raw_results['metadatas'][0]
        distances = raw_results['distances'][0]

        # Zip them together to iterate row by row
        for field_id, doc_string, meta, distance in zip(ids, docs, metadatas, distances):
            # Build the clean, flat dictionary
            flattened_schema.append({
                "field_name": meta.get("field_name", field_id),
                "field_type": meta.get("field_type", "unknown"),
                "distance_score": round(distance, 4), # Optional: good for debugging relevance
                "interpretation": doc_string
            })

        return flattened_schema