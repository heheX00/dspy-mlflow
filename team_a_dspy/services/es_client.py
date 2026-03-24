from services.config import settings

from elasticsearch import Elasticsearch

SAMPLE_SIZE_PER_DAY = 5

class ESClient:
    """
    Client for interacting with Elasticsearch.
    This is used to store and retrieve raw gdelt metadata fields for filtering and retrieval.
    """
    def __init__(self, host: str, username: str, password: str, index: str, verify_ssl: bool) -> None:
        self.es = Elasticsearch(
            hosts=[host],
            basic_auth=(username, password),
            verify_certs=verify_ssl,
        )
        self.index = index
    
    def get_mapping(self):
        return self.es.indices.get_mapping(index=self.index)
    
    def get_last_x_days_samples(self, days: int, date_field: str = "@timestamp") -> list[dict]:
        """
        Pulls 20 random documents for each of the past 7 days using the @timestamp field for date filtering.
        """
        results = []
        for days_ago in range(1, days + 1):
            start_date = f"now-{days_ago}d/d"
            end_date = f"now-{days_ago - 1}d/d"
            query = {
                "size": SAMPLE_SIZE_PER_DAY,
                "query": {
                    "function_score": {
                        "query": {
                            "range": {
                                date_field: {
                                    "gte": start_date,
                                    "lt": end_date
                                }
                            }
                        },
                        # random_score assigns a random value to every matching document
                        # before picking the top 20
                        "random_score": {}
                    }
                }
            }
            results.append(self.es.search(index=self.index, body=query))
        
        all_sampled_docs = []
        for day_result in results:
            hits = day_result.get("hits", {}).get("hits", [])
            all_sampled_docs.extend([hit["_source"] for hit in hits])
        return all_sampled_docs
    
    def flatten_es_mapping(self) -> dict:
        """
        Flattens an Elasticsearch mapping payload into a {"field.path": "type"} dictionary.
        """
        flat_fields = {}

        # Since the payload contains multiple indices (e.g., 'gkg-2026.03.16', 'gkg-2026.03.07'),
        # we just need to grab the properties from the FIRST index. 
        # (Assuming all daily indices share the same mapping).
        mapping = self.get_mapping()
        first_index = list(mapping.keys())[0] # type: ignore
        
        # Drill down to the actual 'properties' block
        try:
            properties = mapping[first_index]['mappings']['properties']
        except KeyError:
            return {"error": "Invalid mapping structure."}

        # Recursive function to walk the tree
        def extract_properties(prop_dict, current_path=""):
            for field_name, field_info in prop_dict.items():
                
                # Construct the current field path (e.g., "V15Tone.Polarity")
                new_path = f"{current_path}.{field_name}" if current_path else field_name

                # 1. If it has 'properties', it's a nested object. Recurse deeper.
                if 'properties' in field_info:
                    extract_properties(field_info['properties'], new_path)
                
                # 2. Otherwise, it's a concrete field. Grab its type.
                else:
                    field_type = field_info.get('type', 'unknown')
                    
                    # Special handling for text fields with keyword sub-fields
                    # (e.g., "V2Persons.V1Person" vs "V2Persons.V1Person.keyword")
                    if field_type == 'text' and 'fields' in field_info and 'keyword' in field_info['fields']:
                        # We store the main text field
                        flat_fields[new_path] = field_type
                        # And we also store the .keyword sub-field
                        flat_fields[f"{new_path}.keyword"] = 'keyword'
                    else:
                        flat_fields[new_path] = field_type

        # Start the recursion
        extract_properties(properties)
        
        return flat_fields
    
    def search(self, query_dsl: dict) -> dict:
        """
        Executes a search query against Elasticsearch using the provided Query DSL.
        
        Args:
            query_dsl (dict): The Elasticsearch Query DSL to be executed.
        Returns:
            dict: The search results returned by Elasticsearch.
        """
        query = query_dsl.get("query_dsl", {})
        response = self.es.search(index=self.index, body=query)
        return response.body
    
    def close(self):
        """
        Closes the Elasticsearch client connection.
        """
        self.es.close()