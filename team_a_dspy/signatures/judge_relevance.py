import dspy


class RelevanceScorerSignature(dspy.Signature):
    """
    Evaluate how relevant the aggregated search results are to the original natural language query.
    Consider:
    1. Entity Coverage: Did we find persons, organizations, and locations that match the query intent?
    2. Theme Alignment: Are the extracted themes semantically related and relevant to what was asked?
    3. Tone Alignment: Does the sentiment/tone of results align with expectations (e.g., negative for crisis-related queries)?
    4. Overall Relevance: How well do the aggregated results answer the original question?
    
    Provide a relevance score from 0-100 where:
    - 0-30: Results are not relevant or off-topic
    - 30-60: Results are partially relevant but missing key aspects
    - 60-80: Results are mostly relevant with minor gaps
    - 80-100: Results are highly relevant and comprehensive
    """
    nl_query: str = dspy.InputField(desc="The original natural language query.")
    aggregation_summary: str = dspy.InputField(desc="JSON summary of aggregated search results including top entities, themes, and tone.")
    
    relevance_score: int = dspy.OutputField(desc="A score from 0-100 indicating the relevance of results to the query.")
    entity_coverage: str = dspy.OutputField(desc="Assessment of whether relevant persons, organizations, and locations were found.")
    theme_alignment: str = dspy.OutputField(desc="Assessment of whether themes align with the query intent.")
    tone_alignment: str = dspy.OutputField(desc="Assessment of whether sentiment/tone aligns with query expectations.")
    reasoning: str = dspy.OutputField(desc="Detailed explanation of the relevance score and key findings.")