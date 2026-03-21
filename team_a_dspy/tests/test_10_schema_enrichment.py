from team_a_dspy.optimizers.compile_pipeline import configure_lm
from team_a_dspy.modules.es_query_pipeline import ESQueryDSPyPipeline

configure_lm()

program = ESQueryDSPyPipeline()
pred = program(
    user_question="Top 10 people mentioned in negative news about Iran last week"
)

print("=== ENRICHED SCHEMA ===")
print(pred.effective_schema_context)
print("\n=== FINAL QUERY DSL ===")
print(pred.querydsl_query)