from __future__ import annotations

from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from team_a_dspy.data.bootstrap_examples import load_examples, split_train_dev
from team_a_dspy.metrics.es_sandbox_metric import ESSandboxJudge
from team_a_dspy.modules.es_query_pipeline import ESQueryDSPyPipeline
from team_a_dspy.utils.config import settings


def configure_lm() -> None:
    lm = dspy.LM(
        model=settings.llm_model_name,
        api_base=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=1800,
    )
    dspy.settings.configure(lm=lm)


def compile_pipeline(
    examples_path: str = "team_a_dspy/data/examples_seed.jsonl",
    save_dir: str = "team_a_dspy/optimizers/compiled",
):
    configure_lm()

    examples = load_examples(examples_path)
    trainset, devset = split_train_dev(examples, dev_ratio=0.2)

    judge = ESSandboxJudge(index_name=settings.es_index)

    # Fresh base student
    base_program = ESQueryDSPyPipeline()

    # Step 1: Bootstrap a separate teacher program
    bootstrap = BootstrapFewShot(
        metric=judge.metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=6,
    )
    teacher_program = bootstrap.compile(
        student=ESQueryDSPyPipeline(),
        trainset=trainset,
    )

    # Step 2: MIPRO must receive an UNCOMPILED student
    fresh_student = ESQueryDSPyPipeline()

    mipro = MIPROv2(
        metric=judge.metric,
        auto="light",
    )
    optimized_program = mipro.compile(
        student=fresh_student,
        teacher=teacher_program,
        trainset=trainset,
        valset=devset,
        max_bootstrapped_demos=4,
        max_labeled_demos=6,
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(save_path / "es_query_dsl_pipeline.json"))

    return optimized_program, trainset, devset


if __name__ == "__main__":
    compile_pipeline()