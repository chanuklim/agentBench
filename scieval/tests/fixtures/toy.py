from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def toy():
    return Task(
        dataset=[
            Sample(input="Say the word apple.", target="apple"),
            Sample(input="Say the word banana.", target="banana"),
        ],
        solver=[generate()],
        scorer=includes(),
    )
