from pathlib import Path
from eval.golden_dataset import GOLDEN_DATASET
from eval.run_correctness_eval import ScriptedSolverClient
from eval.run_systems_ablation import run_prefix_length_sweep, run_no_shared_prefix_control

overrides = {"naive": lambda instance: ScriptedSolverClient(instance)}
out = Path("scratch_ablation.csv")

run_prefix_length_sweep(
    GOLDEN_DATASET[:2], ["short", "medium", "long"], [2, 4], ["naive"], 2,
    out, make_client_overrides=overrides,
)
run_no_shared_prefix_control(
    GOLDEN_DATASET[:2], "medium", [2, 4], ["naive"], 2,
    out, make_client_overrides=overrides,
)

print(out.read_text())
