"""Cheap development-only DAG placement and dispatch co-optimization.

This synthetic problem exists only for paired agentic-workflow, lineage, and
recombination debugging.  It is not a scientific benchmark, performs no real
wall-clock measurement, and provides no evidence for wall-clock dominance or
paper claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent_evolve import ObjectiveSpec


DEVELOPMENT_ONLY_NOTICE: Final = (
    "Synthetic development task only: not a benchmark and not wall-clock evidence."
)

TaskName = Literal[
    "acquire",
    "encode",
    "inspect",
    "compress",
    "classify",
    "audit",
    "package",
    "release",
]
WorkerName = Literal["cpu_a", "cpu_b", "gpu", "npu"]

TASKS: Final[tuple[TaskName, ...]] = (
    "acquire",
    "encode",
    "inspect",
    "compress",
    "classify",
    "audit",
    "package",
    "release",
)
WORKERS: Final[tuple[WorkerName, ...]] = ("cpu_a", "cpu_b", "gpu", "npu")

# Every edge is a hard precedence constraint.  The graph contains two branches
# that reconverge at package/release, so placement and dispatch order affect
# overlap on the critical path.
DEPENDENCY_EDGES: Final[tuple[tuple[TaskName, TaskName], ...]] = (
    ("acquire", "encode"),
    ("acquire", "inspect"),
    ("encode", "compress"),
    ("inspect", "classify"),
    ("compress", "package"),
    ("classify", "package"),
    ("classify", "audit"),
    ("package", "release"),
    ("audit", "release"),
)

ALLOWED_WORKERS: Final[dict[TaskName, tuple[WorkerName, ...]]] = {
    "acquire": ("cpu_a", "cpu_b"),
    "encode": ("cpu_a", "cpu_b", "gpu"),
    "inspect": ("cpu_a", "cpu_b", "gpu", "npu"),
    "compress": ("cpu_a", "cpu_b", "gpu"),
    "classify": ("cpu_a", "cpu_b", "npu"),
    "audit": ("cpu_a", "cpu_b"),
    "package": ("cpu_a", "cpu_b"),
    "release": ("cpu_a", "cpu_b"),
}

# Synthetic deterministic execution times in abstract milliseconds.  They are
# model constants, not measured durations.
EXECUTION_MS: Final[dict[TaskName, dict[WorkerName, float]]] = {
    "acquire": {"cpu_a": 4.0, "cpu_b": 4.5},
    "encode": {"cpu_a": 12.0, "cpu_b": 13.0, "gpu": 5.0},
    "inspect": {"cpu_a": 10.0, "cpu_b": 11.0, "gpu": 6.0, "npu": 5.0},
    "compress": {"cpu_a": 19.0, "cpu_b": 16.0, "gpu": 5.0},
    "classify": {"cpu_a": 20.0, "cpu_b": 18.0, "npu": 4.0},
    "audit": {"cpu_a": 9.0, "cpu_b": 10.0},
    "package": {"cpu_a": 8.0, "cpu_b": 7.0},
    "release": {"cpu_a": 4.0, "cpu_b": 5.0},
}

POWER_MJ_PER_MS: Final[dict[WorkerName, float]] = {
    "cpu_a": 1.0,
    "cpu_b": 0.9,
    "gpu": 1.5,
    "npu": 1.2,
}
WORKING_SET_UNITS: Final[dict[TaskName, int]] = {
    "acquire": 1,
    "encode": 4,
    "inspect": 4,
    "compress": 6,
    "classify": 6,
    "audit": 2,
    "package": 2,
    "release": 1,
}
ACCELERATOR_MEMORY_CAPACITY: Final[dict[WorkerName, int]] = {
    "gpu": 10,
    "npu": 8,
}
MAX_TASKS_PER_WORKER: Final = 5
MAX_CROSS_WORKER_EDGES: Final = 6

# Fusion is available only when the two nodes are adjacent in dispatch order
# and placed on the same worker.  It therefore requires one innovation from
# each frozen branch below and is a deliberately nonlinear recombination cue.
FUSION_DURATION_MULTIPLIER: Final[dict[tuple[TaskName, TaskName], float]] = {
    ("encode", "compress"): 0.7,
}


class TaskAssignment(BaseModel):
    """One exact graph-node placement."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )

    task: TaskName
    worker: WorkerName


class CandidateConfig(BaseModel):
    """Strict candidate containing a canonical placement and DAG order."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )

    assignments: list[TaskAssignment] = Field(
        min_length=len(TASKS),
        max_length=len(TASKS),
    )
    dispatch_order: list[TaskName] = Field(
        min_length=len(TASKS),
        max_length=len(TASKS),
    )


# Explicit valid reference configuration.  Assignment entries always use TASKS
# order; dispatch_order is the independently mutable topological coordinate.
BASE_CONFIG = {
    "assignments": [
        {"task": "acquire", "worker": "cpu_a"},
        {"task": "encode", "worker": "cpu_a"},
        {"task": "inspect", "worker": "cpu_a"},
        {"task": "compress", "worker": "cpu_b"},
        {"task": "classify", "worker": "cpu_b"},
        {"task": "audit", "worker": "cpu_a"},
        {"task": "package", "worker": "cpu_b"},
        {"task": "release", "worker": "cpu_a"},
    ],
    "dispatch_order": [
        "acquire",
        "encode",
        "inspect",
        "classify",
        "audit",
        "compress",
        "package",
        "release",
    ],
}

# Paired development cohort.  LEFT changes only graph-node assignments.  RIGHT
# changes only topological dispatch order.  The target is their exact structural
# union.  All four values are valid; none is a benchmark incumbent.
DEVELOPMENT_BRANCH_LEFT = {
    "assignments": [
        {"task": "acquire", "worker": "cpu_a"},
        {"task": "encode", "worker": "gpu"},
        {"task": "inspect", "worker": "cpu_a"},
        {"task": "compress", "worker": "gpu"},
        {"task": "classify", "worker": "npu"},
        {"task": "audit", "worker": "cpu_a"},
        {"task": "package", "worker": "cpu_b"},
        {"task": "release", "worker": "cpu_a"},
    ],
    "dispatch_order": [
        "acquire",
        "encode",
        "inspect",
        "classify",
        "audit",
        "compress",
        "package",
        "release",
    ],
}
DEVELOPMENT_BRANCH_RIGHT = {
    "assignments": [
        {"task": "acquire", "worker": "cpu_a"},
        {"task": "encode", "worker": "cpu_a"},
        {"task": "inspect", "worker": "cpu_a"},
        {"task": "compress", "worker": "cpu_b"},
        {"task": "classify", "worker": "cpu_b"},
        {"task": "audit", "worker": "cpu_a"},
        {"task": "package", "worker": "cpu_b"},
        {"task": "release", "worker": "cpu_a"},
    ],
    "dispatch_order": [
        "acquire",
        "encode",
        "compress",
        "inspect",
        "classify",
        "audit",
        "package",
        "release",
    ],
}
DEVELOPMENT_RECOMBINATION_TARGET = {
    "assignments": [
        {"task": "acquire", "worker": "cpu_a"},
        {"task": "encode", "worker": "gpu"},
        {"task": "inspect", "worker": "cpu_a"},
        {"task": "compress", "worker": "gpu"},
        {"task": "classify", "worker": "npu"},
        {"task": "audit", "worker": "cpu_a"},
        {"task": "package", "worker": "cpu_b"},
        {"task": "release", "worker": "cpu_a"},
    ],
    "dispatch_order": [
        "acquire",
        "encode",
        "compress",
        "inspect",
        "classify",
        "audit",
        "package",
        "release",
    ],
}


@dataclass(frozen=True, slots=True)
class ScheduleAnalysis:
    """Deterministic trace projection for development debugging."""

    makespan_ms: float
    energy_mj: float
    peak_worker_load_ms: float
    task_start_ms: tuple[tuple[TaskName, float], ...]
    task_finish_ms: tuple[tuple[TaskName, float], ...]
    worker_load_ms: tuple[tuple[WorkerName, float], ...]
    fused_edges: tuple[tuple[TaskName, TaskName], ...]
    cross_worker_edge_count: int


def _transfer_latency_ms(source: WorkerName, target: WorkerName) -> float:
    if source == target:
        return 0.0
    if source.startswith("cpu") and target.startswith("cpu"):
        return 1.0
    if {source, target} == {"gpu", "npu"}:
        return 3.0
    if "npu" in (source, target):
        return 2.0
    return 2.5


def _transfer_energy_mj(source: WorkerName, target: WorkerName) -> float:
    if source == target:
        return 0.0
    if source.startswith("cpu") and target.startswith("cpu"):
        return 0.4
    if {source, target} == {"gpu", "npu"}:
        return 1.6
    return 1.1


class DagDispatchCoDesignProblem:
    """Deterministic heterogeneous DAG assignment/ordering landscape."""

    candidate_model = CandidateConfig
    example_config = BASE_CONFIG
    constraints_description = (
        "Assignments must list every task exactly once in canonical TASKS order. "
        "dispatch_order must be a topological permutation. Placements must respect "
        "worker capabilities, accelerator memory, per-worker task count, and the "
        "cross-worker-edge limit."
    )

    @property
    def objectives(self):
        return [
            ObjectiveSpec("makespan_ms", "min"),
            ObjectiveSpec("energy_mj", "min"),
            ObjectiveSpec("peak_worker_load_ms", "min"),
        ]

    @staticmethod
    def _normalized(config: object) -> dict[str, object]:
        return CandidateConfig.model_validate(config).model_dump(mode="python")

    @staticmethod
    def _validate_normalized(config: dict[str, object]) -> dict[TaskName, WorkerName]:
        assignments = cast(list[dict[str, str]], config["assignments"])
        dispatch_order = cast(list[TaskName], config["dispatch_order"])

        assignment_tasks = tuple(
            cast(TaskName, item["task"]) for item in assignments
        )
        if assignment_tasks != TASKS:
            raise ValueError(
                "assignments must contain every task exactly once in canonical TASKS order"
            )
        assignment_by_task: dict[TaskName, WorkerName] = {
            cast(TaskName, item["task"]): cast(WorkerName, item["worker"])
            for item in assignments
        }

        if len(set(dispatch_order)) != len(TASKS) or set(dispatch_order) != set(TASKS):
            raise ValueError("dispatch_order must be an exact permutation of TASKS")
        position = {task: index for index, task in enumerate(dispatch_order)}
        for predecessor, successor in DEPENDENCY_EDGES:
            if position[predecessor] >= position[successor]:
                raise ValueError(
                    f"dispatch_order violates dependency {predecessor}->{successor}"
                )

        counts = {worker: 0 for worker in WORKERS}
        for task in TASKS:
            worker = assignment_by_task[task]
            if worker not in ALLOWED_WORKERS[task]:
                raise ValueError(f"task {task} cannot run on worker {worker}")
            counts[worker] += 1
            if counts[worker] > MAX_TASKS_PER_WORKER:
                raise ValueError(
                    f"worker {worker} exceeds the {MAX_TASKS_PER_WORKER}-task limit"
                )

        for accelerator, capacity in ACCELERATOR_MEMORY_CAPACITY.items():
            usage = sum(
                WORKING_SET_UNITS[task]
                for task in TASKS
                if assignment_by_task[task] == accelerator
            )
            if usage > capacity:
                raise ValueError(
                    f"worker {accelerator} working set {usage} exceeds capacity {capacity}"
                )

        cross_worker_edges = sum(
            assignment_by_task[source] != assignment_by_task[target]
            for source, target in DEPENDENCY_EDGES
        )
        if cross_worker_edges > MAX_CROSS_WORKER_EDGES:
            raise ValueError(
                "assignment exceeds the cross-worker dependency-edge limit"
            )
        return assignment_by_task

    def validate(self, config: object) -> bool:
        normalized = self._normalized(config)
        self._validate_normalized(normalized)
        return True

    def analyze(self, config: object) -> ScheduleAnalysis:
        """Return a deterministic synthetic schedule trace and objective basis."""

        normalized = self._normalized(config)
        assignment_by_task = self._validate_normalized(normalized)
        dispatch_order = cast(list[TaskName], normalized["dispatch_order"])

        predecessors: dict[TaskName, list[TaskName]] = {task: [] for task in TASKS}
        for predecessor, successor in DEPENDENCY_EDGES:
            predecessors[successor].append(predecessor)

        worker_available = {worker: 0.0 for worker in WORKERS}
        worker_load = {worker: 0.0 for worker in WORKERS}
        start: dict[TaskName, float] = {}
        finish: dict[TaskName, float] = {}
        fused: list[tuple[TaskName, TaskName]] = []
        previous_task: TaskName | None = None
        energy_mj = 0.0

        for task in dispatch_order:
            worker = assignment_by_task[task]
            ready = 0.0
            for predecessor in predecessors[task]:
                predecessor_worker = assignment_by_task[predecessor]
                ready = max(
                    ready,
                    finish[predecessor]
                    + _transfer_latency_ms(predecessor_worker, worker),
                )
            task_start = max(ready, worker_available[worker])
            duration = EXECUTION_MS[task][worker]
            edge = (previous_task, task)
            if (
                previous_task is not None
                and edge in FUSION_DURATION_MULTIPLIER
                and assignment_by_task[previous_task] == worker
            ):
                duration *= FUSION_DURATION_MULTIPLIER[edge]
                fused.append(edge)
            task_finish = task_start + duration
            start[task] = task_start
            finish[task] = task_finish
            worker_available[worker] = task_finish
            worker_load[worker] += duration
            energy_mj += duration * POWER_MJ_PER_MS[worker]
            previous_task = task

        cross_worker_edges = 0
        for source, target in DEPENDENCY_EDGES:
            source_worker = assignment_by_task[source]
            target_worker = assignment_by_task[target]
            if source_worker != target_worker:
                cross_worker_edges += 1
                energy_mj += _transfer_energy_mj(source_worker, target_worker)

        return ScheduleAnalysis(
            makespan_ms=round(max(finish.values()), 6),
            energy_mj=round(energy_mj, 6),
            peak_worker_load_ms=round(max(worker_load.values()), 6),
            task_start_ms=tuple((task, round(start[task], 6)) for task in TASKS),
            task_finish_ms=tuple((task, round(finish[task], 6)) for task in TASKS),
            worker_load_ms=tuple(
                (worker, round(worker_load[worker], 6)) for worker in WORKERS
            ),
            fused_edges=tuple(fused),
            cross_worker_edge_count=cross_worker_edges,
        )

    def evaluate(self, config: object) -> dict[str, float]:
        analysis = self.analyze(config)
        return {
            "makespan_ms": analysis.makespan_ms,
            "energy_mj": analysis.energy_mj,
            "peak_worker_load_ms": analysis.peak_worker_load_ms,
        }

    def search_space_description(self) -> str:
        return f"""{DEVELOPMENT_ONLY_NOTICE}

Assign each node of the fixed eight-task DAG to one compatible worker and
provide a topological dispatch order. Assignments are a canonical TASKS-ordered
list; dispatch_order is an independently optimized permutation.

DAG edges:
  {DEPENDENCY_EDGES}

Workers:
  cpu_a, cpu_b, gpu, npu

Hard constraints:
  - every task appears exactly once in assignments and dispatch_order;
  - dispatch_order respects every DAG edge;
  - task/worker capability pairs must be valid;
  - gpu working-set capacity is 10 and npu capacity is 8;
  - no worker receives more than 5 tasks;
  - at most 6 DAG edges may cross workers.

Minimize synthetic makespan_ms, energy_mj, and peak_worker_load_ms. The
deterministic model includes dependency readiness, worker serialization,
cross-worker transfer costs, heterogeneous execution/power, and an
encode->compress fusion available only when placement and dispatch order align.
These values are abstract model outputs, not measured wall-clock data."""

    @staticmethod
    def render_candidate(config: object) -> str:
        return CandidateConfig.model_validate(config).model_dump_json()


problem = DagDispatchCoDesignProblem()


__all__ = (
    "BASE_CONFIG",
    "CandidateConfig",
    "DEPENDENCY_EDGES",
    "DEVELOPMENT_BRANCH_LEFT",
    "DEVELOPMENT_BRANCH_RIGHT",
    "DEVELOPMENT_ONLY_NOTICE",
    "DEVELOPMENT_RECOMBINATION_TARGET",
    "DagDispatchCoDesignProblem",
    "ScheduleAnalysis",
    "TASKS",
    "TaskAssignment",
    "problem",
)
