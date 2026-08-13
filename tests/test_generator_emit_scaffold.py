"""The emit harness, on the genome shape that measurably broke the generator.

Wave D's telemetry is the specification here. On `upms_j14_m3` -- fourteen
scalar loci, each a Literal over that job's OWN eligible machines -- the
authored generator emitted 7,104 candidates and the harness accepted 23
(0.3%): 6,021 shape rejects, 1,060 out-of-domain, 485 runtime failures,
39,993 pool slots falling back to schema-uniform. `upms_j13_m3` reproduced
it at 33.9%. Healthy instances on the same substrate run 67.3-96.0%, and the
failure rate is monotone in the number of loci, which is what a transcription
failure looks like and is not what a modelling failure looks like.

So the fixture below is an assignment-structured genome with heterogeneous
per-locus domains, and every test asks the same question: can the harness
still be made to reject a candidate for a structure IT already knows?

Nothing here is upms-specific. The scaffold, the census, the echo and the
guard are all driven by ``loci_of(template)`` and the domains mapping, and
the sequence-genome fixture in ``test_llm_generator.py`` exercises the same
code paths through a completely different shape.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Mapping

import pytest

from agent_evolve.core.authored import authored_artifact
from agent_evolve.infrastructure.authored_runtime import AuthoredRuntime
from agent_evolve.policies.emit_scaffold import (
    NOTES_GLOBAL,
    coerce_candidate,
    render_domain_echo,
    scaffold_prelude,
)
from agent_evolve.policies.llm_generator import (
    GENERATOR_PROMPT,
    AuthoredGenerator,
    GeneratorTelemetry,
    RejectionCensus,
    author_generator,
    render_generation_feedback,
    validate_pool,
)
from agent_evolve.policies.llm_surrogate import AuthorTelemetry

#: Five jobs, three machines, eligibility that DIFFERS per job -- the
#: structure the domain card cannot carry at field level and the generator
#: has to get right fourteen times in a row on the failing instance.
TEMPLATE: Dict[str, Any] = {f"job_{j:02d}": "M0" for j in range(5)}
ELIGIBLE = {
    "job_00": ["M0", "M2"],
    "job_01": ["M0", "M1", "M2"],
    "job_02": ["M1"],
    "job_03": ["M0", "M1"],
    "job_04": ["M1", "M2"],
}


def _template() -> Dict[str, Any]:
    return {name: values[0] for name, values in ELIGIBLE.items()}


class _Model:
    """A candidate model whose schema declares the per-job eligibility."""

    @staticmethod
    def model_json_schema() -> Mapping[str, Any]:
        return {"properties": {name: {"enum": list(values)}
                               for name, values in ELIGIBLE.items()}}


def _generator(source: str, **kwargs) -> AuthoredGenerator:
    artifact = authored_artifact("generator", source, name="gen",
                                 authored_by="llm")
    return AuthoredGenerator(artifact, AuthoredRuntime(), **kwargs)


def _propose(generator: AuthoredGenerator, want: int, *, seed: int = 1):
    return generator.propose(
        template=_template(), candidate_model=_Model, restriction=None,
        archive=[_template()], want=want, rng=random.Random(seed), seed=seed)


def _valid(config: Mapping[str, Any]) -> bool:
    return (set(config) == set(ELIGIBLE)
            and all(config[name] in values for name, values in ELIGIBLE.items()))


# --- (b) shape becomes impossible to emit, not caught after the fact ---------

def test_the_scaffold_builds_the_shape_so_the_model_cannot_get_it_wrong():
    """The failure mode itself: a generator that hand-builds its own dicts.

    It names one job wrongly, forgets another, and puts an ineligible machine
    in a third -- exactly the transcription errors whose rate scales with the
    number of loci. Through ``build`` every one of them is a locus-level
    correction and nothing is rejected.
    """

    generator = _generator(
        "def propose(archive, n, domains, seed):\n"
        "    import random\n"
        "    r = random.Random(seed)\n"
        "    out = []\n"
        "    for i in range(n):\n"
        "        picks = {'job_0': 'M0',          # a locus that does not exist\n"
        "                 'job_01': 'M9',         # a machine that does not exist\n"
        "                 'job_02': 'M0',         # ineligible for this job\n"
        "                 'job_03': r.choice(domains['job_03'])}\n"
        "        out.append(build(picks))\n"
        "    return out\n",
        max_revisions=0,
    )
    pool = _propose(generator, want=4)
    assert all(_valid(config) for config in pool)
    counters = generator.telemetry.as_dict()
    assert counters["rejected_shape"] == 0, (
        "a candidate built through the harness cannot have the wrong shape")
    assert counters["repaired"] == 0, "nothing needed harness-side assembly"
    assert counters["accepted"] > 0
    # ... and none of it was silent: the scaffold counted its own repairs.
    assert counters["scaffold_unknown_locus"] == 16      # 'job_0', 16 batches
    assert counters["scaffold_out_of_domain"] == 32      # job_01 and job_02
    assert counters["scaffold_filled"] == 32             # job_00 and job_04
    assert "job_02" in generator.census.out_of_domain_by_locus
    assert generator.census.samples["domain:job_01"] == "M9"


def test_the_scaffold_is_ablatable_and_its_absence_is_the_old_failure():
    """Same emission, scaffold off: the shape error the product used to take."""

    source = (
        "def propose(archive, n, domains, seed):\n"
        "    return [{'job_0': 'M0', 'job_1': 'M1'} for _ in range(n)]\n")
    broken = _generator(source, scaffold=False, repair=False, max_revisions=0)
    _propose(broken, want=2)
    assert broken.telemetry.rejected_shape == 8
    assert broken.telemetry.accepted == 0
    reasons = broken.census.shape_reasons
    assert any("missing fields" in reason for reason in reasons), reasons
    assert any("unexpected fields" in reason for reason in reasons), reasons


def test_a_candidate_the_harness_assembles_is_never_out_of_domain():
    """The harness-side twin, for a generator that ignores the scaffold.

    A bare sequence aligned with the loci is the natural way to write an
    assignment genome, and it is a 100%-shape-reject under the gate alone.
    """

    template = _template()
    domains = {name: list(values) for name, values in ELIGIBLE.items()}
    config, repairs = coerce_candidate(
        ["M2", "M1", "M1", "M9", "M2"], template=template, domains=domains,
        rng=random.Random(0))
    assert _valid(config)
    assert config["job_00"] == "M2" and config["job_02"] == "M1"
    assert repairs["out_of_domain"] == ["job_03"], (
        "only the ineligible pick was overridden")
    assert repairs["filled"] == []


def test_an_emission_with_nothing_admissible_is_still_a_reject():
    """A per-locus fallback must not credit the model with a uniform draw."""

    config, _ = coerce_candidate(
        {"job_00": "M9", "job_01": "M9", "job_02": "M9", "job_03": "M9",
         "job_04": "M9"},
        template=_template(),
        domains={k: list(v) for k, v in ELIGIBLE.items()},
        rng=random.Random(0))
    assert config is None
    assert coerce_candidate("not a candidate", template=_template(),
                            domains={k: list(v) for k, v in ELIGIBLE.items()},
                            rng=random.Random(0)) == (None, {})


def test_the_prelude_is_deterministic_given_the_call():
    """`build`'s fills are seeded from the batch, so a pool is reproducible."""

    domains = {k: list(v) for k, v in ELIGIBLE.items()}
    source = scaffold_prelude(_template(), domains, nonce=7)
    assert source is not None
    first, second = {}, {}
    exec(compile(source, "<prelude>", "exec"), first)      # noqa: S102
    exec(compile(source, "<prelude>", "exec"), second)     # noqa: S102
    a = [first["build"]({"job_00": "M2"}) for _ in range(5)]
    b = [second["build"]({"job_00": "M2"}) for _ in range(5)]
    assert a == b and all(_valid(config) for config in a)
    assert len({json.dumps(c, sort_keys=True) for c in a}) > 1, (
        "the fills must vary within a batch or the scaffold IS the collapse")
    assert first["build"](["M2", "M1", "M1", "M1", "M2"])["job_03"] == "M1", (
        "a sequence aligned with LOCI is a valid picks form")
    assert first[NOTES_GLOBAL]["built"] == 6


def test_a_prelude_that_cannot_be_rendered_degrades_to_no_prelude():
    assert scaffold_prelude({"x": object()}, {"x": [1]}) is None


# --- (a) the per-locus domain echo ------------------------------------------

def test_the_echo_names_every_locus_the_card_cannot():
    echo = render_domain_echo({k: list(v) for k, v in ELIGIBLE.items()})
    assert "job_02: one of ['M1']" in echo
    assert "job_00: one of ['M0', 'M2']" in echo
    for name in ELIGIBLE:
        assert name in echo
    # The singleton call-out, which exists because it is a measured crash:
    # 41 of 51 replayed zero-emission cells died on an empty draw.
    assert "1 locus/loci admit exactly ONE value (job_02)" in echo


def test_the_prelude_binds_the_standard_library_and_a_safe_resample():
    """The two harness-owned crash classes, both closed by the prelude.

    Measured by replaying the DEV probe's 51 zero-emission cells: 36 died on
    `NameError` for a module the gate already allows, and the rest on an
    empty `random.choice` -- which on this family means a locus with exactly
    one admissible value.
    """

    prelude = scaffold_prelude(
        _template(), {k: list(v) for k, v in ELIGIBLE.items()}, nonce=1)
    namespace: Dict[str, Any] = {}
    exec(compile(prelude, "<prelude>", "exec"), namespace)   # noqa: S102
    for module in ("math", "random", "statistics", "itertools", "collections"):
        assert module in namespace, f"{module} is allowed but not bound"
    # job_02 admits exactly one machine: "pick something else" must not raise.
    assert namespace["resample"]("job_02", current="M1") == "M1"
    assert namespace["resample"]("job_00", current="M0") == "M2"
    assert namespace["resample"]("nonexistent", current=7) == 7


def test_the_echo_collapses_runs_and_elides_huge_domains():
    """A 500-position genome costs one line; a heterogeneous one costs five."""

    uniform = {f"g[{i}]": [0, 1] for i in range(500)}
    echo = render_domain_echo(uniform)
    assert echo.count("\n") == 0
    assert "g[0] .. g[499] (500 loci): one of [0, 1]" == echo.strip()
    wide = render_domain_echo({"x": list(range(40))})
    assert "..." in wide and "(40 values)" in wide


#: sha256 of ``GENERATOR_PROMPT`` with every field blank, as it stood at
#: 50b6d19 -- the commit every sealed generator row was authored under.
SEALED_PROMPT_SHA = (
    "40e93a1a288092028cc06079d1776c29598544547729fde74d821e645df25f83")


def test_ablating_the_fix_restores_the_sealed_prompt_byte_for_byte():
    """The BEFORE arm of any before/after row has to be the sealed thing.

    Not "almost the sealed thing": a prompt that gained a blank line is a
    different prompt, and a paired row that changes two factors measures
    neither. With the echo empty and the scaffold off, the authoring prompt
    hashes to what it hashed to before this module existed.
    """

    import hashlib

    blank = GENERATOR_PROMPT.format(goals="", schema="", loci="", scaffold="",
                                    limits="", contract="", imports="")
    assert hashlib.sha256(blank.encode()).hexdigest() == SEALED_PROMPT_SHA


def test_the_authoring_prompt_carries_the_echo_and_the_scaffold_rules():
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    author_generator(complete, objectives=(), schema_text="card", attempts=1,
                     domains={k: list(v) for k, v in ELIGIBLE.items()},
                     telemetry=AuthorTelemetry())
    assert "job_02: one of ['M1']" in prompts[0]
    assert "build(picks)" in prompts[0]
    # ... and with nothing to echo the prompt is the one it always was.
    prompts.clear()
    author_generator(complete, objectives=(), schema_text="card", attempts=1,
                     scaffold=False, telemetry=AuthorTelemetry())
    assert "LOCI AND THEIR ADMISSIBLE VALUES" not in prompts[0]
    assert "build(picks)" not in prompts[0]
    assert "SEARCH SPACE:\ncard" in prompts[0]


# --- (c) counter-driven feedback: which locus, why, and what already failed --

def test_the_feedback_addresses_the_locus_rather_than_the_rate():
    census = RejectionCensus()
    for _ in range(412):
        census.out_of_domain("job_02", "M0")
    census.out_of_domain("job_04", "M0")
    census.shape("missing fields ['job_03']", {"job_00": "M0"})
    census.repaired("job_01")
    text = render_generation_feedback(
        GeneratorTelemetry(batches=3, emitted=100, accepted=1,
                           rejected_out_of_domain=413, rejected_shape=1,
                           repaired=2, repaired_loci=3),
        None, (), census=census,
        domains={k: list(v) for k, v in ELIGIBLE.items()})
    assert "locus job_02 -- out of domain 412 time(s); you used \"M0\"" in text
    assert "its domain is ['M1']" in text
    assert "shape -- missing fields ['job_03']" in text
    assert "the harness had to DECIDE these loci for you: job_01 (1)" in text
    assert "had to ASSEMBLE for you rather than reject: 2" in text
    # the worst offender is named first, because a revision acts on a few
    assert text.index("job_02") < text.index("job_04")


def test_a_revision_is_told_which_fixes_have_already_failed():
    edits = [{"revision": 1, "sha": "deadbeef", "before": 0.9, "after": 0.95,
              "signature": "domain:job_02", "excerpt": "def propose(...): ..."}]
    text = render_generation_feedback(GeneratorTelemetry(batches=1),
                                      rejected_edits=edits)
    assert "EDITS ALREADY TRIED THAT DID NOT FIX THIS" in text
    assert "revision 1 (source deadbeef): defect rate 90% -> 95%" in text
    assert "domain:job_02" in text and "def propose(...): ..." in text


def test_an_edit_that_does_not_move_the_defect_joins_the_memory():
    """The measured failure: 77 revisions fired, none repaired anything.

    Both replacements are as broken as the incumbent, so after each one has
    been measured it is remembered by name -- and the second revision's
    prompt carries the first one's failure.
    """

    broken = ("def propose(archive, n, domains, seed):\n"
              "    return [{'nope': 1} for _ in range(n)]\n")
    seen: list[str] = []

    def revise(artifact: Any, feedback: str) -> Any:
        seen.append(feedback)
        return authored_artifact("generator", broken + f"# {len(seen)}\n",
                                 name=f"rev{len(seen)}", authored_by="llm")

    generator = _generator(broken, revise=revise, max_revisions=2, repair=False,
                           scaffold=False)
    _propose(generator, want=2)
    _propose(generator, want=2, seed=2)          # revision 1 fires, then runs
    _propose(generator, want=2, seed=3)          # revision 2 sees its failure
    assert generator.telemetry.revisions == 2
    assert generator.telemetry.revisions_accepted == 2, (
        "unguarded revision takes what it is given")
    assert generator.telemetry.revisions_rejected == 2, (
        "and both of them were then MEASURED not to have helped")
    assert "EDITS ALREADY TRIED" not in seen[0], "nothing had failed yet"
    assert "EDITS ALREADY TRIED" in seen[1]
    assert "revision 1" in seen[1]


# --- the resource contract, which is what actually emptied the batches -------

def test_the_prompt_states_the_sandbox_budget_it_will_be_killed_by():
    """Measured on DEV seeds before this was added: on `upms_j14_m3` six of
    eight one-shot generators emitted ZERO candidates -- not wrong ones, none
    -- because the authored sampler ran a local search that blew the 10 s / 8 s
    sandbox budget. A per-candidate counter cannot see that (a killed batch
    contributes nothing to `emitted`), which is why Wave D's 7,104 emissions
    against 39,993 uniformly-filled slots read as a shape problem.
    """

    from agent_evolve.infrastructure.authored_runtime import RuntimeLimits

    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    author_generator(complete, objectives=(), schema_text="card", attempts=1,
                     limits=RuntimeLimits(wall_time_s=10.0, cpu_seconds=8,
                                          memory_bytes=512 * 1024 * 1024),
                     max_n=80, telemetry=AuthorTelemetry())
    assert "10 s wall-clock, 8 s" in prompts[0]
    assert "512 MB" in prompts[0] and "n=80" in prompts[0]
    prompts.clear()
    author_generator(complete, objectives=(), schema_text="card", attempts=1,
                     telemetry=AuthorTelemetry())
    assert "HARD RESOURCE LIMITS" not in prompts[0]


def test_a_batch_that_overruns_is_retried_smaller_instead_of_lost():
    """`propose` is a DISTRIBUTION, so fewer draws is the same request cheaper.

    The artifact below costs time proportional to n and is killed at the full
    pool; at a quarter of it, it finishes. A quarter of a guided pool beats
    none of one, and both the overrun and the recovery stay counted.
    """

    from agent_evolve.infrastructure.authored_runtime import RuntimeLimits

    # Cost as a STEP in n rather than a slope, so the test measures the retry
    # rather than the machine's load: the full pool always overruns, the
    # quarter always lands.
    slow = (
        "def propose(archive, n, domains, seed):\n"
        "    import random\n"
        "    if n > 40:\n"
        "        total = 0\n"
        "        while True:\n"
        "            total += 1\n"
        "    r = random.Random(seed)\n"
        "    return [build({k: r.choice(v) for k, v in domains.items()})\n"
        "            for _ in range(n)]\n"
    )
    artifact = authored_artifact("generator", slow, name="slow",
                                 authored_by="llm")
    generator = AuthoredGenerator(
        artifact, AuthoredRuntime(limits=RuntimeLimits(wall_time_s=3.0,
                                                       cpu_seconds=3)),
        max_revisions=0)
    pool = _propose(generator, want=20)          # n = 80, killed
    tel = generator.telemetry.as_dict()
    assert tel["runtime_failures"] == 1 and tel["runtime_retries"] == 1
    assert tel["runtime_recovered"] == 1, "the smaller call must have landed"
    assert tel["emitted"] == 20 and tel["accepted"] > 0
    assert tel["filled_uniform"] == 80 - tel["accepted"]
    assert len(pool) == 80 and all(_valid(c) for c in pool)


def test_the_retry_is_ablatable_and_off_means_the_whole_pool_is_lost():
    from agent_evolve.infrastructure.authored_runtime import RuntimeLimits

    slow = ("def propose(archive, n, domains, seed):\n"
            "    total = 0\n"
            "    while True:\n"
            "        total += 1\n"
            "    return []\n")
    generator = _generator(slow, max_revisions=0, shrink_on_overrun=0)
    generator.runtime = AuthoredRuntime(
        limits=RuntimeLimits(wall_time_s=2.0, cpu_seconds=2))
    pool = _propose(generator, want=5)
    tel = generator.telemetry.as_dict()
    assert tel["runtime_failures"] == 1 and tel["runtime_retries"] == 0
    assert tel["emitted"] == 0 and tel["filled_uniform"] == len(pool) == 20


# --- the guarded third arm ---------------------------------------------------

WORKING = ("def propose(archive, n, domains, seed):\n"
           "    import random\n"
           "    r = random.Random(seed)\n"
           "    return [build({k: r.choice(v) for k, v in domains.items()})\n"
           "            for _ in range(n)]\n")


def test_the_guard_refuses_a_revision_that_does_not_measurably_help():
    broken = ("def propose(archive, n, domains, seed):\n"
              "    return [{'nope': 1} for _ in range(n)]\n")
    offers = [broken, WORKING]

    def revise(artifact: Any, feedback: str) -> Any:
        return authored_artifact("generator", offers.pop(0), name="rev",
                                 authored_by="llm")

    guarded = _generator(broken, revise=revise, max_revisions=2,
                         revision_guard=True, repair=False)
    _propose(guarded, want=2)
    _propose(guarded, want=2, seed=2)            # offered the same defect
    assert guarded.telemetry.revisions_accepted == 0
    assert guarded.telemetry.revisions_rejected == 1
    assert guarded.artifact.name == "gen", "the incumbent was kept"
    _propose(guarded, want=2, seed=3)            # offered a working sampler
    assert guarded.telemetry.revisions_accepted == 1
    assert guarded.artifact.name == "rev"


def test_the_guard_is_off_unless_it_is_asked_for():
    """Every sealed row is defined on unguarded revision; the default is that."""

    broken = ("def propose(archive, n, domains, seed):\n"
              "    return [{'nope': 1} for _ in range(n)]\n")

    def revise(artifact: Any, feedback: str) -> Any:
        return authored_artifact("generator", broken, name="rev",
                                 authored_by="llm")

    plain = _generator(broken, revise=revise, max_revisions=1, repair=False)
    assert plain.revision_guard is False
    _propose(plain, want=2)
    _propose(plain, want=2, seed=2)
    assert plain.telemetry.revisions_accepted == 1, (
        "unguarded revision takes whatever the model returns")


# --- the fallback guarantee the composition rests on -------------------------

def test_a_generator_that_emits_nothing_still_degrades_to_schema_uniform():
    """ROW D-G's +0.0000-in-80-of-80 property, which the fix must not spend.

    A per-locus fallback can only improve on a per-candidate one where the
    model supplied SOMETHING admissible. Where it supplied nothing -- a
    crash, an empty list, a reply with no usable locus at all -- the pool is
    still filled by exactly the shipped schema-uniform sampler.
    """

    crashing = _generator(
        "def propose(archive, n, domains, seed):\n    raise ValueError('no')\n",
        max_revisions=0)
    pool = _propose(crashing, want=3, seed=11)
    assert crashing.telemetry.filled_uniform == len(pool) == 12

    from agent_evolve.policies.genetic import uniform_candidate
    rng = random.Random(11)
    expected = [uniform_candidate(_template(), _Model, rng=rng)
                for _ in range(12)]
    assert pool == expected, "the fallback draw is not bit-identical"


def test_validate_pool_keeps_its_sealed_semantics_when_repair_is_off():
    emitted = [{"job_00": "M2", "job_01": "M1", "job_02": "M1",
                "job_03": "M0", "job_04": "M2"},
               {"job_00": "M1", "job_01": "M2", "job_02": "M1",   # M1: not
                "job_03": "M1", "job_04": "M2"},                  # eligible
               ["M0", "M0", "M1", "M0", "M1"]]
    domains = {k: list(v) for k, v in ELIGIBLE.items()}
    strict = validate_pool(emitted, template=_template(), domains=domains)
    assert len(strict.accepted) == 1
    assert strict.rejected_out_of_domain == 1 and strict.rejected_shape == 1
    assert strict.repaired == 0
    assert strict.census.out_of_domain_by_locus == {"job_00": 1}

    lenient = validate_pool(emitted, template=_template(), domains=domains,
                            repair=True, rng=random.Random(0))
    assert len(lenient.accepted) == 3 and lenient.repaired == 2
    assert lenient.rejected_shape == 0 and lenient.rejected_out_of_domain == 0
    assert all(_valid(config) for config in lenient.accepted)


def test_the_defect_rate_counts_a_repair_as_a_defect():
    report = validate_pool(
        [["M2", "M1", "M1", "M0", "M2"], "junk"],
        template=_template(),
        domains={k: list(v) for k, v in ELIGIBLE.items()},
        repair=True, rng=random.Random(0))
    assert report.repaired == 1 and report.rejected_shape == 1
    assert report.defect_rate == pytest.approx(1.0), (
        "a candidate the harness finished is not one the model wrote")
