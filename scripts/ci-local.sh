#!/usr/bin/env bash
#
# Run the CI workflow locally.
#
# This does not reimplement CI. It hands `.github/workflows/ci.yml` -- the same
# file GitHub reads -- to `act`, which executes it in containers. There is
# therefore no second definition of what CI does, and nothing to drift: if a
# step changes in the workflow, it changes here, because there is only one
# copy. A local runner that quietly does something different from the workflow
# is worse than none.
#
#   scripts/ci-local.sh              # every job
#   scripts/ci-local.sh stranger     # one job
#   scripts/ci-local.sh --list       # what is defined
#
# Requires docker and act (https://github.com/nektos/act).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOW="${REPO_ROOT}/.github/workflows/ci.yml"

# act's default micro image has almost nothing in it, and the workflow needs a
# Python toolchain and a compiler to build a wheel. This is the medium image;
# set ACT_IMAGE to override.
IMAGE="${ACT_IMAGE:-catthehacker/ubuntu:act-latest}"

command -v act >/dev/null 2>&1 || {
  echo "act is not installed: https://github.com/nektos/act" >&2
  exit 127
}
docker info >/dev/null 2>&1 || {
  echo "the docker daemon is not reachable; act needs it" >&2
  exit 127
}
[[ -f "${WORKFLOW}" ]] || { echo "no workflow at ${WORKFLOW}" >&2; exit 1; }

if [[ "${1:-}" == "--list" ]]; then
  exec act -l -W "${WORKFLOW}"
fi

# Credentials must not reach these containers. The stranger job's whole claim
# is that the package works without any, and a key leaking in from the host
# would make it pass for the wrong reason.
ARGS=(
  -W "${WORKFLOW}"
  -P "ubuntu-latest=${IMAGE}"
  --env AGENTEVOLVE_SCRUBBED=OPENAI_API_KEY,ANTHROPIC_API_KEY,OPENROUTER_API_KEY,GEMINI_API_KEY,MISTRAL_API_KEY,DEEPSEEK_API_KEY
  --no-cache-server
)

if [[ $# -gt 0 ]]; then
  echo "==> job: $1"
  exec act push "${ARGS[@]}" -j "$1"
fi

echo "==> all jobs"
exec act push "${ARGS[@]}"
