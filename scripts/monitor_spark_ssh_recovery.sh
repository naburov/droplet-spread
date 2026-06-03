#!/usr/bin/env bash

set -euo pipefail

HOST="${1:-spark-b92-e}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
SSH_TIMEOUT_SECONDS="${SSH_TIMEOUT_SECONDS:-15}"
REMOTE_CHECK_TIMEOUT_SECONDS="${REMOTE_CHECK_TIMEOUT_SECONDS:-90}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/logs/spark_recovery"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ATTEMPT_LOG="${OUT_DIR}/ssh_recovery_${TIMESTAMP}.log"
REPORT_FILE="${OUT_DIR}/spark_health_${TIMESTAMP}.txt"
SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
PYTHON_BIN="${PYTHON_BIN:-}"

mkdir -p "${OUT_DIR}"

if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in python3 /opt/homebrew/bin/python3 /usr/bin/python3; do
        if [[ -x "${candidate}" ]] || command -v "${candidate}" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "${candidate}" 2>/dev/null || printf '%s' "${candidate}")"
            break
        fi
    done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python3 not found; set PYTHON_BIN explicitly" >&2
    exit 1
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${ATTEMPT_LOG}"
}

run_with_timeout() {
    local timeout_seconds="$1"
    shift

    "${PYTHON_BIN}" - "$timeout_seconds" "$@" <<'PY'
import subprocess
import sys

timeout_seconds = float(sys.argv[1])
command = sys.argv[2:]

try:
    completed = subprocess.run(command, timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    raise SystemExit(124)

raise SystemExit(completed.returncode)
PY
}

run_remote_checks() {
    run_with_timeout "${REMOTE_CHECK_TIMEOUT_SECONDS}" "${SSH_BIN}" -o ConnectTimeout=10 "${HOST}" 'bash -s' <<'EOF'
set -u

run_check() {
    local label="$1"
    shift
    echo
    echo "=== ${label} ==="
    if "$@"; then
        return 0
    fi
    local rc=$?
    echo "[command failed with exit code ${rc}]"
    return 0
}

echo "host=$(hostname)"
echo "time=$(date -Is)"

run_check "uptime" uptime
run_check "free -h" free -h
run_check "top cpu processes" bash -lc "ps aux --sort=-%cpu | head"
run_check "dmesg tail" bash -lc "dmesg | tail"
run_check "journalctl ssh tail" bash -lc "journalctl -u ssh --no-pager | tail"
EOF
}

log "Starting SSH recovery monitor for host ${HOST}"
log "Attempt log: ${ATTEMPT_LOG}"
log "Health report target: ${REPORT_FILE}"

attempt=1
while true; do
    log "Attempt ${attempt}: probing SSH"
    if run_with_timeout "${SSH_TIMEOUT_SECONDS}" "${SSH_BIN}" -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" 'echo ssh-ok' >/dev/null 2>&1; then
        log "SSH login succeeded on attempt ${attempt}"
        {
            echo "# Spark health report"
            echo "# Generated: $(date -Is)"
            echo "# Host alias: ${HOST}"
            run_remote_checks
        } > "${REPORT_FILE}" 2>&1 || true
        log "Saved remote health checks to ${REPORT_FILE}"
        exit 0
    else
        probe_rc=$?
    fi

    if [[ "${probe_rc}" -eq 124 ]]; then
        log "SSH probe timed out after ${SSH_TIMEOUT_SECONDS}s; sleeping ${INTERVAL_SECONDS}s"
    else
        log "SSH probe failed with exit code ${probe_rc}; sleeping ${INTERVAL_SECONDS}s"
    fi
    sleep "${INTERVAL_SECONDS}"
    attempt=$((attempt + 1))
done
