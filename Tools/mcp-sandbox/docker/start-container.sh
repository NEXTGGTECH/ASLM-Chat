#!/usr/bin/env bash
set -eu

THREAD_LIMIT="${SANDBOX_THREAD_LIMIT:-8}"
export OMP_NUM_THREADS="$THREAD_LIMIT"
export OPENBLAS_NUM_THREADS="$THREAD_LIMIT"
export MKL_NUM_THREADS="$THREAD_LIMIT"
export NUMEXPR_NUM_THREADS="$THREAD_LIMIT"
export VECLIB_MAXIMUM_THREADS="$THREAD_LIMIT"

NETWORK_LIMIT_MBIT="${SANDBOX_NETWORK_LIMIT_MBIT:-100}"
DEFAULT_TASK_DIR="${SANDBOX_DEFAULT_TASK_DIR:-_sandbox}"
WORKSPACE_ROOT="/workspace"
TASK_ROOT="${WORKSPACE_ROOT}/${DEFAULT_TASK_DIR}"

apply_network_limit() {
    if ! command -v tc >/dev/null 2>&1; then
        return 0
    fi

    if [ "${NETWORK_LIMIT_MBIT}" = "0" ] || [ "${NETWORK_LIMIT_MBIT}" = "" ]; then
        return 0
    fi

    iface="$(ip route show default 2>/dev/null | awk '/default/ {print $5; exit}')"
    if [ -z "${iface}" ]; then
        return 0
    fi

    tc qdisc del dev "${iface}" root >/dev/null 2>&1 || true
    tc qdisc del dev "${iface}" ingress >/dev/null 2>&1 || true

    # burst must be >= rate/kernel_HZ; 1mbit (128 KB) is safe for rates up to ~250mbit
    tc qdisc add dev "${iface}" root tbf rate "${NETWORK_LIMIT_MBIT}mbit" burst 1mbit latency 400ms >/dev/null 2>&1 || true
    tc qdisc add dev "${iface}" handle ffff: ingress >/dev/null 2>&1 || true
    tc filter add dev "${iface}" parent ffff: protocol all u32 match u32 0 0 police rate "${NETWORK_LIMIT_MBIT}mbit" burst 1mbit drop flowid :1 >/dev/null 2>&1 || true
}

apply_network_limit

mkdir -p "${TASK_ROOT}"
chmod -R a+rwX "${TASK_ROOT}" >/dev/null 2>&1 || true
cd "${TASK_ROOT}"

exec tail -f /dev/null
