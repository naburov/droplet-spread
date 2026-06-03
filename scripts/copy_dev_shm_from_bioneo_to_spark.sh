#!/usr/bin/env bash
# From local: copy key to spark, then start rsync in tmux on spark (bioneo2-1 -> spark).
# Remote (bioneo) is source of truth; retries until each dir completes (handles unstable SSH).
# All paths absolute (no ~).
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-spark-b92-e}"
SPARK_HOME="/home/naburov"
SPARK_SSH_DIR="${SPARK_HOME}/.ssh"
SPARK_SCRIPT="${SPARK_HOME}/.rsync_bioneo_to_data.sh"

BIONEO_SSH_HOST="${BIONEO_SSH_HOST:-ssh-sr003-jupyter.ai.cloud.ru}"
BIONEO_USER="${BIONEO_USER:-bioneo2-1.ai0001071-00970}"
BIONEO_PORT="${BIONEO_PORT:-2222}"
BIONEO_KEY="${BIONEO_KEY:-$HOME/.ssh/ssh-keys/cloud-ru-bioneo2}"
BIONEO_RSYNC_PATH="${BIONEO_RSYNC_PATH:-/home/jovyan/naburov/venvs/ml-tools/bin/rsync}"

REMOTE_KEY="${SPARK_SSH_DIR}/id_bioneo_copy"
BIONEO_RSYNC_USERHOST="${BIONEO_USER}@${BIONEO_SSH_HOST}"

LOCAL_KNOWN_HOSTS="${HOME}/.ssh/known_hosts"

[[ -f "$BIONEO_KEY" ]] || { echo "Missing $BIONEO_KEY" >&2; exit 1; }

# 1) Copy key to spark
scp -o BatchMode=yes -o ConnectTimeout=10 "$BIONEO_KEY" "${SPARK_HOST}:${REMOTE_KEY}"
ssh -o BatchMode=yes "$SPARK_HOST" "chmod 600 ${REMOTE_KEY}"
for kh_host in "$BIONEO_SSH_HOST" "bioneo2-1"; do
    [[ -f "$LOCAL_KNOWN_HOSTS" ]] && grep -q "$kh_host" "$LOCAL_KNOWN_HOSTS" 2>/dev/null && grep "$kh_host" "$LOCAL_KNOWN_HOSTS" | ssh "$SPARK_HOST" "mkdir -p ${SPARK_SSH_DIR} && cat >> ${SPARK_SSH_DIR}/known_hosts" || true
    break
done

# 2) Start rsync in tmux on remote (retry until each succeeds). Three paths hardcoded.
ssh -o BatchMode=yes "$SPARK_HOST" "mkdir -p /home/naburov/data"
RSYNC_SSH="ssh -i ${REMOTE_KEY} -p ${BIONEO_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=6"
ssh "$SPARK_HOST" "cat > ${SPARK_SCRIPT} << RSYNC_EOF
set -e
run_until_ok() {
  while true; do
    if \"\$@\"; then break; fi
    echo \"Failed, retrying in 60 s\"
    sleep 60
  done
}
run_until_ok rsync -avz --progress -vv --rsync-path=${BIONEO_RSYNC_PATH} -e \"${RSYNC_SSH}\" \"${BIONEO_RSYNC_USERHOST}:/dev/shm/.830/\" \"/home/naburov/data/.830/\"
run_until_ok rsync -avz --progress -vv --rsync-path=${BIONEO_RSYNC_PATH} -e \"${RSYNC_SSH}\" \"${BIONEO_RSYNC_USERHOST}:/dev/shm/.ub/\" \"/home/naburov/data/.ub/\"
run_until_ok rsync -avz --progress -vv --rsync-path=${BIONEO_RSYNC_PATH} -e \"${RSYNC_SSH}\" \"${BIONEO_RSYNC_USERHOST}:/dev/shm/.Саранск/\" \"/home/naburov/data/.Саранск/\"
echo \"=== all done ===\"
RSYNC_EOF
chmod +x ${SPARK_SCRIPT}
tmux kill-session -t bioneo_copy 2>/dev/null || true
tmux new-session -d -s bioneo_copy bash ${SPARK_SCRIPT}
"
echo "Rsync started in tmux on $SPARK_HOST (session bioneo_copy). Retries until each dir completes. Attach: ssh $SPARK_HOST tmux attach -t bioneo_copy"
