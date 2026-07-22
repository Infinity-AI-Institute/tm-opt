#!/usr/bin/env bash
# Run ONCE after every pod restart. The container disk is ephemeral; only
# /workspace survives. Each block below was learned the hard way — see git
# history / docs/BASELINE_NOTES.md.
set -euo pipefail

#1. CUDA 13.0 toolkit (sm_103a; FlashInfer JIT hard-requires it)
if [ ! -d /usr/local/cuda-13.0 ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  rm -f /etc/apt/sources.list.d/cuda.list   # image ships a conflicting unsigned dup
  apt-get update && apt-get install -y cuda-toolkit-13-0
fi

#2. FlashInfer JIT + autotune caches: ~10+ min per cold start otherwise
mkdir -p /workspace/.flashinfer
rm -rf /root/.cache/flashinfer
ln -sfn /workspace/.flashinfer /root/.cache/flashinfer

#3. git credentials + identity (store lives on the volume)
git config --global credential.helper 'store --file /workspace/.git-credentials'
git config --global user.name  "adity-om"
git config --global user.email "aditya@infinity.inc"
git config --global --add safe.directory /workspace/tm-opt

#4. sshd keepalives + authorized key (direct-TCP VS Code stability)
grep -q "ClientAliveInterval 15" /etc/ssh/sshd_config || {
  printf "ClientAliveInterval 15\nClientAliveCountMax 8\n" >> /etc/ssh/sshd_config
  service ssh restart || true
}
# append your pubkey if missing:
# grep -qf ~/.ssh/id_ed25519_runpod.pub ~/.ssh/authorized_keys || cat pubkey >> ~/.ssh/authorized_keys

#5. python env sanity (venv persists; system pip does not)
source /workspace/venv/bin/activate
python -c "import vllm, scipy" 2>/dev/null || pip install scipy

echo "[pod_init] done. Serve: tmux new -s serve; bash scripts/serve_vllm_bench.sh"
