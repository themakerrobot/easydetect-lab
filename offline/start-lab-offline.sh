#!/usr/bin/env bash
# easydetect lab, offline bundle - start it on Linux (Ubuntu 20.04 or later, x86-64):
#   ./start-lab-offline.sh            (or: bash start-lab-offline.sh)
# Nothing here needs the internet: Python, PyTorch (CPU), the lab and the
# model weights it starts from are all inside this folder, and the lab is told
# never to reach for the network. Pictures, labels and trained models are kept
# in the data folder beside it. Then open http://127.0.0.1:8080 in a browser.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYEXE="$ROOT/python/bin/python3"
if [ ! -x "$PYEXE" ] || [ ! -f "$ROOT/lab/run.py" ]; then
  echo "[lab] python 폴더나 lab 폴더가 없어요. tar.gz를 전부 푼 다음 이 파일을 실행하세요."
  echo "[lab]   tar -xzf easydetect-lab-offline-linux-x64.tar.gz"
  exit 1
fi

# where easydetect keeps the weights (filled when the bundle was built) and
# where the lab keeps its datasets and runs: both inside this folder
export EASYDETECT_HOME="$ROOT/easydetect-cache"
export EASYDETECT_LAB_HOME="${EASYDETECT_LAB_HOME:-$ROOT/data}"
# never the network: the weights mirror at an address nothing answers on,
# and Hugging Face's own libraries (if anything loads them) offline
export EASYDETECT_ASSETS_URL="http://127.0.0.1:9"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
# this folder's Python only: not a Python installed on the PC, nor its packages
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8

echo "[lab] 오프라인 묶음으로 시작해요. 인터넷이 없어도 돼요."
echo "[lab] 처음 켤 때는 1~2분 걸릴 수 있어요. Ctrl+C나 이 창을 닫으면 lab이 꺼져요."
"$PYEXE" "$ROOT/lab/run.py" "$@"
status=$?
if [ $status -ne 0 ]; then
  echo "[lab] 문제가 생겼어요. 위의 메시지를 확인하세요."
  echo "[lab] 8080 포트를 다른 프로그램이 쓰고 있으면: ./start-lab-offline.sh --port 8090"
fi
exit $status
