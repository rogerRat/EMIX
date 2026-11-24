#!/usr/bin/env bash
set -euo pipefail
cd /root/EMIX
export MODELSCOPE_ENDPOINT=https://modelscope.ai
export MODELSCOPE_API_TOKEN="$(cat /root/EMIX/ms.txt)"
export PYTHONPATH=/root/autodl-tmp/ms-hub-client
export PATH=/root/autodl-tmp/ms-hub-client/bin:/root/miniconda3/envs/emix/bin:$PATH
exec /root/miniconda3/envs/emix/bin/python -u hub/ms.py upload data/modelscope_EmixData1122 --namespace longRat --repo-name EmixData1122 --repo-type dataset --visibility public --batch-size 1 --max-retries 3
