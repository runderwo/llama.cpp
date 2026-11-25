#!/usr/bin/env bash
set -e

echo "=== 1. Compiling ggml-vulkan Shaders & llama-server ==="
docker run --rm --user $(id -u):$(id -g) -v ~/llamacpp-dev:/src -w /src \
  mesa-builder:latest -c "cmake --build build --target llama-server -j40" >/dev/null

echo "=== 2. Testing Qwen3-Coder-Next-80B on Vulkan1 (Subgroups ENABLED) ==="
CID=$(docker run -d --rm --entrypoint /custom-bin/llama-server -w /custom-bin \
  -p 8082:8082 --device /dev/dri \
  -v ~/.cache/llama.cpp:/models \
  -v ~/llamacpp-dev/build/bin:/custom-bin:ro \
  -e LLAMA_CACHE=/models \
  -e LD_PRELOAD=/models/numa_gguf_interleave.so \
  -e LD_LIBRARY_PATH=/custom-bin:/models/mesa27 \
  -e GGML_BACKEND_DIR=/custom-bin \
  -e VK_ICD_FILENAMES=/models/mesa27/radeon_icd.json \
  -e GGML_VK_VISIBLE_DEVICES=1 \
  -e GGML_VK_FORCE_FP32=1 \
  ghcr.io/ggml-org/llama.cpp:server-vulkan \
  --hf-repo unsloth/Qwen3-Coder-Next-GGUF \
  --hf-file Qwen3-Coder-Next-Q4_K_M.gguf \
  -ngl 99 --cpu-moe --flash-attn off -c 4096 -np 1 -t 40 --port 8082 --host 0.0.0.0)

trap "docker stop -t 2 $CID >/dev/null 2>&1 || true" EXIT

H="127.0.0.1"
P="8082"
for i in $(seq 1 60); do
  curl -s "http://${H}:${P}/health" | grep -q "ok" && break
  sleep 1
done

echo "=== 3. Running 1,695-Token Subgroup Check ==="
python3 -c '
import json, urllib.request
url = "http://" + "127.0.0.1" + ":8082/v1/chat/completions"
code = open("ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp").read()[:6000]
payload = json.dumps({
    "model": "qwen",
    "messages": [{"role": "user", "content": "Summarize what this Vulkan GLSL shader does in two clear sentences:\n\n" + code}],
    "max_tokens": 60,
    "temperature": 0
}).encode()
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
out = urllib.request.urlopen(req, timeout=180).read().decode()
print(out[:600])
'
