#!/usr/bin/env python3
import json, os, re, subprocess, urllib.request

API_URL = "http://" + "127.0.0.1" + ":8080/v1/chat/completions"
MODEL = "Qwen3-Coder-Next-80B"
TARGET_FILE = "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
MAX_ITERS = 20

def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()

def chat_stream(messages):
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1600,
        "stream": True,
    }).encode()
    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})
    full = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices")
            if not choices:
                continue
            delta = choices.pop(0)["delta"].get("content", "")
            if delta:
                print(delta, end="", flush=True)
                full.append(delta)
    print("\n")
    return "".join(full)

def apply_edits(text):
    pattern = re.compile(
        r"FILE:\s*([^\n]+)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL,
    )
    applied = 0
    for path, search, replace in pattern.findall(text):
        path = path.strip()
        if not os.path.exists(path):
            print(f"[!] File not found: {path}")
            continue
        content = open(path, "r").read()
        if search not in content:
            print(f"[!] SEARCH block not found in {path}")
            continue
        open(path, "w").write(content.replace(search, replace, 1))
        print(f"[+] Patched {path}")
        applied += 1
    return applied

def main():
    print("=== Step 0: Running baseline ./verify_multigpu.sh ===")
    rc, test_out = run_cmd("./verify_multigpu.sh")
    print(test_out[-600:])

    _, anchors = run_cmd(
        f"grep -n -E 'cpy_tensor_async|ggml_backend_vk_buffer|vk_device|vkQueueSubmit|vkCmdCopyBuffer|supports_bda' {TARGET_FILE} | head -n 45"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Vulkan C++ backend engineer fixing multi-GPU splitting (-ts 1,1 across two "
                "Radeon Pro V520 RDNA1 GPUs on Mesa RADV) in llama.cpp ggml/src/ggml-vulkan/ggml-vulkan.cpp. "
                "Single-GPU works cleanly, but splitting across Vulkan0 + Vulkan1 causes cross-device tensor "
                "corruption or GPUVM faults (0x8000fffff000) due to missing host-staged cross-device copies, "
                "buffer device address (BDA) assumptions across separate VkDevices, or missing queue/fence sync. "
                "Keep requests compact! To read lines, emit: READ: path/to/file.cpp:START-END (max 85 lines). "
                "To patch code, emit:\n"
                "FILE: ggml/src/ggml-vulkan/ggml-vulkan.cpp\n<<<<<<< SEARCH\nexact lines\n=======\nnew lines\n>>>>>>> REPLACE"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Baseline ./verify_multigpu.sh (rc={rc}):\n{test_out[-500:]}\n\n"
                f"Key line numbers in {TARGET_FILE}:\n{anchors}\n\n"
                "Emit up to two READ: requests around cpy_tensor_async and cross-device buffer handling."
            ),
        },
    ]

    for step in range(1, MAX_ITERS + 1):
        print(f"\n==================== ITERATION {step}/{MAX_ITERS} ====================")
        if len(messages) > 5:
            messages = [messages[0], messages[1]] + messages[-3:]
        reply = chat_stream(messages)
        messages.append({"role": "assistant", "content": reply})

        reads = re.findall(r"READ:\s*([^\s:]+):(\d+)-(\d+)", reply)
        if reads:
            chunks = []
            for path, s, e in reads[:2]:
                e = str(min(int(e), int(s) + 85))
                _, out = run_cmd(f"sed -n '{s},{e}p' {path} | nl -ba -v {s}")
                chunks.append(f"=== {path}:{s}-{e} ===\n{out}")
            messages.append({"role": "user", "content": "\n\n".join(chunks)})
            continue

        if apply_edits(reply) == 0:
            messages.append({"role": "user", "content": "Emit valid READ: or FILE: SEARCH/REPLACE blocks."})
            continue

        rc, test_out = run_cmd("./verify_multigpu.sh")
        print(test_out[-600:])

        if "GPUVM fault" in test_out or "ErrorDeviceLost" in test_out:
            print("\n[!] GPUVM fault triggered on Vulkan0—stopping loop so you can reset amdgpu cleanly.")
            return

        if rc == 0 and ("softmax" in test_out.lower() or "shader" in test_out.lower()) and "???" not in test_out:
            print("\n[SUCCESS] Multi-GPU (-ts 1,1) passed cleanly!")
            _, diff = run_cmd("git diff")
            print(diff)
            return

        if rc != 0:
            run_cmd(f"git checkout -- {TARGET_FILE}")
        messages.append({"role": "user", "content": f"Test result (rc={rc}):\n{test_out[-500:]}\n\nEmit next READ: or patch."})

if __name__ == "__main__":
    main()
