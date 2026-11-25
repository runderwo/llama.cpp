#!/usr/bin/env python3
import json, os, re, subprocess, urllib.request

API_URL = "http://" + "127.0.0.1" + ":8080/v1/chat/completions"
MODEL = "Qwen3-Coder-Next-80B"
MAX_ITERS = 15

def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()

def chat_stream(messages):
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1500,
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
            print(f"[!] Not found: {path}")
            continue
        content = open(path, "r").read()
        if search not in content:
            print(f"[!] SEARCH mismatch in {path}")
            continue
        open(path, "w").write(content.replace(search, replace, 1))
        print(f"[+] Patched {path}")
        applied += 1
    return applied

def main():
    print("=== Step 0: Running baseline ./verify_vk_shaders.sh ===")
    rc, test_out = run_cmd("./verify_vk_shaders.sh")
    print(test_out[-400:])

    _, files = run_cmd("ls -1 ggml/src/ggml-vulkan/vulkan-shaders/*gated_delta* ggml/src/ggml-vulkan/vulkan-shaders/*ssm* ggml/src/ggml-vulkan/vulkan-shaders/*soft_max* ggml/src/ggml-vulkan/vulkan-shaders/*mul_mat_vec*")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Vulkan GLSL compute shader engineer fixing RDNA 1 (GFX10, Wave32) "
                "precision/subgroup bugs in llama.cpp ggml-vulkan when GGML_VK_FORCE_FP32=1 is enabled. "
                "GPT-OSS-120B (36 layers with attn_sinks) corrupts on RDNA 1 unless both GGML_VK_DISABLE_SUBGROUPS=1 "
                "and --flash-attn off are set. Keep requests small! "
                "To inspect lines, emit: READ: path/to/file.comp:START-END (max 90 lines per READ). "
                "To patch code, emit:\n"
                "FILE: path/to/file.comp\n<<<<<<< SEARCH\nexact lines\n=======\nnew lines\n>>>>>>> REPLACE"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Baseline ./verify_vk_shaders.sh output (rc={rc}):\n{test_out[-400:]}\n\n"
                f"Shader files:\n{files}\n\n"
                "Emit READ: requests for ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp:1-90 first."
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
                e = str(min(int(e), int(s) + 60))
                _, out = run_cmd(f"sed -n '{s},{e}p' {path} | nl -ba -v {s}")
                chunks.append(f"=== {path}:{s}-{e} ===\n{out}")
            messages.append({"role": "user", "content": "\n\n".join(chunks)})
            continue

        if apply_edits(reply) == 0:
            messages.append({"role": "user", "content": "Emit a valid READ: or FILE: SEARCH/REPLACE block."})
            continue

        rc, test_out = run_cmd("./verify_vk_shaders.sh")
        print(test_out[-400:])
        if rc == 0 and ("shader" in test_out.lower() or "softmax" in test_out.lower() or "vulkan" in test_out.lower()) and "???" not in test_out and "_RF" not in test_out:
            print("\n[SUCCESS] Fixed RDNA 1 subgroup/FlashAttention shader bug!")
            _, diff = run_cmd("git diff")
            print(diff)
            return

        if rc != 0:
            run_cmd("git checkout -- ggml/")
        messages.append({"role": "user", "content": f"Test result (rc={rc}):\n{test_out[-400:]}\n\nEmit next READ: or patch."})

if __name__ == "__main__":
    main()
