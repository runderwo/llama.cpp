# MoE cache

When the experts of MoE layers stay in host memory (`-cmoe`, `-ncmoe`, `-ot`), the MoE cache keeps the most used ones in device memory for small batches (decode, speculative decoding). Large batches still upload the experts they use.

| Argument | |
|---|---|
| `--moe-cache-mib N` | cache size in MiB, split among the GPUs like the layers (default: 0, disabled) |
| `--moe-cache-cpu-share F` | share of the cache misses computed on the CPU from the host experts instead of uploaded, once the cache is full (default: 0) |
| `--moe-cache-rebalance` | enable dynamic rebalancing of experts during idle windows to optimize VRAM usage (default: disabled) |

## Dynamic rebalancing

When enabled with `--moe-cache-rebalance`, the MoE cache continuously monitors expert usage patterns during idle windows between inference requests. If the cache becomes imbalanced (e.g., certain experts are evicted too frequently due to LRU thrashing), the system automatically promotes frequently-used experts and evicts rarely-used ones to improve cache hit rates.

### How it works

- **Idle window detection:** Rebalancing only occurs when all inference slots are idle (no active requests).
- **Turnover threshold:** Rebalancing triggers only when ≥10% of cached experts need to change to improve the working set.
- **Hysteresis:** Minimum 2000ms idle window required to avoid frequent rebalancing during brief idle periods.
- **No PCIe stalls:** All rebalancing happens during idle windows; no transfers occur during `llama_decode()`.

### When to use it

Enable `--moe-cache-rebalance` when:
- Running MoE models with `--cpu-moe` and a VRAM cache (`--moe-cache-mib > 0`)
- Your workload has varying expert activation patterns over time
- You want to optimize cache hit rates without manual tuning
- You have idle time between requests (e.g., server mode with bursty traffic)

### Logging

When enabled, the rebalancer emits:
- **Debug logs:** Group-level turnover metrics and expert IDs being evicted/promoted
- **Normal logs:** Each expert migration (upload to VRAM slot) with layer and expert ID

Example debug log:
```
[moe_cache] group 0: turnover=15.3%, used=45/60 slots
[moe_cache] group 0: evicting 3 experts from layer 3, slot 12 (expert 145)
[moe_cache] migrated expert layer=3, expert=148, slot=12
```

The cache hits run on the device. With `--moe-cache-cpu-share F` the CPU computes a share `F` of the misses while the device computes the hits and uploads the other misses into the cache. This is faster when the CPU reads the host experts faster than the device can upload them, for example with fast RAM, a PCIe 4 link or experts that are not in pinned memory. The host experts are pinned with `-lm none`, `-lm mlock` and `-lm dio`, unless `--no-host`, but not when they are mmap'd (the default). The device uploads pinned experts at the full speed of the link, so the CPU share often does not help then. Until the cache is full every miss is uploaded, so the cache fills as fast as without the CPU share. The best share depends on the machine, the model and the cache size.

## Picking the arguments

Put the script in front of your server command:

```bash
python3 scripts/moe-cache-tune.py -- build/bin/llama-server -m model.gguf -cmoe -t 8 -c 32768
```

It starts the server 6 times, at two cache sizes × shares 0, 0.5 and 0.95, and times the generation of a few thousand tokens. That takes a few minutes, mostly model loading.

From these runs it fits the upload and CPU speeds of the machine, then replays the routing of all the runs through a copy of the cache logic for other sizes and shares. It prints the measured speeds, the predicted t/s and the recommended arguments: the lowest share within 5% of the fastest.

- **Server command:** keep your own arguments (`-t`, `-c`, expert placement, load mode); the speeds and the free device memory depend on them.
- **`--max-mib N`:** tunes a smaller cache than what fits on the device. The runs in `--out` are reused while the command stays the same, so trying another value is instant.
- **`--markdown`:** prints Markdown tables to paste in a PR.
