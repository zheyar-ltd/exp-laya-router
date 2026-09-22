import time
import torch
import laya

print("Initializing Laya Router on CUDA...")
router = laya.Router(device="cuda", preload=True)

state = "The server returned HTTP 502 Bad Gateway during checkout for transaction #8839."
questions = {
    "department": {
        "instructions": "Which department should handle this incident?",
        "criteria": {"billing": "", "devops": "", "marketing": "", "logistics": ""},
        "type": "choice"
    },
    "urgency": {
        "instructions": "Rate the severity level of this issue",
        "criteria": {"low": "", "medium": "", "high": "", "critical": ""},
        "type": "choice"
    },
    "escalate": {
        "instructions": "Requires on-call engineer page",
        "criteria": {"yes": "", "no": ""},
        "type": "choice"
    }
}

num_decisions_per_call = len(questions)  # 3 decisions per predict() run
warmup_runs = 10
test_runs = 100

print("Warming up GPU...")
for _ in range(warmup_runs):
    router.predict(state=state, questions=questions)

torch.cuda.synchronize()  # Wait for GPU queues to complete

print(f"Running benchmark ({test_runs} iterations)...")
start_time = time.perf_counter()

for _ in range(test_runs):
    router.predict(state=state, questions=questions)

torch.cuda.synchronize()  # Accurate GPU timing
total_elapsed = time.perf_counter() - start_time

total_calls = test_runs
total_decisions = total_calls * num_decisions_per_call
avg_latency_ms = (total_elapsed / total_calls) * 1000
decisions_per_sec = total_decisions / total_elapsed
calls_per_sec = total_calls / total_elapsed

print("\n--- Benchmark Results ---")
print(f"Total time elapsed:       {total_elapsed:.2f} s")
print(f"Avg latency per call:     {avg_latency_ms:.2f} ms")
print(f"API calls / sec:          {calls_per_sec:.2f} req/s")
print(f"Total decisions / sec:    {decisions_per_sec:.2f} decisions/s")