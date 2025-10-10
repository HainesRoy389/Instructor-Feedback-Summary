import os, sys
from llama_cpp import Llama

MODEL_PATH = os.environ.get("GGUF_PATH", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# Hardware/Computing Parameters
N_CTX        = int(os.environ.get("N_CTX", 32768)) # Change this for laptops?
N_THREADS    = int(os.environ.get("N_THREADS", max(1, (os.cpu_count() or 2) - 1)))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", 0)) # All CPU for now?

# LLM Parameters
MAX_TOKENS = min(int(os.environ.get("MAX_TOKENS", 1500)), 1500)
TEMP       = float(os.environ.get("LLM_TEMP", "0.7"))
TOP_P      = float(os.environ.get("TOP_P", 0.95))
REPEAT_PEN = float(os.environ.get("REPEAT_PEN", 1.1))

# Instance creation
llm = Llama(
	model_path=MODEL_PATH,
	n_ctx=N_CTX,
	n_threads=N_THREADS,
	n_gpu_layers=N_GPU_LAYERS,
	chat_format="llama-3",
	verbose=False,
)

# Prints
print(f"Loaded model: {os.path.basename(MODEL_PATH)}")

# Files/path stuff
base_dir = os.path.dirname(os.path.abspath(__file__))
in_path  = os.path.join(base_dir, "user_prompt.txt")
system_prompt_path = os.path.join(base_dir, "system_prompt6.txt")				# Change to test
out_path = os.path.join(base_dir, "output.txt")

if not os.path.exists(in_path):
	print(f"Missing user_prompt file: {in_path}", file=sys.stderr)
	sys.exit(1)

if not os.path.exists(system_prompt_path):
	print(f"Missing user_prompt file: {system_prompt_path}", file=sys.stderr)
	sys.exit(1)

with open(in_path, "r", encoding="utf-8") as f:
	user_msg = f.read().strip()

with open(system_prompt_path, "r", encoding="utf-8") as f:
	system_prompt = f.read().strip()

if not user_msg:
	print("user_prompt.txt is empty. Nothing to do.", file=sys.stderr)
	sys.exit(2)

if not system_prompt:
	print("system_prompt.txt is empty. Fix this!", file=sys.stderr)
	sys.exit(2)

messages = [
	{"role": "system", "content": system_prompt},
	{"role": "user",   "content": user_msg},
]

# Stream tokens directly to output file
stream = llm.create_chat_completion(
	messages=messages,
	max_tokens=MAX_TOKENS,
	temperature=TEMP,
	top_p=TOP_P,
	repeat_penalty=REPEAT_PEN,
	stream=True,
)

chars = 0
with open(out_path, "w", encoding="utf-8") as out_f:
	for chunk in stream:
		delta = chunk["choices"][0]["delta"]
		text = delta.get("content", "")
		if text:
			out_f.write(text)
			chars += len(text)

print(f"Wrote response to {out_path} ({chars} chars).")
