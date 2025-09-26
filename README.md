# Course Evaluation Summarizer (Local LLM via llama.cpp)

This repo runs a local LLM to turn raw course-evaluation comments into a structured summary (positives, neutral/mixed, negatives, and an overall tone) using `llama-cpp-python` and a quantized GGUF model.

## Quick Start

1. **Download a GGUF model**
	- Any GGUF model should work as long as the path is correct in the code or via an environment variable.
	- Example (recommended): download a Qwen 2.5 7B Instruct GGUF from Hugging Face:  
	  https://huggingface.co/paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF/tree/main
	- Either place the file at the repo root **with the filename used in the script**, or point to it with an environment variable (see below).

2. **Install dependency**
	```bash
	pip install llama-cpp-python
	```

3. **Prepare your inputs**
	- `system_prompt.txt` is already included in this repo. You may customize it if you want to change the summarization style/thresholds.
	- `input.txt` (the user prompt) already contains some starter text. **Append your course-evaluation comments at the bottom** where the placeholder data lives. Keep each comment numbered, e.g.:
		```
		Feedback 1: ...
		Feedback 2: ...
		...
		```
	- The script won’t run if `input.txt` is empty or if `system_prompt.txt` is empty/missing.

4. **Run**
	```bash
	python run_file_io.py
	```
	The model will load, read `input.txt` and `system_prompt.txt`, and write the summary to `output.txt`.

## Model Path & Environment Variables

- By default the script looks for a file named like a Qwen 2.5 7B Instruct GGUF in the repo root. If you use a different model or filename, either:
	- **Set an environment variable** before running:
		```bash
		# Example
		export GGUF_PATH=/absolute/path/to/your-model.gguf
		python run_file_io.py
		```
	- **Or edit** the `MODEL_PATH` at the top of `run_file_io.py`.

- Optional tuning knobs (set as environment variables):
	- `N_CTX` (context length), `N_THREADS` (CPU threads), `N_GPU_LAYERS` (offload layers to GPU if supported)
	- `MAX_TOKENS` (capped at 500 in the script), `LLM_TEMP`, `TOP_P`, `REPEAT_PEN`

## What the script does (summary)

- Loads the GGUF model (path from `GGUF_PATH` or the default in code)
- Reads `system_prompt.txt` (already included) and `input.txt` (your comments go at the bottom)
- Streams the generated summary into `output.txt`
- Exits with a clear message if a required file is missing or empty

## Tips for good results

- Keep comments short and clear, one per line, and **maintain numbering** (`Feedback 1`, `Feedback 2`, …).
- If you edit `system_prompt.txt`, keep instructions concise and consistent with how your comments are formatted.
- If you have a GPU-supported build of `llama-cpp-python`, try setting `N_GPU_LAYERS` to offload some layers.
