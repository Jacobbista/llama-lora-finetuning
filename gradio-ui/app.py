import os
import time
import gc
from huggingface_hub import hf_hub_download, try_to_load_from_cache
import subprocess

# --- ⛔️ DO NOT TOUCH SECTION (Your Requirements) -----------------
if not os.path.exists("llama_cpp_python==0.3.15"):
    subprocess.run("pip install -V llama_cpp_python==0.3.15", shell=True)
else:
    print("✓ llama_cpp_python already installed")

from llama_cpp import Llama
import gradio as gr
import pandas as pd 

# ==========================================
# 📚 MODEL LIBRARY
# ==========================================
MODELS = {
    "Llama-3.2-3B (Default)": {
        "repo_id": "jacobbista/llama3-3b-finetome",
        "filename": "Llama-3.2-3B-Instruct.Q4_K_M.gguf",
        "revision": "main",
    },
    "LLama-3.2-1B-QLoRa": {
        "repo_id": "jacobbista/llama3-1b-finetome",
        "filename": "Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        "revision": "1B_QLoRA_N1000",
    },
    "Llama-3.2-1B-LoRA":{
        "repo_id": "jacobbista/llama3-1b-finetome",
        "filename": "Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        "revision": "1B_LoRA_N1000",
    }
}

# Global variables
llm = None
current_model_name = "None"

# ==========================================
# 💾 SMART MODEL MANAGEMENT
# ==========================================
def check_model_status():
    """
    Scans the disk to see which models are already downloaded.
    Returns a list for the Dataframe.
    """
    status_data = []
    for name, config in MODELS.items():
        # 1. Check if file is in the same folder as app.py (Manual placement)
        local_path = os.path.join(os.path.dirname(__file__), config['filename'])
        
        # 2. Check if file is in Hugging Face Cache (Automatic download)
        cached_path = try_to_load_from_cache(
            repo_id=config['repo_id'], 
            filename=config['filename'],
            revision=config['revision']
        )
        
        if os.path.exists(local_path):
            status = "✅ Local File (Ready)"
        elif cached_path:
            status = "✅ Cached (Ready)"
        else:
            status = "❌ Not Downloaded"
            
        status_data.append([name, status])
    return status_data

def load_llm(model_key):
    global llm, current_model_name
    
    config = MODELS[model_key]
    filename = config['filename']
    repo_id = config['repo_id']
    revision = config['revision']
    
    # Check if local file exists first (Priority)
    local_path = os.path.join(os.path.dirname(__file__), filename)
    
    # 1. Memory Cleanup
    if llm is not None:
        print("🗑️ Freeing memory...")
        del llm
        llm = None
        gc.collect()

    # 2. Determine Source
    if os.path.exists(local_path):
        print(f"📂 Found model locally: {local_path}")
        model_path = local_path
        status_msg = f"✅ Loaded {model_key} from Local File."
    else:
        # Check cache before downloading to give better UI feedback
        cached_path = try_to_load_from_cache(repo_id=repo_id, filename=filename, revision=revision)
        if cached_path:
            print(f"📂 Found model in HF Cache: {cached_path}")
            status_msg = f"✅ Loaded {model_key} from Cache (Instant)."
        else:
            print(f"⬇️ Downloading {model_key} from Hugging Face...")
            status_msg = f"✅ Downloaded & Loaded {model_key}."

        # Download (or get path if cached)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

    # 3. Initialize
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=min(8, os.cpu_count() or 2),
            n_gpu_layers=0,
            verbose=False,
        )
        current_model_name = model_key
        return status_msg, check_model_status() # Return updated table
    except Exception as e:
        return f"⚠️ Error: {str(e)}", check_model_status()

# --- INITIAL LOAD ---
print("⚙️  Startup: Loading default model...")
load_llm("Llama-3.2-3B (Default)")
load_llm("LLama-3.2-1B-QLoRa")
load_llm("Llama-3.2-1B-LoRA")
print("✓ Startup complete.")

# ==========================================
# 🎨 PERSONAS & CHAT LOGIC
# ==========================================
PERSONAS = {
    "Default Assistant": {"prompt": "You are a helpful AI assistant.", "icon": "🤖"},
    "Cyberpunk Hacker": {"prompt": "You are a runner in a cyberpunk dystopia. Speak in slang.", "icon": "📟"},
    "Medieval Knight": {"prompt": "You are a noble knight. Speak in Old English.", "icon": "🛡️"},
    "Grumpy IT Support": {"prompt": "You are a sarcastic IT sysadmin.", "icon": "☕"},
    "Haiku Master": {"prompt": "Answer ONLY in Haiku format.", "icon": "🌸"}
}

MAX_HISTORY_TURNS = 8

def respond(message, history, persona_name, temperature, max_tokens):
    global llm
    if llm is None:
        yield "⚠️ No model loaded! Go to 'Model Settings' to load one."
        return

    selected_persona = PERSONAS.get(persona_name, PERSONAS["Default Assistant"])
    system_prompt_text = selected_persona["prompt"]

    if len(history) > 2 * MAX_HISTORY_TURNS:
        history = history[-2 * MAX_HISTORY_TURNS :]

    prompt_str = "<|begin_of_text|>"
    prompt_str += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt_text}<|eot_id|>"

    for turn in history:
        role = turn['role']
        content = turn['content']
        if isinstance(content, list):
            text_content = ""
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_content += item.get('text', "")
            content = text_content
        prompt_str += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    prompt_str += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
    prompt_str += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    stream = llm(
        prompt_str,
        max_tokens=max_tokens,
        stop=["<|eot_id|>"],
        temperature=temperature,
        top_p=0.9,
        echo=False,
        stream=True 
    )
    
    partial = ""
    for output in stream:
        partial += output['choices'][0]['text']
        yield partial

# ==========================================
# 📊 BENCHMARK
# ==========================================
def run_benchmark(progress=gr.Progress()):
    if llm is None: return [["Error", "No model loaded", "0 t/s"]]
    
    test_prompts = [
        ("Logic", "If I have 3 apples and eat one, how many do I have?"),
        ("Creativity", "Write a one-sentence story about a robot."),
        ("Coding", "Write a Python function to add two numbers."),
        ("Knowledge", "What is the capital of France?"),
        ("Instruction", "Reply with exactly the word 'Banana' and nothing else."),
        ("Instruction", "Write this sentence backwards: 'Hello World!'"),
        ("Instruction", "Write this sentence with no spaces: 'I love programming.'"),
    ]
    results = []
    for cat, prompt in progress.tqdm(test_prompts, desc="Benchmarking..."):
        start = time.time()
        output = llm(f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n", max_tokens=100, echo=False)
        dur = time.time() - start
        tps = output['usage']['completion_tokens'] / dur
        results.append([cat, prompt, output['choices'][0]['text'].strip(), f"{tps:.2f} t/s"])
    return results

# ==========================================
# 💅 UI SETUP
# ==========================================
custom_css = """
h1 {text-align: center; color: #4F46E5;}
.chat-message {border-radius: 12px;}
"""

persona_dropdown = gr.Dropdown(choices=list(PERSONAS.keys()), value="Default Assistant", label="🎭 Choose AI Persona")
temp_slider = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
max_tokens_slider = gr.Slider(minimum=64, maximum=1024, value=256, step=64, label="Max Tokens")

with gr.Blocks() as demo:
    gr.Markdown("# ⚡ Llama 3.2: Multi-Model Lab")
    
    with gr.Tabs():
        # TAB 1: Chat
        with gr.Tab("💬 Chat Mode"):
            gr.ChatInterface(
                fn=respond,
                additional_inputs=[persona_dropdown, temp_slider, max_tokens_slider],
                examples=[["Hello! Who are you?"], ["Explain quantum physics."], ["Write a short poem about coding."]],
                cache_examples=False,
            )

        # TAB 2: Benchmark
        with gr.Tab("📊 Benchmark"):
            run_btn = gr.Button("🚀 Run Benchmark", variant="primary")
            benchmark_table = gr.Dataframe(headers=["Category", "Prompt", "Response", "Speed"])
            run_btn.click(fn=run_benchmark, outputs=benchmark_table)

        # TAB 3: Settings (UPDATED)
        with gr.Tab("⚙️ Model Settings"):
            gr.Markdown("### 📂 Manage Models")
            
            # The Status Table
            status_table = gr.Dataframe(
                value=check_model_status(),
                headers=["Model Name", "Current Status"],
                datatype=["str", "str"],
                interactive=False,
                label="Disk Status"
            )

            gr.Markdown("---")
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=list(MODELS.keys()), 
                    value="Llama-3.2-3B (Default)", 
                    label="Select Model to Load"
                )
                load_btn = gr.Button("📥 Load Selected Model", variant="primary")
            
            status_output = gr.Textbox(label="System Log", value=f"✅ Ready. Loaded: {current_model_name}")

            # Events
            load_btn.click(
                fn=load_llm, 
                inputs=model_selector, 
                outputs=[status_output, status_table] # Update both text and table
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"), css=custom_css)
