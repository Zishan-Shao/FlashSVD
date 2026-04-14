import os
import shlex
import subprocess
import sys
import time
import gradio as gr

# Path to the unified training script (adjust if you moved it)
SCRIPT_PATH = os.environ.get("TRAIN_UNIFIED_SCRIPT", "train_bert_unified_min.py")

# Define which tasks appear for each mode
CLS_TASKS = ["cola","sst2","qnli","qqp","mnli","stsb"]  # 6 tasks for CLS
MLM_TASKS = ["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte"]  # 8 tasks for MLM

def task_choices_for_mode(mode: str):
    return CLS_TASKS if mode == "cls" else MLM_TASKS

def build_cmd(
    mode, task, model, epochs, batch_size, lr, logging_steps, eval_steps, seed,
    output_dir, no_cuda, cuda_visible_devices, extra_args
):
    cmd = [
        sys.executable, "-u", SCRIPT_PATH,
        "--mode", mode,
        "--task", task,
        "--model", model,
        "--epochs", str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--learning_rate", str(lr),
        "--logging_steps", str(int(logging_steps)),
        "--eval_steps", str(int(eval_steps)),
        "--seed", str(int(seed)),
    ]
    if output_dir:
        cmd += ["--output_dir", output_dir]
    if no_cuda:
        cmd += ["--no_cuda"]
    if extra_args:
        cmd += shlex.split(extra_args)
    return cmd, (cuda_visible_devices or "").strip()

def run_training(
    mode, task, model, epochs, batch_size, lr, logging_steps, eval_steps, seed,
    output_dir, no_cuda, cuda_visible_devices, extra_args,
    log_dir, run_name, append_log
):
    # Prepare command & env
    cmd, cudavis = build_cmd(
        mode, task, model, epochs, batch_size, lr, logging_steps, eval_steps, seed,
        output_dir, no_cuda, cuda_visible_devices, extra_args
    )
    env = os.environ.copy()
    if cudavis:
        env["CUDA_VISIBLE_DEVICES"] = cudavis

    # Prepare log file
    if not log_dir:
        log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    if not run_name:
        run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{mode}_{task}"
    log_path = os.path.join(log_dir, f"{run_name}.log")
    mode_flag = "a" if append_log else "w"

    # Start & stream logs (and persist to file)
    header = "$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n"
    log_text = header
    yield log_text, None  # stream to UI first

    try:
        with open(log_path, mode_flag, encoding="utf-8") as fp:
            fp.write(header)
            fp.flush()

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env
            )

            last_flush = time.time()
            for line in proc.stdout:
                log_text += line
                fp.write(line)
                now = time.time()
                # Limit UI refresh rate for performance
                if now - last_flush >= 0.1:
                    yield log_text, None
                    last_flush = now

            ret = proc.wait()
            tail = f"\n[Process exited with code {ret}]"
            log_text += tail
            fp.write(tail)
            fp.flush()
    except Exception as e:
        err = f"[Launch Error] {e}"
        log_text += "\n" + err
        # Try to write error to file as well
        try:
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write("\n" + err + "\n")
        except Exception:
            pass
        yield log_text, log_path
        return

    # At the end, return the log file for download
    yield log_text, log_path

def ui():
    with gr.Blocks(title="Unified BERT Trainer (CLS / MLM)") as demo:
        gr.Markdown("# Unified BERT Trainer\nTrain BERT on GLUE (CLS) or MLM using your unified script.")

        with gr.Row():
            mode = gr.Dropdown(["cls","mlm"], value="cls", label="Mode")
            task = gr.Dropdown(task_choices_for_mode("cls"), value=CLS_TASKS[0], label="Task (GLUE)")

        # Dynamically update task choices when mode changes
        def _on_mode_change(m):
            choices = task_choices_for_mode(m)
            new_label = "Task / Corpus (GLUE)" if m == "mlm" else "Task (GLUE)"
            return gr.update(choices=choices, value=choices[0], label=new_label)

        mode.change(fn=_on_mode_change, inputs=mode, outputs=task)

        with gr.Row():
            model = gr.Textbox(value="bert-base-uncased", label="Model Checkpoint")
            output_dir = gr.Textbox(value="", label="Output Dir (optional)")

        with gr.Row():
            epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
            batch_size = gr.Slider(1, 128, value=32, step=1, label="Batch Size")
            lr = gr.Number(value=2e-5, label="Learning Rate")

        with gr.Row():
            logging_steps = gr.Slider(1, 500, value=100, step=1, label="Logging Steps")
            eval_steps = gr.Slider(0, 5000, value=0, step=10, label="Eval Steps (0=per-epoch)")
            seed = gr.Slider(0, 10000, value=0, step=1, label="Seed")

        with gr.Accordion("Advanced", open=False):
            no_cuda = gr.Checkbox(value=False, label="Force CPU (--no_cuda)")
            cuda_visible_devices = gr.Textbox(value="", label="CUDA_VISIBLE_DEVICES (e.g., 0 or 0,1)")
            extra_args = gr.Textbox(value="", label="Extra CLI Args (appended to command)")

        with gr.Accordion("Logging", open=True):
            log_dir = gr.Textbox(value="runs", label="Log Directory")
            run_name = gr.Textbox(value="", label="Run Name (auto if empty)")
            append_log = gr.Checkbox(value=False, label="Append if log exists")

        run_btn = gr.Button("Start Training", variant="primary")
        output = gr.Textbox(label="Logs", lines=24)
        log_file = gr.File(label="Download Log File", interactive=False)

        run_btn.click(
            fn=run_training,
            inputs=[mode, task, model, epochs, batch_size, lr, logging_steps, eval_steps, seed,
                    output_dir, no_cuda, cuda_visible_devices, extra_args,
                    log_dir, run_name, append_log],
            outputs=[output, log_file],
        )

    return demo

if __name__ == "__main__":
    demo = ui()
    demo.launch()
