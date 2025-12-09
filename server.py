import uuid
import shutil
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json


# ---------------------------------------------------------
# GLOBAL STATE (same as Node.js Map)
# ---------------------------------------------------------
runs: Dict[str, Dict[str, Any]] = {}

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

app = FastAPI(title="LTUAS API", version="1.0.0")

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # optional
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load latest result JSON from outputs/results
# ---------------------------------------------------------
def load_latest_result_json():
    results_dir = Path("outputs/results")
    if not results_dir.exists():
        return None

    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        return None

    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)

    try:
        with latest_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Could not load JSON: {e}"}


# ---------------------------------------------------------
# Create run object
# ---------------------------------------------------------
def create_run(audio_path: str, prompt: Optional[str]):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "runId": str(uuid.uuid4()),
        "audioPath": audio_path,
        "prompt": prompt,
        "status": "idle",
        "createdAt": now,
        "updatedAt": now,
        "nodes": {
            "audio-file": "idle",
            "user-prompt": "idle",
            "describe-audio": "idle",
            "clap": "idle",
            "whisper": "idle",
            "llm-layer": "idle",
            "mellow": "idle",
            "json-output": "idle",
        },
        "result": None,
        "error": None,
        "logs": [],
    }


def update_run(run_id, updates):
    run = runs.get(run_id)
    if run:
        run.update(updates)
        run["updatedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")


def update_node(run_id, node, status):
    run = runs.get(run_id)
    if run:
        run["nodes"][node] = status
        run["updatedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")


def add_log(run_id, message):
    run = runs.get(run_id)
    if run:
        run["logs"].append(
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message}
        )


# ---------------------------------------------------------
# Pipeline Worker (Background thread)
# ---------------------------------------------------------
def python_pipeline_worker(run_id: str, audio_path: str, prompt: Optional[str]):
    try:
        add_log(run_id, "Pipeline started")
        update_node(run_id, "clap", "running")
        update_node(run_id, "whisper", "running")

        # Build command
        cmd = ["python", "main.py", audio_path]
        if prompt:
            cmd += ["--prompt", prompt]

        add_log(run_id, f"Executing: {' '.join(cmd)}")
        print(f"\n===== RUN {run_id}: EXECUTING SUBPROCESS =====")
        print(" ".join(cmd), "\n")

        # Popen with realtime streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

        # ----- STREAM STDOUT -----
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            # PRINT TO SERVER TERMINAL
            print(f"[subprocess stdout] {line}")

            # SAVE TO RUN LOGS
            add_log(run_id, line)

            # Stage detection
            if "CLAP" in line and "✓" in line:
                update_node(run_id, "clap", "success")

            if "Whisper" in line and "✓" in line:
                update_node(run_id, "whisper", "success")

            if "LLM" in line and "start" in line.lower():
                update_node(run_id, "llm-layer", "running")

            if "LLM" in line and "✓" in line:
                update_node(run_id, "llm-layer", "success")

            if "MELLOW" in line and "start" in line.lower():
                update_node(run_id, "mellow", "running")

            if "generated" in line.lower():
                update_node(run_id, "mellow", "success")

            if "Output saved" in line:
                update_node(run_id, "json-output", "running")

        # ----- STREAM STDERR -----
        for err in process.stderr:
            err = err.strip()
            if err:
                print(f"[subprocess stderr] {err}")
                add_log(run_id, f"[stderr] {err}")

        code = process.wait()
        print(f"\n===== SUBPROCESS EXITED (code {code}) =====\n")

        if code != 0:
            update_run(run_id, {"status": "error", "error": f"Exited with {code}"})
            add_log(run_id, f"Pipeline failed with exit code {code}")
            return

        # SUCCESS → load JSON
        final_json = load_latest_result_json()
        if final_json:
            runs[run_id]["result"] = final_json
            add_log(run_id, "Final JSON loaded")

        update_node(run_id, "json-output", "success")
        update_run(run_id, {"status": "success"})
        add_log(run_id, "Pipeline complete")

    except Exception as e:
        update_run(run_id, {"status": "error", "error": str(e)})
        add_log(run_id, f"Pipeline crashed: {e}")
        print(f"[pipeline error] {e}")

# ---------------------------------------------------------
# /run endpoint
# ---------------------------------------------------------
@app.post("/run")
async def start_pipeline(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in AUDIO_EXT:
        raise HTTPException(status_code=400, detail="Invalid audio format")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    audio_path = tmp.name

    run = create_run(audio_path, prompt)
    runs[run["runId"]] = run

    update_node(run["runId"], "audio-file", "success")
    update_node(run["runId"], "describe-audio", "success")
    if prompt:
        update_node(run["runId"], "user-prompt", "success")

    update_run(run["runId"], {"status": "running"})

    background.add_task(python_pipeline_worker, run["runId"], audio_path, prompt)

    return {"runId": run["runId"], "status": "running"}


# ---------------------------------------------------------
# /status/{runId}
# ---------------------------------------------------------
@app.get("/status/{run_id}")
async def get_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(404, "Run not found")
    return runs[run_id]


# ---------------------------------------------------------
# /runs (list)
# ---------------------------------------------------------
@app.get("/runs")
async def list_runs():
    arr = []
    for r in runs.values():
        arr.append({
            "runId": r["runId"],
            "status": r["status"],
            "createdAt": r["createdAt"],
            "updatedAt": r["updatedAt"],
            "prompt": r["prompt"],
            "hasResult": r["result"] is not None,
            "error": r["error"],
        })

    arr.sort(key=lambda x: x["createdAt"], reverse=True)
    return {"total": len(arr), "runs": arr}


# ---------------------------------------------------------
# /cancel/{runId}
# ---------------------------------------------------------
@app.post("/cancel/{run_id}")
async def cancel_run(run_id: str):
    if run_id not in runs:
        raise HTTPException(404, "Run not found")

    run = runs[run_id]
    if run["status"] != "running":
        raise HTTPException(400, "Pipeline not running")

    update_run(run_id, {"status": "cancelled"})
    add_log(run_id, "Pipeline cancelled")

    for node, st in run["nodes"].items():
        if st == "running":
            run["nodes"][node] = "idle"

    return {"message": "Cancelled", "runId": run_id}


# ---------------------------------------------------------
# /clear old runs
# ---------------------------------------------------------
@app.delete("/clear")
async def clear_old_runs(maxAge: int = 3600000):
    now = time.time()
    deleted = 0

    for run_id, r in list(runs.items()):
        created = time.mktime(time.strptime(r["createdAt"], "%Y-%m-%d %H:%M:%S"))

        if (now - created) > maxAge / 1000 and r["status"] != "running":
            try:
                if Path(r["audioPath"]).exists():
                    Path(r["audioPath"]).unlink()
            except:
                pass
            runs.pop(run_id, None)
            deleted += 1

    return {"message": f"Cleared {deleted}", "remaining": len(runs)}


# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=4000, reload=True)
