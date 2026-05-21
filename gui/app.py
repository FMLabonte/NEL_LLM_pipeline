"""
Flask-based web interface for the biomedical entity linking pipeline.

Start with:
    cd NEL_LLM_pipeline
    python3 gui/app.py

Then open http://localhost:5555 in your browser.

Requirements:
    pip install flask PyMuPDF
"""

import sys
import json
import time
import threading
import subprocess
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response

# ── Path setup ────────────────────────────────────────────────────────────
GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "candidate-generation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "domain-rules"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "llm-disambiguation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "linguistic-rules"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "improvements"))

app = Flask(__name__, template_folder=str(GUI_DIR / "templates"))

# ── Global pipeline state ────────────────────────────────────────────────
pipeline = None
pipeline_status = {"state": "not_loaded", "message": "Pipeline not initialized"}
pipeline_lock = threading.Lock()

# ── Evaluation state ─────────────────────────────────────────────────────
eval_status = {"running": False, "output": "", "finished": False}
eval_lock = threading.Lock()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except ImportError:
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            return text.strip()
        except ImportError:
            raise ImportError(
                "PDF support requires PyMuPDF or pdfplumber. "
                "Install with: pip install PyMuPDF  or  pip install pdfplumber"
            )


def init_pipeline(config: dict):
    """Initialize the pipeline with given configuration."""
    global pipeline, pipeline_status

    pipeline_status = {"state": "loading", "message": "Building MeSH index..."}

    try:
        from pipeline import BioLinkerPipeline

        mesh_xml = str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml")
        supp_xml = str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml")

        pipeline_status["message"] = "Building MeSH index (this may take a minute)..."

        # Knowledge base paths
        mrconso_path = str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF")
        mrrel_path = str(PROJECT_ROOT / "Data" / "UMLS" / "MRREL.RRF")
        umls_path = mrconso_path if Path(mrconso_path).exists() else None

        pipeline = BioLinkerPipeline(
            mesh_xml=mesh_xml,
            supp_xml=supp_xml,
            backend="rapidfuzz",
            top_k=config.get("top_k", 10),
            enrich_wikidata=True,
            enrich_dbpedia=True,
            enrich_umls=umls_path,
            use_expansion=config.get("use_expansion", True),
            use_phase3=config.get("use_phase3", True),
            use_phase4=config.get("use_phase4", False),
            mrrel_path=mrrel_path if Path(mrrel_path).exists() else None,
            llm_model=config.get("llm_model", "qwen3.5-9b"),
            llm_base_url=config.get("llm_base_url", "http://localhost:1234/v1"),
            use_abbreviation_expansion=config.get("use_abbreviation_expansion", True),
            use_embedding_retrieval=config.get("use_embedding_retrieval", True),
            hybrid_alpha=config.get("hybrid_alpha", 0.7),
        )

        pipeline_status = {"state": "ready", "message": "Pipeline ready"}

    except Exception as e:
        pipeline_status = {"state": "error", "message": f"Error: {str(e)}"}
        raise


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    return jsonify(pipeline_status)


@app.route("/api/init", methods=["POST"])
def api_init():
    """Initialize pipeline with given config."""
    config = request.json or {}

    def _init():
        try:
            init_pipeline(config)
        except Exception as e:
            print(f"Pipeline init error: {e}")

    thread = threading.Thread(target=_init, daemon=True)
    thread.start()

    return jsonify({"status": "initializing"})


@app.route("/api/extract-pdf", methods=["POST"])
def api_extract_pdf():
    """Extract text from uploaded PDF."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    try:
        pdf_bytes = f.read()
        text = extract_text_from_pdf(pdf_bytes)
        return jsonify({"text": text, "filename": f.filename})
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500


@app.route("/api/link", methods=["POST"])
def api_link():
    """Run entity linking on provided mentions."""
    global pipeline

    if pipeline is None or pipeline_status["state"] != "ready":
        return jsonify({"error": "Pipeline not ready. Initialize first."}), 503

    data = request.json
    mentions = data.get("mentions", [])
    context = data.get("context", "")
    title = data.get("title", "")
    config = data.get("config", {})

    if not mentions:
        return jsonify({"error": "No mentions provided"}), 400

    with pipeline_lock:
        orig_phase3 = pipeline.use_phase3
        orig_expansion = pipeline.use_expansion
        orig_phase4 = pipeline.use_phase4
        orig_abbrev = pipeline.use_abbreviation_expansion

        pipeline.use_phase3 = config.get("use_phase3", orig_phase3)
        pipeline.use_expansion = config.get("use_expansion", orig_expansion)
        pipeline.use_phase4 = config.get("use_phase4", orig_phase4)
        pipeline.use_abbreviation_expansion = config.get("use_abbreviation_expansion", orig_abbrev)

        try:
            gold_entities = []
            for m in mentions:
                if isinstance(m, str):
                    gold_entities.append({"text": m, "entity_type": None})
                else:
                    gold_entities.append(m)

            t0 = time.time()
            results = pipeline.link_entities(
                gold_entities=gold_entities,
                context=context,
                title=title,
            )
            elapsed = time.time() - t0

            response = {
                "results": [],
                "elapsed": round(elapsed, 2),
                "count": len(results),
            }

            for r in results:
                candidates = []
                for i, c in enumerate(r.candidates[:20]):
                    candidates.append({
                        "rank": i + 1,
                        "mesh_id": c.mesh_id,
                        "label": c.preferred_label,
                        "score": round(c.score, 2),
                        "tree_numbers": getattr(c, "tree_numbers", [])[:3],
                    })

                response["results"].append({
                    "mention": r.mention,
                    "entity_type": r.entity_type,
                    "mesh_id": r.mesh_id,
                    "preferred_label": r.preferred_label,
                    "phase2_top1": r.phase2_top1,
                    "phase3_top1": r.phase3_top1,
                    "confidence": r.confidence,
                    "candidates": candidates,
                })

            return jsonify(response)

        finally:
            pipeline.use_phase3 = orig_phase3
            pipeline.use_expansion = orig_expansion
            pipeline.use_phase4 = orig_phase4
            pipeline.use_abbreviation_expansion = orig_abbrev


# ── Evaluation ────────────────────────────────────────────────────────────

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Start a pipeline evaluation run in background."""
    global eval_status

    with eval_lock:
        if eval_status["running"]:
            return jsonify({"error": "An evaluation is already running"}), 409

        eval_status = {"running": True, "output": "", "finished": False}

    data = request.json or {}

    def _run_eval():
        global eval_status
        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "evaluate_pipeline.py"),
                "--dataset", data.get("dataset", "bc5cdr"),
                "--no-phase4",
            ]

            limit = data.get("limit")
            if limit:
                cmd += ["--limit", str(limit)]

            if data.get("embedding", True):
                cmd.append("--embedding")
                alpha = data.get("hybrid_alpha", 0.7)
                cmd += ["--hybrid-alpha", str(alpha)]

            if data.get("no_expansion", False):
                cmd.append("--no-expansion")

            if data.get("no_abbreviation", False):
                cmd.append("--no-abbreviation-expansion")

            if data.get("no_string_norm", False):
                cmd.append("--no-string-normalization")

            if data.get("no_topic_scoring", False):
                cmd.append("--no-topic-scoring")

            with eval_lock:
                eval_status["output"] = f"$ {' '.join(cmd[-8:])}\n\n"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT / "src"),
                bufsize=1,
            )

            for line in iter(process.stdout.readline, ""):
                with eval_lock:
                    eval_status["output"] += line

            process.wait()

            with eval_lock:
                eval_status["output"] += f"\n\nProcess exited with code {process.returncode}"
                eval_status["finished"] = True
                eval_status["running"] = False

        except Exception as e:
            with eval_lock:
                eval_status["output"] += f"\n\nERROR: {str(e)}"
                eval_status["finished"] = True
                eval_status["running"] = False

    thread = threading.Thread(target=_run_eval, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/evaluate/status")
def api_eval_status():
    """Get current evaluation status and output."""
    with eval_lock:
        return jsonify(eval_status)


@app.route("/api/evaluate/stop", methods=["POST"])
def api_eval_stop():
    """Mark evaluation as stopped (process runs to completion but UI resets)."""
    global eval_status
    with eval_lock:
        eval_status["running"] = False
        eval_status["finished"] = True
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    print("=" * 55)
    print("  MedLinker — Biomedical Entity Linking")
    print("  Open http://localhost:5555 in your browser")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5555, debug=False)
