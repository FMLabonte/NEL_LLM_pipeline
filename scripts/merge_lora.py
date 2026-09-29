#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model, without re-training.

Why this exists: `finetune_lora.py --merge` writes the merged model at the end of
training. If the disk quota runs out during that write, the training job still
reports COMPLETED but leaves a truncated directory behind -- typically a 0-byte
tokenizer.json, which surfaces much later as

    json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

when vLLM tries to serve it. The adapter itself is written before the merge and
is only a few hundred MB, so it usually survives. Re-merging takes minutes;
re-training takes hours.

    python3 scripts/merge_lora.py --adapter out/qwen3-4b-nel-lora-s3 \
                                  --base Qwen/Qwen3-4B

Writes <adapter>-merged, refusing to overwrite unless --force. Runs on CPU:
no GPU needed, but it wants ~16 GB of RAM for a 4B model in fp16.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# A directory only counts as a usable model if these are present AND non-empty.
# Checking that the directory merely exists is what let the broken merge through.
REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json")


def looks_complete(d: Path) -> tuple[bool, str]:
    if not d.is_dir():
        return False, "Verzeichnis fehlt"
    for name in REQUIRED:
        p = d / name
        if not p.is_file():
            return False, f"{name} fehlt"
        if p.stat().st_size == 0:
            return False, f"{name} ist 0 Bytes (abgeschnittener Schreibvorgang)"
        if name.endswith(".json"):
            try:
                json.loads(p.read_text())
            except Exception as e:
                return False, f"{name} ist kein gueltiges JSON ({e.__class__.__name__})"
    weights = list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
    if not weights:
        return False, "keine Gewichtsdateien"
    total = sum(w.stat().st_size for w in weights)
    if total < 2 * 1024**3:
        return False, f"Gewichte nur {total/1024**3:.1f} GB -- zu klein fuer ein 4B-Modell"
    return True, f"{total/1024**3:.1f} GB Gewichte, {len(weights)} Dateien"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Verzeichnis mit dem LoRA-Adapter")
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default=None, help="Standard: <adapter>-merged")
    ap.add_argument("--force", action="store_true",
                    help="ein vorhandenes Ausgabeverzeichnis loeschen")
    ap.add_argument("--check-only", action="store_true",
                    help="nur pruefen, ob das Ergebnis vollstaendig aussieht")
    a = ap.parse_args()

    adapter = Path(a.adapter)
    out = Path(a.out) if a.out else Path(str(adapter).rstrip("/") + "-merged")

    ok, why = looks_complete(out)
    if a.check_only:
        print(f"{out}: {'vollstaendig' if ok else 'UNVOLLSTAENDIG'} -- {why}")
        sys.exit(0 if ok else 1)
    if ok and not a.force:
        sys.exit(f"{out} sieht bereits vollstaendig aus ({why}). --force zum Neubauen.")

    if not (adapter / "adapter_config.json").is_file():
        sys.exit(f"FEHLER: {adapter} enthaelt keinen LoRA-Adapter "
                 f"(adapter_config.json fehlt). Hier hilft nur Neu-Trainieren.")

    if out.exists():
        print(f"loesche unvollstaendiges {out} ({why})")
        shutil.rmtree(out)

    print(f"lade Basismodell {a.base} auf CPU ...")
    base = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.float16, device_map="cpu")
    print(f"wende Adapter {adapter} an und merge ...")
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    # Tokenizer aus dem Adapter-Verzeichnis, nicht vom Hub: finetune_lora.py legt
    # ihn dort ab, und genau diese Fassung hat das Modell im Training gesehen.
    src = adapter if (adapter / "tokenizer_config.json").is_file() else a.base
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(out)

    ok, why = looks_complete(out)
    print(f"\n{out}: {'vollstaendig' if ok else 'UNVOLLSTAENDIG'} -- {why}")
    if not ok:
        sys.exit("FEHLER: das Ergebnis ist unvollstaendig. 'quota -s' pruefen.")
    print("fertig. Danach:  sbatch --array=<seed> cluster/bender_eval_ft_seeds.sbatch")


if __name__ == "__main__":
    main()
