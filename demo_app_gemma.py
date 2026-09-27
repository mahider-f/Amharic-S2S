import math
import re
import time
import unicodedata

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import uroman as ur

from scipy.signal import resample_poly
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    VitsModel,
)


# ============================================================
# CONFIG
# ============================================================

ASR_MODEL = "badrex/Ethio-ASR-multilingual-94M"

# Gemma GGUF is downloaded automatically on first run.
# You can override this by setting GEMMA_GGUF_PATH.
import os

LLM_MODEL = os.environ.get("GEMMA_GGUF_PATH")

if not LLM_MODEL:
    print("Resolving Gemma 3 4B model...")
    LLM_MODEL = hf_hub_download(
        repo_id="ggml-org/gemma-3-4b-it-GGUF",
        filename="gemma-3-4b-it-Q4_K_M.gguf",
    )

TTS_MODEL = "facebook/mms-tts-amh"

TARGET_SR = 16000

torch.set_num_threads(4)

LANG_TAG = re.compile(
    r"^\[(AMH|TIR|ORM|SID|WAL)\]\s*"
)


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a short-response Amharic voice assistant.

The user's text comes from speech recognition and may contain errors.

Rules:
- Reply in Amharic.
- Use only one short sentence unless clarification requires two.
- Do not simply repeat the user's text.
- Do not invent information.
- If the entire input is unintelligible, ask the user to repeat it.
- If a name, date, time, number, amount, or place appears corrupted or uncertain,
  ask the user to repeat that specific information.
- If the input is understandable, respond naturally to its meaning.
""".strip()


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("\n============================================")
print("  LOCAL AMHARIC SPEECH ASSISTANT")
print("============================================")

print("\n[1/4] Loading Ethio-ASR...")

asr_processor = AutoProcessor.from_pretrained(
    ASR_MODEL
)

asr_model = AutoModelForCTC.from_pretrained(
    ASR_MODEL
)

asr_model.eval()

print("✅ Ethio-ASR ready")


print("\n[2/4] Loading Gemma 3 4B...")

llm = Llama(
    model_path=LLM_MODEL,
    n_ctx=1024,
    n_threads=4,
    n_batch=128,
    verbose=False,
)

print("✅ Gemma ready")


print("\n[3/4] Loading MMS-TTS...")

tts_tokenizer = AutoTokenizer.from_pretrained(
    TTS_MODEL
)

tts_model = VitsModel.from_pretrained(
    TTS_MODEL
)

tts_model.eval()

print("✅ MMS-TTS ready")


print("\n[4/4] Initializing cached Uroman...")

romanizer = ur.Uroman()

print("✅ Uroman ready")
print("\n✅ ALL MODELS READY\n")


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):

    audio, sr = sf.read(
        path,
        dtype="float32",
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise gr.Error(
            "The recording is empty."
        )

    if sr != TARGET_SR:

        gcd = math.gcd(
            int(sr),
            TARGET_SR,
        )

        audio = resample_poly(
            audio,
            TARGET_SR // gcd,
            int(sr) // gcd,
        ).astype(np.float32)

        sr = TARGET_SR

    duration = len(audio) / sr

    peak = float(
        np.max(np.abs(audio))
    )

    rms = float(
        np.sqrt(
            np.mean(audio ** 2)
        )
    )

    print(
        f"[AUDIO] duration={duration:.3f}s "
        f"peak={peak:.5f} "
        f"rms={rms:.5f}"
    )

    if duration < 0.5:
        raise gr.Error(
            "Recording is too short."
        )

    if peak < 1e-5:
        raise gr.Error(
            "Recording appears silent."
        )

    # Normalize microphone recordings.
    audio = (
        0.95 * audio / peak
    ).astype(np.float32)

    return audio, sr, duration


# ============================================================
# ASR
# ============================================================

def transcribe(audio, sr):

    inputs = asr_processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    )

    start = time.perf_counter()

    with torch.inference_mode():

        logits = asr_model(
            **inputs
        ).logits

    elapsed = (
        time.perf_counter()
        - start
    )

    ids = torch.argmax(
        logits,
        dim=-1,
    )

    raw = asr_processor.batch_decode(
        ids
    )[0].strip()

    match = LANG_TAG.match(raw)

    language = (
        match.group(1)
        if match
        else "unknown"
    )

    clean = LANG_TAG.sub(
        "",
        raw
    ).strip()

    return (
        raw,
        clean,
        language,
        elapsed,
    )


# ============================================================
# GEMMA
# ============================================================


def sanitize_response(raw):

    raw = raw.strip()

    if not raw:
        return (
            "ይቅርታ፣ አልገባኝም። "
            "እባክዎን እንደገና ይናገሩ።"
        ), True

    cleaned = []

    for char in raw:

        code = ord(char)

        # Keep Ethiopic
        if 0x1200 <= code <= 0x137F:
            cleaned.append(char)
            continue

        # Keep ordinary ASCII:
        # English technical terms, numbers, punctuation
        if code < 128:
            cleaned.append(char)
            continue

        # Keep punctuation / spaces from Unicode
        category = unicodedata.category(char)

        if (
            category.startswith("P")
            or category.startswith("Z")
        ):
            cleaned.append(char)

        # Any other foreign-script letters are simply dropped

    cleaned = "".join(cleaned).strip()

    ethiopic_count = sum(
        1
        for c in cleaned
        if 0x1200 <= ord(c) <= 0x137F
    )

    # Only fallback if there is essentially no usable Amharic left
    if ethiopic_count < 4:
        return (
            "ይቅርታ፣ አልገባኝም። "
            "እባክዎን እንደገና ይናገሩ።"
        ), True

    return cleaned, cleaned != raw


def safe_response(raw):
    return sanitize_response(raw)


def generate_response(transcript):

    start = time.perf_counter()

    first_token_time = None
    pieces = []

    stream = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": transcript,
            },
        ],

        max_tokens=48,

        temperature=0.1,
        top_p=0.9,

        stream=True,
    )

    for chunk in stream:

        piece = (
            chunk
            .get("choices", [{}])[0]
            .get("delta", {})
            .get("content")
        )

        if not piece:
            continue

        if first_token_time is None:

            first_token_time = (
                time.perf_counter()
                - start
            )

        pieces.append(piece)

    total = (
        time.perf_counter()
        - start
    )

    raw = "".join(
        pieces
    ).strip()

    final, guard_used = (
        safe_response(raw)
    )

    return (
        raw,
        final,
        first_token_time,
        total,
        guard_used,
    )


# ============================================================
# FAST MMS-TTS
# ============================================================

def fast_tts_inputs(text):

    normalized = (
        tts_tokenizer
        .normalize_text(text)
    )

    preprocessed = (
        tts_tokenizer
        ._preprocess_char(normalized)
    )

    romanized = (
        romanizer
        .romanize_string(
            preprocessed
        )
    )

    return tts_tokenizer(
        romanized,
        return_tensors="pt",
    )


def synthesize(text):

    start = time.perf_counter()

    inputs = fast_tts_inputs(
        text
    )

    prep_done = (
        time.perf_counter()
    )

    with torch.inference_mode():

        waveform = tts_model(
            **inputs
        ).waveform[0]

    end = time.perf_counter()

    waveform = (
        waveform
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    sample_rate = int(
        tts_model.config.sampling_rate
    )

    duration = (
        len(waveform)
        / sample_rate
    )

    return (
        (sample_rate, waveform),
        prep_done - start,
        end - prep_done,
        duration,
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_assistant(audio_path):

    if not audio_path:

        raise gr.Error(
            "Record or upload audio first."
        )

    pipeline_start = (
        time.perf_counter()
    )

    # --------------------
    # AUDIO
    # --------------------

    audio, sr, input_duration = (
        load_audio(audio_path)
    )

    # --------------------
    # ASR
    # --------------------

    (
        raw_asr,
        transcript,
        language,
        asr_time,
    ) = transcribe(
        audio,
        sr,
    )

    print("\n==============================")
    print("[ASR]")
    print(raw_asr)

    if not transcript:

        raise gr.Error(
            "ASR did not return text."
        )

    # --------------------
    # GEMMA
    # --------------------

    (
        raw_llm,
        response,
        llm_ttft,
        llm_total,
        guard_used,
    ) = generate_response(
        transcript
    )

    print("\n[RAW GEMMA]")
    print(raw_llm)

    print("\n[FINAL RESPONSE]")
    print(response)

    # --------------------
    # TTS
    # --------------------

    (
        output_audio,
        tts_prep,
        tts_synthesis,
        response_audio_duration,
    ) = synthesize(
        response
    )

    total = (
        time.perf_counter()
        - pipeline_start
    )

    if llm_ttft is None:
        llm_ttft = llm_total

    guard_text = (
        "Yes"
        if guard_used
        else "No"
    )

    metrics = f"""
### ⚡ Pipeline Timing

| Stage | Time |
|---|---:|
| User speech duration | {input_duration:.2f} s |
| Ethio-ASR | {asr_time:.3f} s |
| Gemma time to first token | {llm_ttft:.3f} s |
| Gemma total generation | {llm_total:.3f} s |
| TTS preprocessing | {tts_prep:.4f} s |
| TTS synthesis | {tts_synthesis:.3f} s |
| Response audio duration | {response_audio_duration:.2f} s |
| **Backend response ready** | **{total:.3f} s** |

**ASR language ID:** `{language}`  
**Output safety guard triggered:** `{guard_text}`

🔒 **Fully local inference — no API key**
"""

    return (
        raw_asr,
        response,
        output_audio,
        metrics,
    )


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Local Amharic Speech Assistant"
) as demo:

    gr.Markdown(
        """
# 🎙️ Amharic Speech-to-Speech Assistant

### Fully Local AI Pipeline

**Speech → Ethio-ASR → Gemma 3 4B → MMS-TTS → Speech**

No external inference API • No API key
"""
    )

    audio_input = gr.Audio(
        sources=[
            "microphone",
            "upload",
        ],

        type="filepath",
        format="wav",

        label=(
            "🎤 Speak in Amharic "
            "or upload audio"
        ),
    )

    run_button = gr.Button(
        "▶ Run Assistant",
        variant="primary",
    )

    with gr.Row():

        asr_box = gr.Textbox(
            label="📝 ASR Transcript",
            lines=4,
        )

        response_box = gr.Textbox(
            label="🤖 Gemma Response",
            lines=4,
        )

    speech_output = gr.Audio(
        label="🔊 Spoken Response",
        autoplay=True,
        interactive=False,
    )

    metrics_box = gr.Markdown()

    run_button.click(
        fn=run_assistant,
        inputs=[
            audio_input,
        ],
        outputs=[
            asr_box,
            response_box,
            speech_output,
            metrics_box,
        ],
    )


if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=1
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
