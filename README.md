# EPUB / SRT / TXT / PDF / Kindle / FB2 / DOCX Translator with LLM

<p align="center">
</p>


<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-lightgrey" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1%20%7C%2012.4-76b900?logo=nvidia&logoColor=white" />
  <img alt="Status" src="https://img.shields.io/badge/status-stable-brightgreen" />
  <img alt="License" src="https://img.shields.io/badge/License-GPLv3-blue" />
</p>

A desktop application for translating **EPUB**, **SRT**, **TXT**, **ASS**, **PDF**, **MOBI/AZW/AZW3**, **FB2**, and **DOCX** files using local or cloud-hosted language models.

---

<img width="1502" height="850" alt="tar" src="https://github.com/user-attachments/assets/e367cca8-55c1-4fa4-ae2d-05d3ab494323" />

---

## Screenshots .epub before and after:

<details>
<summary>Click to expand</summary>

<br>
<img width="726" height="807" alt="santo_en" src="https://github.com/user-attachments/assets/8794fe34-088d-44b7-9450-c2c693aa6257" />
<img width="726" height="806" alt="santo_pl" src="https://github.com/user-attachments/assets/7ddb780f-1ee8-4b91-a928-f8a9638a8206" />
<img width="730" height="805" alt="crim_en" src="https://github.com/user-attachments/assets/0d7bab6c-ce89-444e-8eb3-180361d3dfc4" />
<img width="728" height="806" alt="crim_fr" src="https://github.com/user-attachments/assets/65025235-22e0-4362-a7cd-000495462fbb" />
<img width="729" height="805" alt="ennew" src="https://github.com/user-attachments/assets/51d10874-0d9c-487f-98f7-4c03b5e3cf5b" />
<img width="729" height="806" alt="gernew" src="https://github.com/user-attachments/assets/a0a72762-8263-46fb-b719-fdff41f4b534" />
<img width="728" height="806" alt="zielo_en" src="https://github.com/user-attachments/assets/8c1f3ff2-a305-4188-a955-e68fafe4ab8d" />
<img width="729" height="806" alt="zielo_es" src="https://github.com/user-attachments/assets/687303ad-b221-4610-abad-f54fab012755" />

</details>

---

## Changelog

<details>
<summary><b>Version history</b> (click to expand)</summary>

<br>

### [1.0.0] - 2026-10-23
**Added**
- Initial public release version 1.0.0.

</details>

---

## Supported formats

| Format | Extensions | Notes |
|--------|------------|-------|
| EPUB | `.epub` | Full HTML structure preserved; two processing modes (see below) |
| SRT | `.srt` | Subtitle blocks with timestamps; translated lines split proportionally |
| TXT | `.txt` | Plain text, paragraph-by-paragraph |
| PDF | `.pdf` | Text extracted by page block; layout preserved on PDF→PDF save |
| Kindle | `.mobi` `.azw` `.azw3` | Extracted internally; AZW3 unpacked as EPUB structure |
| FictionBook | `.fb2` | XML-based; sections, titles, paragraphs and epigraphs preserved |
| Word | `.docx` | Heading styles detected automatically (`Heading 1`, `Heading 2`, etc.) |
| ASS | `.ass` | Advanced SubStation Alpha subtitles; dialogue events with timestamps preserved, inline style tags (`{\i1}`, `{\b1}`, `{\u1}`, `{\p1}` drawings) handled by the parser |

---

## LLM backends

| Backend | Connection | Notes |
|---------|------------|-------|
| LM Studio | `http://localhost:1234/v1/chat/completions` | Default. Any model loaded in LM Studio. |
| Ollama | `http://localhost:11434/api/generate` | Requires model name (e.g. `llama3.2:3b`) |
| OpenRouter | Cloud API | Requires API key and model name (e.g. `openai/gpt-4o`, `openai/gpt-4o:free`) |

OpenRouter requests include automatic rate-limit detection and retry with backoff.

---

## JSON Payload mode

For models or endpoints that require a custom JSON input format (e.g. local inference servers with non-standard APIs), the prompt editor can be switched to **JSON Payload mode**. In this mode the entire request body is defined as a raw JSON template instead of role-based chat messages.

Available template variables:

| Variable | Description |
|----------|-------------|
| `{core_text}` | The text to translate |
| `{context_before}` | Previous paragraphs (read-only context) |
| `{context_after}` | Following paragraphs (read-only context) |

`temperature` is injected automatically from the UI slider and must not be included in the template.

The **Response field** setting (dot-notation path) tells the application where to find the translation in the server's response. Leave it empty for auto-detection of common keys (`translation`, `text`, `result`, `output`, `translated`).

Examples:

| Response field | Matches |
|----------------|---------|
| *(empty)* | Auto-detect common keys |
| `translation` | `{"translation": "..."}` |
| `choices.0.message.content` | Standard OpenAI-style chat response |
| `choices.0.message.content.translation` | JSON string nested inside `content` |

---

**Quick translation** (no LLM required):

| Service | Notes |
|---------|-------|
| Google Translate | Free, via `deep-translator` |
| DeepL Free | API key required |
| DeepL Pro | API key required |

Quick translation supports single fragments or all checked fragments in bulk. A character-count warning is shown before bulk runs.

---

## EPUB processing

EPUB files contain HTML with inline formatting tags (`<i>`, `<b>`, `<span>`, `<em>`, `<strong>`, `<u>`, `<sup>`, `<sub>`, `<small>`) that should not be sent to an LLM as raw HTML. The application replaces them with numbered placeholders before translation and restores them afterward.

> **Note — configurable tag names.**
> All placeholder names shown on this page (`id_`, `p_`, `nt_`, `ps`, `z`) and the
> response wrapper (`translated`) are **defaults**. They can all be renamed in
> **Options → Tag Profile (Advanced)** if your model mangles the defaults.
>
> **If you change any of them, you must:**
> 1. Update your prompt files (`.txt`) so the LLM uses the new names.
> 2. Reload the current EPUB (Open File → same file) — existing fragments keep
>    the *old* placeholder numbering in memory until reload.
> 3. Saved sessions and already-translated fragments are unaffected — they
>    retain the tag names they were created with.
>
> The parser has a hard-coded fallback to `<translated>` / `<translation>` for
> the response wrapper, so an accidental mismatch there is non-fatal. Placeholder
> prefixes have **no fallback** — a mismatch will leave raw `<p_01>` tokens in
> the translated output.

### Placeholder reference

| Placeholder | Default | Setting key (in `tag_profile`) | UI label |
|---|---|---|---|
| `<id_01>` | `id_` | `reserve_prefix` | Reserve elements prefix |
| `<p_01>...</p_01>` | `p_` | `inline_prefix` | Inline formatting prefix |
| `<nt_01/>` | `nt_` | `non_trans_prefix` | Non-translatable marker prefix |
| `<ps>` | `ps` | `paragraph_separator` | Paragraph separator |
| `<z>` | `z` | `batch_separator` | Batch separator |
| `<translated>` | `translated` | `response_wrapper_tag` (top-level, *not* in `tag_profile`) | Response Wrapper Tag |

### Inline mode (recommended for larger LLMs)

All inline formatting tags are replaced with paired placeholders: `<p_01>...</p_01>`, `<p_02>...</p_02>`, etc. *(default prefix: `p_`)*
Non-translatable content (padding spaces, empty anchors) is marked as `<nt_01/>` *(default prefix: `nt_`)*.
Structural elements (`img`, `code`, `br`, `hr`, `kbd`, `abbr`, `wbr`, `var`, `canvas`, `svg`, `script`, `style`, `math`) become `<id_01>` reserve markers *(default prefix: `id_`)*.

After translation, placeholders are resolved back to the original tags in their correct positions.

This mode relies on the model correctly preserving structured placeholder tokens.
It works best with larger, instruction-following LLMs that can reliably handle synthetic markers in the text. Smaller or lightweight models may occasionally drop, duplicate, or reorder placeholders.

### Legacy mode

Only structural reserve elements are protected (`<id_xx>`). Inline tags are not replaced with placeholders and are instead re-inserted after translation using a multilingual alignment model.

### Tags that can be individually skipped

In Inline mode, specific tags can be excluded from placeholder substitution if an EPUB has broken or unreliable markup:  
`<span>` · `<i>` · `<b>` · `<em>` · `<strong>` · `<u>` · `<sup>` · `<sub>` · `<small>`

> `<a>` (inline links) cannot be skipped — it is always converted to a
> placeholder because link targets must be preserved verbatim for the EPUB
> writer. All other inline formatting tags above are toggleable.

### Paragraph structure

Multi-paragraph elements can optionally preserve their internal paragraph breaks:
- **Inline mode:** breaks are sent to the LLM as `<ps>` markers *(default separator: `ps` — configurable as `paragraph_separator` in Tag Profile)* and restored after translation.
- **Legacy / TXT:** breaks are restored by proportional word-count split after translation.

When disabled, all newlines within a fragment are flattened before sending.

---

## Tag alignment (EPUB Legacy mode)

When saving an EPUB translated in Legacy mode, the application can run a multilingual transformer model to compute word-level semantic embeddings for the original and translated text, find word correspondences, and insert inline tags (`<i>`, `<b>`, `<span>`, etc.) at the correct positions in the translation.

**Supported models (downloaded from HuggingFace Hub, stored locally):**

| Model | Size | Notes |
|-------|------|-------|
| `bert-base-multilingual-cased` | ~700 MB | CPU-friendly |
| `microsoft/mdeberta-v3-base` | ~1.0 GB | CPU-friendly |
| `xlm-roberta-base` | ~1.1 GB | |
| `xlm-roberta-large` | ~2.4 GB | Default. GPU recommended. |

Models are stored in `<app_directory>/models/<model_name>/`. Each model has its own subfolder; changing the model name does not overwrite previously downloaded models. CUDA is supported.

The alignment step runs after translation, as a batch process at save time. Paragraphs without inline tags and those containing reserve elements are skipped automatically.

### Alignment quality indicators

Each translated fragment in Legacy mode displays a coloured dot showing its alignment status:

| Dot | Meaning |
|-----|---------|
| ○ | Not yet aligned (plain text will be saved) |
| 🟢 | Auto-wrap applied, or manually confirmed as correct |
| 🟡 | Neural model aligned — probably OK (0–1 corrections) |
| 🟠 | Neural model aligned — worth checking (2+ corrections) |
| ⚫ | No inline tags in original — alignment not applicable |
| 🔴 | Manually flagged as bad — always excluded from save |

When saving a Legacy-mode EPUB that contains aligned fragments, a dialog lets you choose which quality levels to include. Fragments whose dot is not selected are saved with plain translated text instead of aligned HTML.

---

## Mismatch detection

After each translation the application runs a series of checks and flags the fragment if any check fails. Flagged fragments are shown in red in the list.

> All checks below can be toggled in **Options → Mismatch Detection Settings**.
> Placeholder names (`<id_xx>`, `<nt_xx/>`, `<p_xx>`, `<ps>`) follow the
> configurable Tag Profile — see [EPUB processing](#epub-processing).

| Check | Description | Applies to | Toggle | Extra thresholds |
|-------|-------------|:---:|:---:|:---:|
| Paragraph / line count | Number of paragraphs differs between original and translation | All | ✓ | — |
| First character type | Type of the leading character changed (uppercase, lowercase, digit, quote, special) | All | ✓ | — |
| Last character type | Ending punctuation type changed | All | ✓ | — |
| Length ratio | Translation length is disproportionate to the original | All | ✓ | ✓ |
| Quote parity | Odd number of double quotation marks in the translation | All | ✓ | — |
| Untranslated | Translation is identical to the original (ignores short texts, proper nouns, URLs, single tokens) | All | ✓ | ✓ |
| Reserve elements `<id_xx>` | Structural placeholders missing, duplicated, or spurious closing tags added | EPUB only | ✓ | ✓ |
| NT markers `<nt_xx/>` | Non-translatable markers missing or extra | EPUB Inline only | ✓ | ✓ |
| Inline formatting `<p_xx>` | Opening/closing tags missing, extra, or unpaired | EPUB Inline only | ✓ | ✓ |
| PS markers `<ps>` | Paragraph breaks present in original not reflected in translation | EPUB Inline only | ✗ *always on* | — |

**Configurable thresholds:**

- **Length ratio (upper bound):** three separate thresholds depending on original text length — short (≤100 chars, default 1.6), medium (≤500 chars, default 1.4), long (>500 chars, default 1.3). Texts ≤20 chars are always skipped.
- **Length ratio (lower bound / "too short"):** translation is flagged when shorter than a fraction of the original (default 0.45). Applied only when original ≥8 chars. Catches translations that skipped a large portion of the source.
- **Untranslated ratio:** minimum fraction of lowercase-initial words required to trigger the "untranslated" check (default 0.30).
- **Tag position shift for `<id_xx>` / `<nt_xx/>`:** default 0.15 — structural markers should stay put.
- **Tag position shift for `<p_xx>` (inline formatting):** default 0.30 — higher tolerance because translated words have different lengths.

A fragment can be manually marked as correct (suppresses all mismatch flags, stored as `ignore_mismatch`) or flagged for review regardless of check results (stored as `force_mismatch`).

---

## Auto-fix

If mismatch is detected and Auto-fix is enabled, the application retries the translation automatically. On each retry, the specific error details are appended to the prompt so the model can correct them. Temperature increases slightly on each attempt to discourage identical outputs. After all attempts are exhausted, the result with the fewest mismatch errors is returned.

- Max attempts: 1–999 (default: 3)
- Temperature increment: configurable per attempt

---

## Prompts

The application maintains separate prompt variants for each file type and processing mode:

| Variant | Used for |
|---------|----------|
| `epub_inline` | EPUB in Inline mode |
| `epub_legacy` | EPUB in Legacy mode, AZW3 |
| `srt` | SRT and ASS subtitle files |
| `txt` | TXT, PDF, FB2, DOCX, MOBI |

ASS files share the `srt` variant because both formats use the same line-by-line subtitle model: translation operates on `subtitle_block` elements, output lines are re-split by proportional word count, and inline style tags — `<i>`/`<b>` in SRT, `{\i1}`/`{\b1}`/`{\p1}` drawings in ASS — are re-inserted positionally by the writer (not via `<p_xx>` placeholders). Prompts therefore do not need to know which of the two formats produced the text.

The variant is selected automatically based on the file type and, for Kindle files, on what was found inside the archive after extraction. AZW3 files unpack internally as an EPUB structure and are processed in Legacy mode — reserve element placeholders (`<id_xx>`) may be present, so they use `epub_legacy`. Plain MOBI files unpack as HTML and are processed as plain text with no placeholders, so they use `txt`. PDF, FB2 and DOCX processors also extract plain text only. Writers restore non-image reserve elements (`<br>`, `<code>`, etc.) where the output format supports them, and silently drop image placeholders since the new file has no embedded image resources.

For LM Studio and OpenRouter, prompts are split into **System**, **Assistant (context/instruction)**, and **User** roles.  
For Ollama, a single combined prompt is used.

**Single-prompt mode** merges all parts into one string — for instruct-only models that do not support role-based messages (e.g. Gemma instruct). Reduces "Channel Error"-type failures with some local models.

Prompts can be edited directly in the application's built-in editor, saved to disk as `.txt` files, reset to the last saved state, or hard-reset (deletes saved files and restores factory defaults).

---

## Context window

Each fragment is sent to the LLM with optional surrounding context:
- **Previous paragraphs:** 0–∞ (default: 3)
- **Next paragraphs:** 0–∞ (default: 2)

Context is provided as read-only reference — only the current fragment is translated.

---

## Sentence batching

Multiple fragments can be combined into a single LLM request to reduce API call overhead and speed up translation. Fragments are joined with a `<z>` separator marker; the model is instructed to preserve all markers and return the same number of segments. The response is then split back and mapped to the original fragments.

- **Batch size:** 2–999 fragments per request (default: 5)
- Works in both **Inline** and **Legacy** mode
- Fragments from different chapters are never merged into the same batch
- Available for EPUB files only (disabled automatically for other formats)
- When batching is active, the context window controls are hidden — context is implicit within the batch

If the model returns a wrong number of `<z>` separators, the batch is rejected and the fragments are retried individually.

---

## Session management

The current translation state can be saved to a JSON file and restored later. The session includes:
- All paragraphs (original text, translated text, mismatch flags, translation status)
- Translation settings (temperature, context window, processing mode, prompt variant)
- Custom prompts

On load, the application re-parses the original file to reconstruct the internal book object (required for saving) and remaps paragraph IDs by matching normalized original text.

Sessions are stored as plain JSON and are not tied to a specific file path — the original file location is confirmed on load via a file picker.

---

## Requirements

**Python 3.10+**

```
PyQt6
PyQt6-WebEngine
lxml
ebooklib
requests
deep-translator
deepl
openrouter
PyMuPDF
mobi
python-docx
```

---

## Installation

### Windows

1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/) — check *"Add Python to PATH"* during setup.

2. Clone the repository:
   ```bat
   git clone https://github.com/Mubumbutu/EPUB-SRT-LLM-Translator.git
   cd EPUB-SRT-LLM-Translator
   ```

3. Run `install.bat`. The installer will:
   - Create a virtual environment in `venv/`
   - Detect whether an NVIDIA GPU is present and which CUDA version the driver supports
   - Ask whether to install the CPU or GPU (CUDA) variant of PyTorch
   - Install all dependencies from `requirements.txt`
   - Verify the installation

   CUDA version is selected automatically based on the detected driver:

   | Driver version | CUDA |
   |----------------|------|
   | ≥ 550 | CUDA 12.4 |
   | ≥ 525 | CUDA 12.1 |
   | ≥ 450 | CUDA 11.8 |
   | < 450 | CPU fallback |

4. **Run:** double-click `launcher.vbs` — starts the application from the virtual environment without a console window.

---

### Linux

```bash
git clone https://github.com/Mubumbutu/EPUB-SRT-LLM-Translator.git
cd EPUB-SRT-LLM-Translator

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# CPU (always works):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU — check your driver version with: nvidia-smi
# CUDA 12.4 (driver ≥ 550):
# pip install torch --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.1 (driver ≥ 525):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8 (driver ≥ 450):
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Run:
```bash
source venv/bin/activate
python app.py
```

---

### macOS

```bash
git clone https://github.com/Mubumbutu/EPUB-SRT-LLM-Translator.git
cd EPUB-SRT-LLM-Translator

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install torch
```

> **Note:** CUDA is not available on macOS. Tag alignment (EPUB Legacy mode) will run on CPU only, which is noticeably slower for large files. Apple Silicon (M1/M2/M3) is not explicitly configured for MPS acceleration.

Run:
```bash
source venv/bin/activate
python app.py
```

---

> `torch` and `transformers` are required **only** for tag alignment in EPUB Legacy mode. The rest of the application works without them.

Make sure your chosen LLM backend (LM Studio / Ollama) is running before starting a translation session.

---

## Preview and reader

### Inline preview (EPUB)

The main window displays an inline HTML preview of the currently selected EPUB fragment, rendered with the actual EPUB stylesheet. Translated paragraphs are injected live into the preview — no save required. Clicking a paragraph in the preview selects the corresponding fragment in the list. Right-clicking opens a context menu with quick actions (re-translate, mark as correct, etc.).

The preview toolbar provides:
- **Refresh** — regenerate preview for the current fragment
- **Chapter navigation** — jump to previous / next chapter
- **Dark mode toggle** — switches the preview to dark background

### Full-screen reader

A separate full-screen reader window can be opened from the preview toolbar. The reader opens modeless alongside the main application — translations are pushed in live as they complete without interrupting reading.

**EPUB reader** features:
- Chapter navigation with `←` / `→` keys or on-screen arrows
- Dark mode and sepia mode
- Drag-to-scroll
- Status bar showing translated / total fragment count for the current chapter

**Generic reader** (FB2, DOCX, MOBI, PDF) works identically to the EPUB reader but uses format-specific rendering engines.

---

## Workflow

```
Open file → Select fragments → Configure LLM → Translate → Review → Save file
```

1. **Open file** — EPUB, SRT, TXT, ASS, PDF, MOBI/AZW, FB2 or DOCX via `📂 Open File`.
2. **Options tab** — select backend, enter API keys or model name, click **Save Settings**.
3. **Select fragments** — checkboxes in the list; `Select All`, `Select Untranslated`, `Select Mismatch`, or Shift+click for range selection. Searchable by original or translated text.
4. **Translate** — `▶ Translate Selected`. Status bar shows fragment index, progress count, elapsed time, timeout, and auto-fix attempt number.
5. **Review** — click a fragment to see original and translation side by side. The translation panel is editable; changes are applied immediately.
6. **Handle mismatches** — red fragments have detected problems. Hover for details. Options: edit manually, re-translate, mark as correct (ignore), or flag for review.
7. **Save** — `💾 Save as New File`. A dialog lets you choose the output format. For EPUB Legacy with translations, an additional dialog offers to run tag alignment before saving.

---

## Notes

- **OpenRouter free models (`:free`)** — require enabling *"Allow free endpoints to publish prompts"* in [OpenRouter privacy settings](https://openrouter.ai/settings/privacy). Without it, all requests return 404.
- **Alignment and VRAM** — the alignment model loads into the same GPU as the LLM. Shut down LM Studio or Ollama before running alignment to avoid out-of-memory errors.
- **Switching Inline ↔ Legacy** — if a file is already loaded, the mode change dialog offers to reload immediately. Translating without reloading uses the new prompts but the old placeholder structure.
- **Ruby annotations** — `<rt>` and `<rp>` tags (Japanese furigana) are stripped during EPUB parsing and not included in fragments.
- **MOBI/AZW write** — saving back to MOBI or AZW format is not supported. Amazon has not published a write specification; use EPUB as the output format for Kindle content.
- **KFX** — Kindle Format 10 is not supported. It requires Calibre, which is not a dependency of this application.
- **PDF with DRM** — encrypted PDFs cannot be processed. Remove DRM before opening.
- **FB2 compressed** — gzip-compressed `.fb2.zip` files are not supported directly; extract the `.fb2` file first.

---

## License

[GNU General Public License v3.0](LICENSE)
