# format_alignment.py
from __future__ import annotations
import gc
import logging
import os
import re
import time
import torch
import torch.nn.functional as F
from collections import Counter
from dataclasses import dataclass, replace
from lxml import etree
from transformers import AutoModel, AutoTokenizer
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

INLINE_TRANSFER_TAGS = {
    'i', 'em', 'b', 'strong', 'u', 'span', 'small', 'mark',
    'cite', 'q', 's', 'del', 'ins'
}
RESERVE_TAGS = {
    'img', 'code', 'br', 'hr', 'sub', 'sup', 'kbd',
    'abbr', 'wbr', 'var', 'canvas', 'svg', 'script',
    'style', 'math', 'a'
}
EXTENDED_TRANSFER_TAGS = INLINE_TRANSFER_TAGS | {'a', 'sub', 'sup', 'code'}
EXTENDED_RESERVE_TAGS = RESERVE_TAGS - {'a', 'sub', 'sup', 'code'}
XHTML_NS = 'http://www.w3.org/1999/xhtml'
MODELS_SUBDIR = 'models'

_RE_APOSTROPHES  = re.compile(r"['\u2019\u2018\u02bc\u0060\u00b4]")
_RE_PUNCT_STRIP  = re.compile(r'^[.,;:!?"\(\)\[\]\-]+|[.,;:!?"\(\)\[\]\-]+$')
_RE_SENTENCE_END = re.compile(r'[.!?]["\'\)\]]*$')
_RE_ABBREV       = re.compile(r'^[A-Za-z]{1,3}\.$|^\d+\.$')
_RE_ELLIPSIS     = re.compile(r'\.\.\.$')
_RE_CONTENT_WORD = re.compile(r'\w', re.UNICODE)
_HYPHEN_JOINERS  = frozenset({'-', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015'})
ALIGNMENT_MAX_LENGTH = 512


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    align_layers: Optional[Tuple[int, ...]] = None
    pooling: str = 'mean'
    word_id_source: str = 'auto'

    def __post_init__(self) -> None:
        if self.pooling not in ('mean', 'first'):
            raise ValueError(
                f"Unsupported pooling '{self.pooling}' for model '{self.model_id}'. "
                f"Use 'mean' or 'first'."
            )
        if self.word_id_source not in ('auto', 'legacy'):
            raise ValueError(
                f"Unsupported word_id_source '{self.word_id_source}' for model "
                f"'{self.model_id}'. Use 'auto' or 'legacy'."
            )


AVAILABLE_MODELS: Dict[str, ModelSpec] = {
    'xlm-roberta-large': ModelSpec(
        model_id='xlm-roberta-large',
        word_id_source='legacy',
    ),
    'FacebookAI/xlm-roberta-base': ModelSpec(model_id='FacebookAI/xlm-roberta-base'),
    'microsoft/xlm-align-base': ModelSpec(model_id='microsoft/xlm-align-base'),
    'microsoft/infoxlm-base': ModelSpec(model_id='microsoft/infoxlm-base'),
    'microsoft/infoxlm-large': ModelSpec(model_id='microsoft/infoxlm-large'),
    'microsoft/mdeberta-v3-base': ModelSpec(model_id='microsoft/mdeberta-v3-base'),
}


def resolve_model_spec(model_name: str) -> ModelSpec:
    name = (model_name or '').strip()
    spec = AVAILABLE_MODELS.get(name)
    if spec is not None:
        return spec
    return ModelSpec(model_id=name, word_id_source='legacy')


HF_CACHE_SUBDIR = '_hf_cache'


def _get_hf_cache_dir(models_dir: str) -> str:
    return os.path.join(models_dir, HF_CACHE_SUBDIR)


def _load_tokenizer(source: str, cache_dir: Optional[str] = None):
    errors: List[str] = []
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                source,
                use_fast=use_fast,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            errors.append(f"use_fast={use_fast}: {exc}")
    raise RuntimeError(
        f"Could not load tokenizer from '{source}'. "
        f"SentencePiece-based models may require the 'sentencepiece' and "
        f"'protobuf' packages. Details: " + ' | '.join(errors)
    )


def _load_encoder(source: str, cache_dir: Optional[str] = None):
    return AutoModel.from_pretrained(
        source,
        output_hidden_states=True,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )


@dataclass
class WordEmbeddings:
    words: List[str]
    vectors: Any
    token_word_ids: List[Optional[int]]
    word_token_counts: Any


class WordEmbeddingAdapter:
    _ALIGN_LAYER_FALLBACK = 8

    def __init__(
        self,
        model_name: str,
        device: str = 'cpu',
        local_models_dir: str = '',
        spec: Optional[ModelSpec] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_models_dir = local_models_dir
        self.spec = spec if spec is not None else resolve_model_spec(model_name)
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.sp_model = False
        self.align_layers: List[int] = []

    def load(self) -> None:
        os.makedirs(self.local_models_dir, exist_ok=True)
        cache_dir = _get_hf_cache_dir(self.local_models_dir)
        os.makedirs(cache_dir, exist_ok=True)
        local_path = get_local_model_path(self.model_name, self.local_models_dir)

        if is_model_downloaded(self.model_name, self.local_models_dir):
            load_source = local_path
            logger.info(f"[FormatAlignment] Loading model from local directory: {local_path}")
        else:
            load_source = self.spec.model_id
            logger.info(
                f"[FormatAlignment] Local model not found at: {local_path}\n"
                f"[FormatAlignment] Loading from HuggingFace Hub: '{self.spec.model_id}'"
            )

        logger.info(f"[FormatAlignment] Device: {self.device}")
        logger.info(f"[FormatAlignment] HF cache dir: {cache_dir}")

        try:
            tokenizer = _load_tokenizer(load_source, cache_dir=cache_dir)
            model = _load_encoder(load_source, cache_dir=cache_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self.model_name}' from '{load_source}': {exc}"
            ) from exc

        if load_source != local_path:
            try:
                os.makedirs(local_path, exist_ok=True)
                tokenizer.save_pretrained(local_path)
                model.save_pretrained(local_path)
                logger.info(f"[FormatAlignment] Model saved to local directory: {local_path}")
            except Exception as exc:
                logger.warning(
                    f"[FormatAlignment] Could not save model to {local_path}: {exc}"
                )

        model.eval()
        model.to(torch.device(self.device))

        self.tokenizer = tokenizer
        self.model = model
        self.loaded = True

        self.detect_tokenizer_type()
        self.detect_align_layers()

        logger.info(
            f"[FormatAlignment] Model '{self.model_name}' loaded successfully "
            f"on '{self.device}'. "
            f"Alignment layers: {self.align_layers}. "
            f"SentencePiece: {self.sp_model}. "
            f"Pooling: {self.spec.pooling}. "
            f"Word id source: {self.spec.word_id_source}."
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.align_layers = []

    def detect_tokenizer_type(self) -> None:
        test_tokens = self.tokenizer.tokenize("hello world")
        self.sp_model = any(t.startswith('\u2581') for t in test_tokens)
        logger.debug(
            f"[FormatAlignment] Test tokens: {test_tokens} | "
            f"SentencePiece={self.sp_model}"
        )

    def detect_align_layers(self) -> None:
        if self.spec.align_layers:
            self.align_layers = list(self.spec.align_layers)
            logger.info(
                f"[FormatAlignment] Using configured alignment layers: {self.align_layers}"
            )
            return

        n_layers = None

        try:
            cfg = self.model.config
            if hasattr(cfg, 'encoder_layers'):
                n_layers = cfg.encoder_layers
            elif hasattr(cfg, 'num_hidden_layers'):
                n_layers = cfg.num_hidden_layers
        except Exception as exc:
            logger.warning(f"[FormatAlignment] Failed to read layer count from config: {exc}")

        if n_layers and n_layers > 0:
            target = max(2, round(n_layers / 3))
            first = max(1, target - 1)
            last = min(n_layers, target + 1)
            self.align_layers = list(range(first, last + 1))
        else:
            logger.warning(
                f"[FormatAlignment] Unknown layer count – "
                f"using fallback layer {self._ALIGN_LAYER_FALLBACK}"
            )
            self.align_layers = [self._ALIGN_LAYER_FALLBACK]

        logger.info(
            f"[FormatAlignment] Detected {n_layers} layers. "
            f"Alignment layers: {self.align_layers}"
        )

    def get_layer_output(self, outputs) -> 'torch.Tensor':
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            available = len(outputs.hidden_states)
            layers = [l for l in self.align_layers if l < available]
            if not layers:
                layers = [available - 1]
            stacked = torch.stack(
                [outputs.hidden_states[l][0] for l in layers], dim=0
            )
            return stacked.mean(dim=0)

        if (hasattr(outputs, 'encoder_hidden_states')
                and outputs.encoder_hidden_states is not None):
            available = len(outputs.encoder_hidden_states)
            layers = [l for l in self.align_layers if l < available]
            if not layers:
                layers = [available - 1]
            stacked = torch.stack(
                [outputs.encoder_hidden_states[l][0] for l in layers], dim=0
            )
            return stacked.mean(dim=0)

        if (hasattr(outputs, 'last_hidden_state')
                and outputs.last_hidden_state is not None):
            logger.debug("[FormatAlignment] Using last_hidden_state as fallback")
            return outputs.last_hidden_state[0]

        raise RuntimeError(
            "[FormatAlignment] Model returned no hidden_states. "
            "Make sure the model is loaded with output_hidden_states=True."
        )

    def build_word_ids(
        self,
        words: List[str],
        tokens: List[str],
    ) -> List[Optional[int]]:
        word_ids: List[Optional[int]] = [None] * len(tokens)

        try:
            special_tokens: set = set(self.tokenizer.all_special_tokens)
        except Exception:
            special_tokens = {
                '[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]',
                '<s>', '</s>', '<pad>', '<mask>', '<unk>',
            }

        if self.sp_model:
            word_idx = -1
            first_content_seen = False
            prev_bare_tok: Optional[str] = None

            for tok_pos, tok in enumerate(tokens):
                if tok in special_tokens:
                    continue

                bare_tok = tok.lstrip('\u2581')
                starts_new_word = tok.startswith('\u2581') or not first_content_seen

                if starts_new_word and prev_bare_tok in _HYPHEN_JOINERS:
                    starts_new_word = False

                if starts_new_word:
                    word_idx += 1
                    first_content_seen = True

                if 0 <= word_idx < len(words):
                    word_ids[tok_pos] = word_idx

                prev_bare_tok = bare_tok

        else:
            word_idx = -1

            for tok_pos, tok in enumerate(tokens):
                if tok in special_tokens:
                    continue

                if not tok.startswith('##'):
                    word_idx += 1

                if 0 <= word_idx < len(words):
                    word_ids[tok_pos] = word_idx

        return word_ids

    def _hf_word_ids(self, encoding, n_words: int) -> Optional[List[Optional[int]]]:
        if not getattr(self.tokenizer, 'is_fast', False):
            return None
        try:
            ids = list(encoding.word_ids(0))
        except Exception:
            return None
        if len(ids) != int(encoding['input_ids'].shape[1]):
            return None
        if sorted({i for i in ids if i is not None}) != list(range(n_words)):
            return None
        return ids

    def embed_words(self, words: List[str]) -> Optional[WordEmbeddings]:
        if not words:
            return None
        if not self.loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        text = ' '.join(words)
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            max_length=ALIGNMENT_MAX_LENGTH,
        )

        hf_word_ids: Optional[List[Optional[int]]] = None
        if self.spec.word_id_source == 'auto':
            hf_word_ids = self._hf_word_ids(encoding, len(words))
            if hf_word_ids is None:
                logger.debug(
                    "[FormatAlignment] HF word_ids unavailable or inconsistent – "
                    "using legacy word id builder"
                )

        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding, output_hidden_states=True)

        hidden = self.get_layer_output(outputs)

        token_ids = encoding['input_ids'][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        if hf_word_ids is not None:
            word_ids = hf_word_ids
        else:
            word_ids = self.build_word_ids(words, tokens)

        H = hidden.shape[-1]
        word_embs = torch.zeros(len(words), H, device=self.model.device)
        word_counts = torch.zeros(len(words), device=self.model.device)
        first_only = self.spec.pooling == 'first'

        for tok_idx, wi in enumerate(word_ids):
            if wi is None or not 0 <= wi < len(words):
                continue
            if first_only and word_counts[wi] > 0:
                continue
            word_embs[wi] += hidden[tok_idx]
            word_counts[wi] += 1

        if word_counts.sum().item() == 0:
            logger.warning(
                f"[FormatAlignment] _get_word_embeddings: no word received an embedding "
                f"(words={words[:5]}, tokens={tokens[:10]})"
            )
            return None

        divisor = word_counts.clamp(min=1.0)
        word_embs = word_embs / divisor.unsqueeze(-1)

        return WordEmbeddings(
            words=list(words),
            vectors=word_embs,
            token_word_ids=word_ids,
            word_token_counts=word_counts,
        )

def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace('/', '_').replace('\\', '_')

def get_local_model_path(model_name: str, models_dir: str) -> str:
    safe_name = _sanitize_model_name(model_name)
    return os.path.join(models_dir, safe_name)

def is_model_downloaded(model_name: str, models_dir: str) -> bool:
    local_path = get_local_model_path(model_name, models_dir)
    config_path = os.path.join(local_path, 'config.json')
    return os.path.isdir(local_path) and os.path.isfile(config_path)

def download_model(
    model_name: str = 'microsoft/xlm-align-base',
    models_dir: str = '',
) -> str:
    if not models_dir:
        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), MODELS_SUBDIR
        )

    spec = resolve_model_spec(model_name)

    os.makedirs(models_dir, exist_ok=True)
    cache_dir = _get_hf_cache_dir(models_dir)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"[FormatAlignment] Models directory: {models_dir}")
    logger.info(f"[FormatAlignment] HF cache directory: {cache_dir}")

    local_model_path = get_local_model_path(model_name, models_dir)
    os.makedirs(local_model_path, exist_ok=True)

    logger.info(f"[FormatAlignment] Downloading model '{spec.model_id}' → {local_model_path}")

    logger.info("[FormatAlignment] Downloading tokenizer...")
    tokenizer = _load_tokenizer(spec.model_id, cache_dir=cache_dir)
    tokenizer.save_pretrained(local_model_path)
    logger.info("[FormatAlignment] Tokenizer saved.")

    logger.info("[FormatAlignment] Downloading model (this may take several minutes)...")
    model = _load_encoder(spec.model_id, cache_dir=cache_dir)
    model.save_pretrained(local_model_path)
    logger.info(f"[FormatAlignment] Model saved to: {local_model_path}")

    del model
    del tokenizer
    gc.collect()

    return local_model_path

def _parse_element_html(html_str: str) -> Optional[etree._Element]:
    try:
        return etree.fromstring(html_str.encode('utf-8'))
    except etree.XMLSyntaxError:
        pass

    try:
        wrapped = f'<__root__ xmlns="{XHTML_NS}">{html_str}</__root__>'
        root = etree.fromstring(wrapped.encode('utf-8'))
        if len(root):
            return root[0]
    except etree.XMLSyntaxError:
        pass

    try:
        fixed = re.sub(
            r'^(<\w[^>]*?)>',
            lambda m: m.group(1) + f' xmlns="{XHTML_NS}">',
            html_str.strip(),
            count=1
        )
        return etree.fromstring(fixed.encode('utf-8'))
    except etree.XMLSyntaxError:
        pass

    return None

def _element_to_html(element: etree._Element) -> str:
    return etree.tostring(element, encoding='unicode', method='xml')

def _get_plain_text(element: etree._Element) -> str:
    text = etree.tostring(element, encoding='unicode', method='text')
    return re.sub(r'\s+', ' ', text).strip()

def _tokenize_words(text: str) -> List[str]:
    return [w for w in text.split() if w]

def _build_attrs_str(attrs: Dict[str, str]) -> str:
    if not attrs:
        return ''
    parts = []
    for k, v in attrs.items():
        if '{' in k:
            continue
        parts.append(f' {k}="{_escape_xml(v)}"')
    return ''.join(parts)

def _escape_xml(text: str) -> str:
    return (
        text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )

def _try_auto_wrap(
    original_html: str,
    translated_text: str,
    transfer_tags=INLINE_TRANSFER_TAGS,
) -> Optional[str]:
    root = _parse_element_html(original_html)
    if root is None:
        return None

    direct_text = (root.text or '').strip()
    children = list(root)

    if direct_text or len(children) != 1:
        return None

    child = children[0]
    tag_local = etree.QName(child).localname if not hasattr(child.tag, '__call__') else ''
    if tag_local not in transfer_tags:
        return None

    trailing = (child.tail or '').strip()
    if trailing:
        return None

    child_inline_children = [
        c for c in child
        if not callable(c.tag)
        and etree.QName(c).localname.lower() in transfer_tags
    ]
    if child_inline_children:
        return None

    attrs_str = _build_attrs_str(dict(child.attrib))

    escaped_trans = _escape_xml(translated_text.strip())

    root_local = etree.QName(root).localname if not hasattr(root.tag, '__call__') else 'p'
    root_attrs_str = _build_attrs_str(dict(root.attrib))

    return (
        f'<{root_local}{root_attrs_str}>'
        f'<{tag_local}{attrs_str}>{escaped_trans}</{tag_local}>'
        f'</{root_local}>'
    )

class FormatAlignmentEngine:
    _ALIGN_LAYER_FALLBACK = 8

    def __init__(
        self,
        model_name: str = 'microsoft/xlm-align-base',
        device: str = 'cpu',
        local_models_dir: str = '',
        model_spec: Optional[ModelSpec] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device

        if local_models_dir:
            self.local_models_dir = local_models_dir
        else:
            self.local_models_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), MODELS_SUBDIR
            )

        self.model_spec = model_spec if model_spec is not None else resolve_model_spec(model_name)
        self._adapter: Optional[WordEmbeddingAdapter] = None
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self._last_src_len: int = 1
        self._align_layers: List[int] = []
        self._sp_model: bool = False
        self._full_src_embs = None
        self._full_tgt_embs = None
        self._last_alignment_corrections: int = 0
        self._last_alignment_corrections_detail: List[str] = []
        self._last_sent_map: Dict[int, Tuple[int, int, int, int]] = {}
        self._transfer_tags = INLINE_TRANSFER_TAGS
        self._reserve_tags = RESERVE_TAGS

    def _set_tag_mode(self, extended: bool) -> None:
        if extended:
            self._transfer_tags = EXTENDED_TRANSFER_TAGS
            self._reserve_tags = EXTENDED_RESERVE_TAGS
        else:
            self._transfer_tags = INLINE_TRANSFER_TAGS
            self._reserve_tags = RESERVE_TAGS

    def load_model(self) -> None:
        adapter = WordEmbeddingAdapter(
            model_name=self.model_name,
            device=self.device,
            local_models_dir=self.local_models_dir,
            spec=self.model_spec,
        )
        adapter.load()

        self._adapter = adapter
        self.tokenizer = adapter.tokenizer
        self.model = adapter.model
        self._align_layers = list(adapter.align_layers)
        self._sp_model = adapter.sp_model
        self._loaded = True

    def _detect_tokenizer_type(self) -> None:
        if self._adapter is None:
            return
        self._adapter.detect_tokenizer_type()
        self._sp_model = self._adapter.sp_model

    def _detect_align_layers(self) -> None:
        if self._adapter is None:
            return
        self._adapter.detect_align_layers()
        self._align_layers = list(self._adapter.align_layers)

    def unload_model(self) -> None:
        if self._adapter is not None:
            self._adapter.unload()
        self._adapter = None
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._align_layers = []

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()
        logger.info("[FormatAlignment] Model unloaded from memory.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _get_layer_output(self, outputs) -> 'torch.Tensor':
        if self._adapter is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        return self._adapter.get_layer_output(outputs)

    def align_batch(self, paragraphs: List[Dict], extended_tags: bool = False) -> int:
        if not self._loaded:
            raise RuntimeError(
                "Model is not loaded. Call load_model() first."
            )

        self._set_tag_mode(extended_tags)

        processed = 0
        skipped_no_inline = 0
        skipped_reserve = 0
        skipped_other = 0

        for para in paragraphs:
            if para.get('processing_mode') != 'legacy':
                skipped_other += 1
                continue

            if not para.get('is_translated'):
                skipped_other += 1
                continue

            if para.get('is_non_translatable'):
                skipped_other += 1
                continue

            original_html = para.get('original_html', '').strip()
            translated_text = para.get('translated_text', '').strip()

            if not original_html or not translated_text:
                logger.debug(
                    f"[FormatAlignment] Para {para.get('id', '?')} – "
                    f"empty original_html or translated_text – skipping"
                )
                skipped_other += 1
                continue

            original_plain = re.sub(r'<[^>]+>', '', original_html)
            original_plain = re.sub(r'\s+', ' ', original_plain).strip()
            clean_translated = re.sub(r'</?id_\d{2}>', '', translated_text).strip()
            clean_translated = re.sub(r'\s+', ' ', clean_translated).strip()

            if not clean_translated:
                skipped_other += 1
                continue

            if clean_translated == original_plain:
                logger.debug(
                    f"[FormatAlignment] Para {para.get('id', '?')} – "
                    f"translated_text == original_text – skipping"
                )
                skipped_other += 1
                continue

            if not extended_tags and ('<a ' in original_html or '<a>' in original_html):
                logger.debug(
                    f"[FormatAlignment] Para {para.get('id', '?')} "
                    f"contains <a> – skipping alignment"
                )
                skipped_reserve += 1
                continue

            auto_wrapped = _try_auto_wrap(original_html, clean_translated, self._transfer_tags)
            if auto_wrapped is not None:
                para['aligned_translated_html'] = auto_wrapped
                para['alignment_corrections'] = 0
                para['alignment_uncertain'] = False
                para['alignment_auto_wrap'] = True
                processed += 1
                logger.debug(
                    f"[FormatAlignment] Para {para.get('id', '?')} "
                    f"auto-wrapped (full content in single inline tag) – skipped model."
                )
                continue

            if para.get('reserve_elements'):
                skipped_reserve += 1
                continue

            try:
                result_html = self.align_element(original_html, clean_translated, extended_tags)
                if result_html:
                    para['aligned_translated_html'] = result_html
                    _ALIGNMENT_UNCERTAIN_THRESHOLD = 2
                    corrections = self._last_alignment_corrections
                    para['alignment_corrections'] = corrections
                    if self._last_alignment_corrections_detail:
                        para['alignment_corrections_detail'] = list(
                            self._last_alignment_corrections_detail
                        )
                    if corrections >= _ALIGNMENT_UNCERTAIN_THRESHOLD:
                        para['alignment_uncertain'] = True
                        logger.debug(
                            f"[FormatAlignment] Para {para.get('id', '?')} "
                            f"UNCERTAIN (corrections={corrections}): "
                            f"{self._last_alignment_corrections_detail}"
                        )
                    else:
                        para['alignment_uncertain'] = False
                    processed += 1
                else:
                    skipped_no_inline += 1
            except Exception as exc:
                logger.warning(
                    f"[FormatAlignment] Error processing para {para.get('id', '?')}: {exc}",
                    exc_info=True
                )
                skipped_other += 1

        logger.info(
            f"[FormatAlignment] Batch complete: "
            f"processed={processed}, "
            f"skipped(reserve+links)={skipped_reserve}, "
            f"skipped(no inline tags)={skipped_no_inline}, "
            f"skipped(other/untranslated)={skipped_other}"
        )
        return processed

    def _split_to_sentence_spans(
        self,
        words: List[str],
    ) -> List[Tuple[int, int, List[str]]]:
        if not words:
            return []

        sentences: List[Tuple[int, int, List[str]]] = []
        current_start = 0

        for i, word in enumerate(words):
            is_last = (i == len(words) - 1)

            if is_last:
                sub = words[current_start:]
                sentences.append((current_start, i, sub))
                break

            if _RE_SENTENCE_END.search(word):
                if _RE_ABBREV.match(word) or _RE_ELLIPSIS.search(word):
                    continue

                next_word = words[i + 1]
                if not (next_word[0].isupper() or next_word[0].isdigit()):
                    continue

                sub = words[current_start: i + 1]

                if len(sub) < 3 and sentences:
                    prev_start, _, prev_sub = sentences[-1]
                    merged_sub = prev_sub + sub
                    sentences[-1] = (prev_start, i, merged_sub)
                else:
                    sentences.append((current_start, i, sub))

                current_start = i + 1

        if not sentences:
            return [(0, len(words) - 1, words)]

        return sentences

    def _micro_segment_align(
        self,
        s_words: List[str],
        t_words: List[str],
        s_offset: int,
        t_offset: int,
    ) -> List[Tuple[int, int]]:
        """Run alignment with micro-segmentation at strong forced anchors.

        Instead of running _compute_alignments on the whole sentence at once,
        we first locate "strong" anchors — exact normalized matches that are
        unique in both src and tgt and at least 4 characters long (proper
        nouns, rare content words).  The sentence is then split at those
        anchors into independent sub-problems, each solved separately.

        This dramatically reduces greedy competition: a repeated word like
        "grandmother's" can no longer steal the target slot of "<em>and …
        before that</em>" because the competition is now local to the
        sub-segment that contains the em span.

        Returns pairs with global paragraph offsets already applied.
        """
        def _norm(w: str) -> str:
            w = w.lower()
            w = _RE_APOSTROPHES.sub('', w)
            w = _RE_PUNCT_STRIP.sub('', w)
            return w

        S = len(s_words)
        T = len(t_words)

        if S < 2 or T < 2:
            pairs = self._compute_alignments(s_words, t_words)
            return [(s + s_offset, t + t_offset) for s, t in pairs]

        src_norm = [_norm(w) for w in s_words]
        tgt_norm = [_norm(w) for w in t_words]

        src_counts = Counter(src_norm)
        tgt_counts = Counter(tgt_norm)

        _MIN_ANCHOR_LEN = 4
        anchors_s2t: Dict[int, int] = {}
        used_tgt: set = set()

        for s_idx, s_n in enumerate(src_norm):
            if len(s_n) < _MIN_ANCHOR_LEN:
                continue
            if src_counts[s_n] > 1:
                continue
            for t_idx, t_n in enumerate(tgt_norm):
                if t_idx in used_tgt:
                    continue
                if t_n == s_n and tgt_counts[t_n] == 1:
                    anchors_s2t[s_idx] = t_idx
                    used_tgt.add(t_idx)
                    break

        if not anchors_s2t:
            pairs = self._compute_alignments(s_words, t_words)
            return [(s + s_offset, t + t_offset) for s, t in pairs]

        sorted_anchors: List[Tuple[int, int]] = []
        last_t = -1
        for s_a, t_a in sorted(anchors_s2t.items()):
            if t_a > last_t:
                sorted_anchors.append((s_a, t_a))
                last_t = t_a

        if not sorted_anchors:
            pairs = self._compute_alignments(s_words, t_words)
            return [(s + s_offset, t + t_offset) for s, t in pairs]

        result: List[Tuple[int, int]] = []

        for s_a, t_a in sorted_anchors:
            result.append((s_a + s_offset, t_a + t_offset))
            logger.debug(
                f"[FormatAlignment] MicroSeg anchor: '{s_words[s_a]}' "
                f"s={s_a} → '{t_words[t_a]}' t={t_a}"
            )

        boundaries = [(-1, -1)] + sorted_anchors + [(S, T)]

        for i in range(len(boundaries) - 1):
            prev_s, prev_t = boundaries[i]
            next_s, next_t = boundaries[i + 1]

            seg_s0 = prev_s + 1
            seg_s1 = next_s - 1
            seg_t0 = prev_t + 1
            seg_t1 = next_t - 1

            if seg_s0 > seg_s1 or seg_t0 > seg_t1:
                continue

            seg_s = s_words[seg_s0: seg_s1 + 1]
            seg_t = t_words[seg_t0: seg_t1 + 1]

            if not seg_s or not seg_t:
                continue

            try:
                pairs = self._compute_alignments(seg_s, seg_t)
                for s_loc, t_loc in pairs:
                    result.append((
                        s_loc + seg_s0 + s_offset,
                        t_loc + seg_t0 + t_offset,
                    ))
                logger.debug(
                    f"[FormatAlignment] MicroSeg [{seg_s0}-{seg_s1}]→"
                    f"[{seg_t0}-{seg_t1}]: {len(pairs)} pairs"
                )
            except Exception as exc:
                logger.warning(
                    f"[FormatAlignment] MicroSeg error at segment {i}: {exc}"
                )

        return result

    def _try_merge_sentences(
        self,
        longer: List[Tuple[int, int, List[str]]],
        shorter: List[Tuple[int, int, List[str]]],
    ) -> Optional[List[Tuple[int, int, List[str]]]]:
        """Try to merge one pair of consecutive sentences in `longer`
        to match len(shorter). Picks the merge that minimises total
        word-count difference vs the corresponding sentence in `shorter`.
        Returns the merged list, or None if diff != 1.
        """
        if len(longer) - len(shorter) != 1:
            return None

        best_merged: Optional[List] = None
        best_cost = float('inf')

        for merge_at in range(len(longer) - 1):
            candidate = list(longer)
            s0, e0, w0 = candidate[merge_at]
            s1, e1, w1 = candidate[merge_at + 1]
            candidate[merge_at:merge_at + 2] = [(s0, e1, w0 + w1)]

            cost = sum(
                abs(len(ms[2]) - len(ss[2]))
                for ms, ss in zip(candidate, shorter)
            )
            if cost < best_cost:
                best_cost = cost
                best_merged = candidate

        return best_merged

    def _compute_alignments_by_sentences(
        self,
        src_words: List[str],
        tgt_words: List[str],
    ) -> List[Tuple[int, int]]:
        MIN_WORDS_FOR_SENTENCE_SPLIT = 9

        self._last_sent_map = {}

        if len(src_words) < MIN_WORDS_FOR_SENTENCE_SPLIT:
            logger.debug(
                f"[FormatAlignment] Short text ({len(src_words)} words) "
                f"– alignment without sentence splitting"
            )
            result = self._compute_alignments(src_words, tgt_words)
            self._last_src_len = len(src_words)
            return result

        src_sentences = self._split_to_sentence_spans(src_words)
        tgt_sentences = self._split_to_sentence_spans(tgt_words)

        logger.debug(
            f"[FormatAlignment] Split: {len(src_sentences)} src sentences, "
            f"{len(tgt_sentences)} tgt sentences"
        )

        if len(src_sentences) != len(tgt_sentences):
            diff = len(tgt_sentences) - len(src_sentences)
            merged_ok = False

            if diff == 1:
                merged = self._try_merge_sentences(tgt_sentences, src_sentences)
                if merged is not None:
                    logger.debug(
                        f"[FormatAlignment] Merge tgt sentences: "
                        f"{len(tgt_sentences)}→{len(merged)} "
                        f"(matching {len(src_sentences)} src)"
                    )
                    tgt_sentences = merged
                    merged_ok = True
            elif diff == -1:
                merged = self._try_merge_sentences(src_sentences, tgt_sentences)
                if merged is not None:
                    logger.debug(
                        f"[FormatAlignment] Merge src sentences: "
                        f"{len(src_sentences)}→{len(merged)} "
                        f"(matching {len(tgt_sentences)} tgt)"
                    )
                    src_sentences = merged
                    merged_ok = True

            if not merged_ok:
                logger.debug(
                    f"[FormatAlignment] Sentence count mismatch src≠tgt "
                    f"({len(src_sentences)}≠{len(tgt_sentences)}) – full fallback"
                )
                result = self._compute_alignments(src_words, tgt_words)
                self._last_src_len = len(src_words)
                return result

        if len(src_sentences) == 1:
            result = self._compute_alignments(src_words, tgt_words)
            self._last_src_len = len(src_words)
            return result

        all_alignments: List[Tuple[int, int]] = []

        for sent_idx, (src_sent, tgt_sent) in enumerate(
            zip(src_sentences, tgt_sentences)
        ):
            s_start, s_end, s_words = src_sent
            t_start, t_end, t_words = tgt_sent

            if not s_words or not t_words:
                continue

            src_counter = Counter(w.lower() for w in s_words if len(w) >= 4)
            has_repetitions = any(v >= 2 for v in src_counter.values())

            _PATH_B_MAX_WORDS = 14
            if has_repetitions and len(s_words) <= _PATH_B_MAX_WORDS:
                logger.debug(
                    f"[FormatAlignment] Sentence {sent_idx}: repetitions "
                    f"({[k for k, v in src_counter.items() if v >= 2]}) "
                    f"– Path B ({len(s_words)} words): proportional per-sentence fallback"
                )
                s_len_sent = len(s_words)
                t_len_sent = len(t_words)
                for s_i in range(s_len_sent):
                    t_i = min(
                        t_len_sent - 1,
                        round(s_i / max(1, s_len_sent - 1) * (t_len_sent - 1)),
                    )
                    all_alignments.append((s_i + s_start, t_i + t_start))
                for s_i in range(s_start, s_end + 1):
                    self._last_sent_map[s_i] = (s_start, s_end, t_start, t_end)
                continue
            elif has_repetitions:
                logger.debug(
                    f"[FormatAlignment] Sentence {sent_idx}: repetitions "
                    f"({[k for k, v in src_counter.items() if v >= 2]}) "
                    f"– long sentence ({len(s_words)} words), proceeding with neural alignment"
                )

            try:
                seg_pairs = self._micro_segment_align(
                    s_words, t_words, s_start, t_start
                )
            except Exception as exc:
                logger.warning(
                    f"[FormatAlignment] Alignment error for sentence {sent_idx}: {exc}"
                )
                continue

            all_alignments.extend(seg_pairs)

            for s_i in range(s_start, s_end + 1):
                self._last_sent_map[s_i] = (s_start, s_end, t_start, t_end)

            logger.debug(
                f"[FormatAlignment] Sentence {sent_idx}: {len(seg_pairs)} pairs "
                f"(src[{s_start}:{s_end}] → tgt[{t_start}:{t_end}])"
            )

        self._last_src_len = len(src_words)
        logger.debug(
            f"[FormatAlignment] _last_src_len reset to {len(src_words)}"
        )

        return all_alignments

    def align_element(self, original_html: str, translated_text: str, extended_tags: bool = False) -> str:
        self._set_tag_mode(extended_tags)

        if not original_html or not translated_text:
            return ''

        self._last_alignment_corrections = 0
        self._last_alignment_corrections_detail = []

        orig_elem = _parse_element_html(original_html)
        if orig_elem is None:
            logger.debug("[FormatAlignment] Failed to parse original_html")
            return ''

        if not extended_tags:
            has_links = orig_elem.xpath('.//*[local-name()="a"]')
            if has_links:
                logger.debug("[FormatAlignment] Element contains <a> – skipping")
                return ''

        inline_spans = self._extract_inline_spans(orig_elem)
        if not inline_spans:
            return ''

        word_tag_pairs: List = []
        self._walk_collect_words(orig_elem, parent_tags=[], output=word_tag_pairs)
        src_words: List[str] = [word for word, _ in word_tag_pairs]

        tgt_words = _tokenize_words(translated_text)

        if len(src_words) < 1 or len(tgt_words) < 1:
            return ''

        if ' '.join(src_words).lower() == ' '.join(tgt_words).lower():
            logger.debug("[FormatAlignment] src_words == tgt_words – skipping")
            return ''

        try:
            self._full_src_embs = self._get_word_embeddings(src_words)
            self._full_tgt_embs = self._get_word_embeddings(tgt_words)

            alignments = self._compute_alignments_by_sentences(src_words, tgt_words)
        except Exception as exc:
            logger.warning(f"[FormatAlignment] Alignment error: {exc}")
            self._full_src_embs = None
            self._full_tgt_embs = None
            return ''

        if not alignments:
            return ''

        src_to_tgt: Dict[int, List[int]] = {}
        for (s, t) in alignments:
            src_to_tgt.setdefault(s, []).append(t)

        tgt_spans = self._transfer_spans(inline_spans, src_to_tgt, len(tgt_words), tgt_words, src_words)
        self._full_src_embs = None
        self._full_tgt_embs = None
        if not tgt_spans:
            return ''

        tgt_spans = self._enforce_nesting(inline_spans, tgt_spans)

        return self._build_result_html(orig_elem, tgt_words, tgt_spans)

    def _enforce_nesting(
        self,
        src_spans: List[Dict],
        tgt_spans: List[Dict],
    ) -> List[Dict]:
        """Ensure child spans stay within their parent span after transfer.

        When the original HTML has nested inline tags, e.g.:
            <b>"He mentioned <em>everyone's</em> name,"</b>
        the transferred em must lie entirely within the transferred b.
        Without this, em can land outside b, producing malformed HTML like:
            <b>something <em>surprised.</em> <b>"He mentioned</b>

        We use parent_src_id recorded during extraction to find the parent
        span in tgt_spans and clamp the child's boundaries to [parent.ws, parent.we].
        """
        if not tgt_spans:
            return tgt_spans

        n = min(len(src_spans), len(tgt_spans))

        src_span_key_to_idx: Dict[tuple, int] = {}
        for i, sp in enumerate(src_spans[:n]):
            key = (sp['word_start'], sp['word_end'], sp['tag'])
            src_span_key_to_idx[key] = i

        result = list(tgt_spans)

        for i in range(n):
            src_child = src_spans[i]
            parent_src_id = src_child.get('parent_src_id')
            if parent_src_id is None:
                continue

            best_parent_idx = None
            best_size = 10**9
            for j in range(n):
                if j == i:
                    continue
                sp = src_spans[j]
                if (sp['word_start'] <= src_child['word_start']
                        and sp['word_end'] >= src_child['word_end']):
                    size = sp['word_end'] - sp['word_start']
                    if size < best_size:
                        best_size = size
                        best_parent_idx = j

            if best_parent_idx is None:
                continue

            if best_parent_idx >= len(tgt_spans) or i >= len(tgt_spans):
                continue

            parent_tgt = tgt_spans[best_parent_idx]
            child_tgt = result[i]

            new_ws = max(child_tgt['word_start'], parent_tgt['word_start'])
            new_we = min(child_tgt['word_end'],   parent_tgt['word_end'])

            if new_ws > new_we:
                new_ws = parent_tgt['word_start']
                new_we = parent_tgt['word_start']

            if new_ws != child_tgt['word_start'] or new_we != child_tgt['word_end']:
                logger.debug(
                    f"[FormatAlignment] Nesting fix '{child_tgt['tag']}': "
                    f"[{child_tgt['word_start']}-{child_tgt['word_end']}] → "
                    f"[{new_ws}-{new_we}] (parent '{parent_tgt['tag']}' "
                    f"[{parent_tgt['word_start']}-{parent_tgt['word_end']}])"
                )
                result[i] = dict(child_tgt)
                result[i]['word_start'] = new_ws
                result[i]['word_end'] = new_we

        result_sorted = sorted(enumerate(result), key=lambda x: x[1]['word_start'])
        for ii in range(len(result_sorted)):
            for jj in range(ii + 1, len(result_sorted)):
                idx_a, span_a = result_sorted[ii]
                idx_b, span_b = result_sorted[jj]
                src_a = src_spans[idx_a] if idx_a < len(src_spans) else None
                src_b = src_spans[idx_b] if idx_b < len(src_spans) else None
                if src_a is None or src_b is None:
                    continue
                a_contains_b = (src_a['word_start'] <= src_b['word_start']
                                and src_a['word_end'] >= src_b['word_end'])
                b_contains_a = (src_b['word_start'] <= src_a['word_start']
                                and src_b['word_end'] >= src_a['word_end'])
                if a_contains_b or b_contains_a:
                    continue
                cur_a = result[idx_a]
                cur_b = result[idx_b]
                if cur_a['word_end'] >= cur_b['word_start']:
                    new_end = cur_b['word_start'] - 1
                    if new_end >= cur_a['word_start']:
                        logger.debug(
                            f"[FormatAlignment] Overlap fix '{cur_a['tag']}': "
                            f"word_end {cur_a['word_end']} → {new_end}"
                        )
                        result[idx_a] = dict(result[idx_a])
                        result[idx_a]['word_end'] = new_end

        return result

    def _extract_inline_spans(self, element: etree._Element) -> List[Dict]:
        word_tag_pairs: List[Tuple[str, List[Dict]]] = []
        self._walk_collect_words(element, parent_tags=[], output=word_tag_pairs)

        if not word_tag_pairs:
            return []

        spans: List[Dict] = []
        seen_tag_ids: set = set()

        for i, (word, tags) in enumerate(word_tag_pairs):
            for depth, tag_info in enumerate(tags):
                tid = id(tag_info)
                if tid in seen_tag_ids:
                    continue

                seen_tag_ids.add(tid)

                word_end = i
                for j in range(i, len(word_tag_pairs)):
                    _, jtags = word_tag_pairs[j]
                    if any(t is tag_info for t in jtags):
                        word_end = j

                parent_id = id(tags[depth - 1]) if depth > 0 else None

                spans.append({
                    'tag': tag_info['tag'],
                    'attrs': tag_info['attrs'],
                    'word_start': i,
                    'word_end': word_end,
                    'parent_src_id': parent_id,
                })

        return spans

    def _walk_collect_words(
        self,
        element: etree._Element,
        parent_tags: List[Dict],
        output: List[Tuple[str, List[Dict]]],
    ) -> None:
        if callable(element.tag):
            return

        tag_name = etree.QName(element).localname

        current_tags = list(parent_tags)
        if tag_name.lower() in self._transfer_tags:
            attrs = {
                k: v for k, v in element.attrib.items()
                if not k.startswith('{')
            }
            tag_info = {'tag': tag_name, 'attrs': attrs}
            current_tags = parent_tags + [tag_info]

        if element.text:
            for word in element.text.split():
                if word and _RE_CONTENT_WORD.search(word):
                    output.append((word, current_tags))

        for child in element:
            if callable(child.tag):
                continue
            child_tag = etree.QName(child).localname

            if child_tag.lower() in self._reserve_tags:
                if child.tail:
                    for word in child.tail.split():
                        if word and _RE_CONTENT_WORD.search(word):
                            output.append((word, list(current_tags)))
                continue

            if (child_tag.lower() == 'span'
                    and 'first-letter' in (child.get('class', '') or '').split()):
                child_text = child.text or ''
                child_tail = child.tail or ''
                tail_words = child_tail.split()

                attrs = {k: v for k, v in child.attrib.items() if not k.startswith('{')}
                tag_info = {'tag': child_tag, 'attrs': attrs}
                span_tags = current_tags + [tag_info]

                if child_text and tail_words:
                    merged_word = child_text + tail_words[0]
                    output.append((merged_word, span_tags))
                    for word in tail_words[1:]:
                        if _RE_CONTENT_WORD.search(word):
                            output.append((word, list(parent_tags)))
                elif child_text:
                    output.append((child_text, span_tags))
                else:
                    for word in tail_words:
                        if _RE_CONTENT_WORD.search(word):
                            output.append((word, list(parent_tags)))
                continue

            self._walk_collect_words(child, current_tags, output)

            if child.tail:
                for word in child.tail.split():
                    if word and _RE_CONTENT_WORD.search(word):
                        output.append((word, list(current_tags)))

    def _transfer_spans(
        self,
        spans: List[Dict],
        src_to_tgt: Dict[int, List[int]],
        tgt_len: int,
        tgt_words: List[str],
        src_words: List[str],
    ) -> List[Dict]:
        src_len = max(1, self._last_src_len)
        result = []

        is_content: List[bool] = [bool(_RE_CONTENT_WORD.search(w)) for w in tgt_words]

        tgt_content_indices: List[int] = [i for i, c in enumerate(is_content) if c]
        tgt_content_len = max(1, len(tgt_content_indices))

        def _prop_tgt(src_pos: int) -> int:
            sent = self._last_sent_map.get(src_pos)
            if sent:
                ss, se, ts, te = sent
                sent_src_len = max(1, se - ss)
                sent_tgt_len = max(1, te - ts)
                local = src_pos - ss
                t_local = round(local / sent_src_len * sent_tgt_len)
                t_global = ts + min(t_local, sent_tgt_len)
                return _nearest_content(min(tgt_len - 1, t_global))
            ci = min(tgt_content_len - 1,
                     round(src_pos / max(1, src_len - 1) * (tgt_content_len - 1)))
            return tgt_content_indices[ci]

        def _nearest_content(idx: int) -> int:
            if is_content[idx]: return idx
            for offset in range(1, tgt_len):
                for c in (idx + offset, idx - offset):
                    if 0 <= c < tgt_len and is_content[c]:
                        return c
            return idx

        def _trim_punct_boundaries(ts: int, te: int) -> Optional[Tuple[int, int]]:
            while ts <= te and not is_content[ts]: ts += 1
            while te >= ts and not is_content[te]: te -= 1
            return (ts, te) if ts <= te else None

        def _norm_w(w: str) -> str:
            w = w.lower()
            w = _RE_APOSTROPHES.sub('', w)
            w = _RE_PUNCT_STRIP.sub('', w)
            return w

        def _strip_trailing_punct(word: str) -> str:
            return _RE_PUNCT_STRIP.sub('', word)

        for span in spans:
            if 'first-letter' in span.get('attrs', {}).get('class', '').split():
                result.append({'tag': span['tag'], 'attrs': span['attrs'], 'word_start': 0, 'word_end': 0})
                continue

            ws = span['word_start']
            we = span['word_end']
            span_len = we - ws + 1

            if ws == 0 and span_len / src_len >= 0.9:
                result.append({'tag': span['tag'], 'attrs': span['attrs'], 'word_start': 0, 'word_end': tgt_len - 1})
                continue

            src_total = max(1, src_len - 1)
            prop_t_start = _nearest_content(_prop_tgt(ws))
            prop_t_end   = _nearest_content(_prop_tgt(we))

            tgt_indices = []
            for s in range(ws, we + 1):
                tgt_indices.extend(src_to_tgt.get(s, []))

            _neural_span_n = (max(tgt_indices) - min(tgt_indices) + 1) if tgt_indices else 0
            _neural_is_contiguous_multi = len(tgt_indices) > 1 and _neural_span_n == len(tgt_indices) and _neural_span_n >= span_len

            if not tgt_indices:
                t_start = min(tgt_len - 1, int(ws / src_total * (tgt_len - 1)))
                t_end   = min(tgt_len - 1, int(we / src_total * (tgt_len - 1)))
            else:
                t_start = max(0, min(tgt_indices))
                t_end   = min(tgt_len - 1, max(tgt_indices))

            # Floor of 2 prevents 1-position neural vs prop diffs from wiping a good contiguous span
            drift_limit = max(2, int(0.20 * tgt_len) if span_len <= 3 else int(0.35 * tgt_len))
            if abs(t_start - prop_t_start) >= drift_limit:
                t_start, t_end = prop_t_start, prop_t_end

            _has_exact_anchor = bool(
                {_norm_w(src_words[i]) for i in range(ws, we + 1) if len(_norm_w(src_words[i])) >= 2} &
                {_norm_w(tgt_words[i]) for i in range(t_start, t_end + 1) if len(_norm_w(tgt_words[i])) >= 2}
            )

            _QUOTE_CHARS = {'"', '\u201c', '\u201e', '\u00ab', '\u2039'}
            if src_words[ws][0] in _QUOTE_CHARS:
                _search_lo = max(0, t_start - 2)
                _search_hi = min(tgt_len - 1, t_start + 2)
                _quote_anchor: Optional[int] = None
                for _qi in range(t_start, _search_hi + 1):
                    if tgt_words[_qi][0] in _QUOTE_CHARS:
                        _quote_anchor = _qi
                        break
                if _quote_anchor is None:
                    for _qi in range(t_start - 1, _search_lo - 1, -1):
                        if _qi >= 0 and tgt_words[_qi][0] in _QUOTE_CHARS:
                            _quote_anchor = _qi
                            break
                if _quote_anchor is not None and _quote_anchor != t_start:
                    logger.debug(
                        f"[FormatAlignment] QuoteAnchor '{span['tag']}': "
                        f"t_start {t_start}→{_quote_anchor} "
                        f"(src='{src_words[ws][:8]}', tgt='{tgt_words[_quote_anchor][:8]}')"
                    )
                    shift = _quote_anchor - t_start
                    t_start = _quote_anchor
                    t_end   = min(tgt_len - 1, t_end + shift)
                    self._last_alignment_corrections += 1
                    self._last_alignment_corrections_detail.append(
                        f"QuoteAnchor(shift={shift})@{span['tag']}[{ws}-{we}]"
                    )

            current_n = t_end - t_start + 1
            expected_n = max(1, round(span_len * tgt_len / max(1, src_len)))

            # Left-expand when neural under-covers a multi-word span
            if span_len >= 2 and current_n < span_len and t_start > 0:
                deficit = span_len - current_n
                first_src_mapped = bool(src_to_tgt.get(ws))
                prop_suggests_left = prop_t_start < t_start
                if (not first_src_mapped) or prop_suggests_left or deficit <= 2:
                    expand = min(deficit, t_start)
                    new_start = t_start
                    for k in range(1, expand + 1):
                        cand = t_start - k
                        if cand < 0 or not is_content[cand]:
                            break
                        new_start = cand
                    if new_start < t_start:
                        logger.debug(
                            f"[FormatAlignment] LeftExpand '{span['tag']}': "
                            f"[{t_start}-{t_end}]→[{new_start}-{t_end}] "
                            f"(deficit={deficit}, first_mapped={first_src_mapped})"
                        )
                        t_start = new_start
                        current_n = t_end - t_start + 1
                        self._last_alignment_corrections += 1
                        self._last_alignment_corrections_detail.append(
                            f"LeftExpand(def={deficit})@{span['tag']}[{ws}-{we}]"
                        )

            # Span-level mean-embedding re-anchor (kept; neutral in current benchmark)
            if (span_len >= 2
                    and span.get('parent_src_id') is None
                    and self._full_src_embs is not None
                    and self._full_tgt_embs is not None):
                src_mean = F.normalize(
                    self._full_src_embs[ws:we + 1].mean(dim=0, keepdim=True),
                    p=2, dim=-1,
                )
                window = max(span_len, min(span_len + 1, expected_n))
                search_lo = max(0, prop_t_start - 4)
                search_hi = min(tgt_len - window, prop_t_start + 6)
                best_sim = -9.0
                best_ts = t_start
                for _ts in range(search_lo, search_hi + 1):
                    _te = _ts + window - 1
                    if _te >= tgt_len:
                        break
                    tgt_mean = F.normalize(
                        self._full_tgt_embs[_ts:_te + 1].mean(dim=0, keepdim=True),
                        p=2, dim=-1,
                    )
                    _sim = torch.mm(src_mean, tgt_mean.t()).item()
                    if _sim > best_sim:
                        best_sim = _sim
                        best_ts = _ts
                cur_te = min(tgt_len - 1, t_start + window - 1)
                cur_mean = F.normalize(
                    self._full_tgt_embs[t_start:cur_te + 1].mean(dim=0, keepdim=True),
                    p=2, dim=-1,
                )
                cur_sim = torch.mm(src_mean, cur_mean.t()).item()
                if best_sim > cur_sim + 0.06 and best_ts != t_start:
                    logger.debug(
                        f"[FormatAlignment] SpanMeanReanchor '{span['tag']}': "
                        f"[{t_start}-{t_end}]→[{best_ts}-{best_ts + window - 1}] "
                        f"(sim={best_sim:.3f} vs {cur_sim:.3f})"
                    )
                    t_start = best_ts
                    t_end = min(tgt_len - 1, best_ts + window - 1)
                    current_n = t_end - t_start + 1
                    self._last_alignment_corrections += 1
                    self._last_alignment_corrections_detail.append(
                        f"SpanMeanReanchor(sim={best_sim:.3f})@{span['tag']}[{ws}-{we}]"
                    )

            if _neural_is_contiguous_multi or current_n >= span_len - 1:
                if span_len <= 3:
                    max_allowed = span_len

                    _exact_anchor: Optional[int] = None
                    for _si in range(ws, we + 1):
                        _ea_norm = _norm_w(src_words[_si])
                        if len(_ea_norm) < 3:
                            continue
                        _ea_window = max(4, span_len + 2)
                        _ea_lo = max(0, prop_t_start - _ea_window)
                        _ea_hi = min(tgt_len, prop_t_start + _ea_window)
                        for _ti in range(_ea_lo, _ea_hi):
                            if _norm_w(tgt_words[_ti]) == _ea_norm:
                                _exact_anchor = _ti
                                break
                        if _exact_anchor is not None:
                            break

                    if _exact_anchor is not None:
                        t_start = _exact_anchor

                    t_end = min(tgt_len - 1, t_start + span_len - 1)

                    if (span_len == 1
                            and _exact_anchor is None
                            and span.get('parent_src_id') is None
                            and self._full_src_embs is not None
                            and self._full_tgt_embs is not None):
                        _search_start = max(0, prop_t_start - 2)
                        _search_end   = min(tgt_len - 1, prop_t_start + 4)
                        _src_vec = F.normalize(
                            self._full_src_embs[ws].unsqueeze(0), p=2, dim=-1
                        )
                        _best_sim = -9.0
                        _best_ti  = t_start
                        for _ti in range(_search_start, _search_end + 1):
                            _tgt_vec = F.normalize(
                                self._full_tgt_embs[_ti].unsqueeze(0), p=2, dim=-1
                            )
                            _sim_val = torch.mm(_src_vec, _tgt_vec.t()).item()
                            if _sim_val > _best_sim:
                                _best_sim = _sim_val
                                _best_ti  = _ti
                        if _best_ti != t_start:
                            logger.debug(
                                f"[FormatAlignment] DirectSim1 '{span['tag']}': "
                                f"t_start {t_start}→{_best_ti} "
                                f"(sim={_best_sim:.3f}, window=[{_search_start}-{_search_end}])"
                            )
                            t_start = _best_ti
                            t_end   = _best_ti
                            self._last_alignment_corrections += 1
                            self._last_alignment_corrections_detail.append(
                                f"DirectSim1(sim={_best_sim:.3f})@{span['tag']}[{ws}-{we}]"
                            )

                        if (t_end < tgt_len - 1
                                and len(_norm_w(tgt_words[t_start])) <= 2):
                            logger.debug(
                                f"[FormatAlignment] FuncWordShift1 '{span['tag']}': "
                                f"tgt[{t_start}]='{tgt_words[t_start]}' (len≤2) → +1"
                            )
                            t_start += 1
                            t_end   += 1
                            self._last_alignment_corrections += 1
                            self._last_alignment_corrections_detail.append(
                                f"FuncWordShift1@{span['tag']}[{ws}-{we}]"
                            )

                    _SW2_MIN_GAIN = 0.08
                    if (span_len == 2
                            and _exact_anchor is None
                            and span.get('parent_src_id') is None
                            and self._full_src_embs is not None
                            and self._full_tgt_embs is not None):
                        _src_first = F.normalize(
                            self._full_src_embs[ws].unsqueeze(0), p=2, dim=-1
                        )
                        _src_last = F.normalize(
                            self._full_src_embs[we].unsqueeze(0), p=2, dim=-1
                        )
                        _delta0_sum = None
                        _best_sum   = -99.0
                        _best_ts    = t_start
                        _best_te    = t_end
                        for _delta in (-1, 0, 1):
                            _ts = t_start + _delta
                            _te = t_end   + _delta
                            if _ts < 0 or _te >= tgt_len:
                                continue
                            _v0 = F.normalize(
                                self._full_tgt_embs[_ts].unsqueeze(0), p=2, dim=-1
                            )
                            _v1 = F.normalize(
                                self._full_tgt_embs[_te].unsqueeze(0), p=2, dim=-1
                            )
                            _s = (torch.mm(_src_first, _v0.t())
                                  + torch.mm(_src_last,  _v1.t())).item()
                            if _delta == 0:
                                _delta0_sum = _s
                            if _s > _best_sum:
                                _best_sum = _s
                                _best_ts  = _ts
                                _best_te  = _te

                        _gain = (_best_sum - _delta0_sum) if _delta0_sum is not None else 0.0
                        if (_best_ts == t_start and _best_te == t_end) or _gain < _SW2_MIN_GAIN:
                            def _bigram_j(a: str, b: str) -> float:
                                if len(a) < 2 or len(b) < 2:
                                    return 1.0 if a == b else 0.0
                                a_bg = {a[i:i+2] for i in range(len(a) - 1)}
                                b_bg = {b[i:i+2] for i in range(len(b) - 1)}
                                union = len(a_bg | b_bg)
                                return len(a_bg & b_bg) / union if union else 0.0

                            _src_last_w    = _norm_w(src_words[we])
                            _delta0_ss     = 0.0
                            _best_ss_sum   = -1.0
                            _best_ss_ts    = t_start
                            _best_ss_te    = t_end
                            for _delta in (-1, 0, 1):
                                _ts = t_start + _delta
                                _te = t_end   + _delta
                                if _ts < 0 or _te >= tgt_len:
                                    continue
                                _ss = _bigram_j(_src_last_w, _norm_w(tgt_words[_te]))
                                if _delta == 0:
                                    _delta0_ss = _ss
                                if _ss > _best_ss_sum:
                                    _best_ss_sum = _ss
                                    _best_ss_ts  = _ts
                                    _best_ss_te  = _te
                            if (_best_ss_sum > 0.0
                                    and _best_ss_sum > _delta0_ss
                                    and (_best_ss_ts != t_start or _best_ss_te != t_end)):
                                logger.debug(
                                    f"[FormatAlignment] SlidingWin2 '{span['tag']}' "
                                    f"(str-sim={_best_ss_sum:.2f}): "
                                    f"[{t_start}-{t_end}]→[{_best_ss_ts}-{_best_ss_te}]"
                                )
                                t_start = _best_ss_ts
                                t_end   = _best_ss_te
                                self._last_alignment_corrections += 2
                                self._last_alignment_corrections_detail.append(
                                    f"SlidingWin2(strsim={_best_ss_sum:.2f})@{span['tag']}[{ws}-{we}]"
                                )
                        else:
                            logger.debug(
                                f"[FormatAlignment] SlidingWin2 '{span['tag']}': "
                                f"[{t_start}-{t_end}]→[{_best_ts}-{_best_te}] "
                                f"(sum_sim={_best_sum:.3f}, gain={_gain:.3f})"
                            )
                            t_start = _best_ts
                            t_end   = _best_te
                            self._last_alignment_corrections += 1
                            self._last_alignment_corrections_detail.append(
                                f"SlidingWin2(cosine,gain={_gain:.2f})@{span['tag']}[{ws}-{we}]"
                            )

                    # Only shift when the LEADING target word is a short function word.
                    # Previous "any word len<=2" caused non-converging rightward drift
                    # on Polish "w" inside otherwise good spans (n2).
                    if _exact_anchor is None and span_len >= 2:
                        for _cs in range(3):
                            if t_start >= tgt_len:
                                break
                            if len(_norm_w(tgt_words[t_start])) > 2:
                                break
                            if t_end + 1 >= tgt_len:
                                break
                            t_start += 1
                            t_end += 1

                else:
                    if current_n < span_len and t_start > 0:
                        deficit = span_len - current_n
                        new_start = max(0, t_start - deficit)
                        while new_start < t_start and not is_content[new_start]:
                            new_start += 1
                        if new_start < t_start:
                            logger.debug(
                                f"[FormatAlignment] LongLeftExpand '{span['tag']}': "
                                f"[{t_start}-{t_end}]→[{new_start}-{t_end}]"
                            )
                            t_start = new_start
                            current_n = t_end - t_start + 1
                            self._last_alignment_corrections += 1
                            self._last_alignment_corrections_detail.append(
                                f"LongLeftExpand(def={deficit})@{span['tag']}[{ws}-{we}]"
                            )
                    max_allowed = span_len + 2
                    if current_n > max_allowed:
                        excess = current_n - max_allowed
                        t_start += excess
                        t_start = max(0, t_start)
                        if t_start > t_end:
                            t_start = t_end

                logger.debug(f"[FormatAlignment] RE-ANCHOR + SHRINK '{span['tag']}' [{ws}-{we}] → [{t_start}-{t_end}]")
            else:
                extra = expected_n - current_n
                half = (extra + 1) // 2
                left_floor = prop_t_start if len(_norm_w(tgt_words[prop_t_start])) >= 3 else t_start
                t_start = max(t_start - half, left_floor)
                t_end = min(tgt_len - 1, t_start + expected_n - 1)

            if span_len >= 2 and t_end > prop_t_end:
                t_end = prop_t_end
                if t_start > prop_t_start and (t_start - prop_t_start) > span_len // 2:
                    t_start = prop_t_start
                min_end = min(tgt_len - 1, t_start + max(1, span_len - 2))
                if t_end < min_end:
                    t_end = min_end
                if t_start > t_end:
                    t_start = t_end

            trimmed = _trim_punct_boundaries(t_start, t_end)
            if trimmed is None:
                continue
            t_start, t_end = trimmed

            if t_end > t_start:
                last_w = tgt_words[t_end]
                if not _RE_CONTENT_WORD.search(_strip_trailing_punct(last_w)):
                    t_end -= 1

            result.append({
                'tag':        span['tag'],
                'attrs':      span['attrs'],
                'word_start': t_start,
                'word_end':   t_end,
            })

        return result

    def _build_result_html(
        self,
        orig_elem: etree._Element,
        tgt_words: List[str],
        tgt_spans: List[Dict],
    ) -> str:
        word_spans: List[List[Dict]] = [[] for _ in range(len(tgt_words))]
        for span in tgt_spans:
            for wi in range(span['word_start'], min(span['word_end'] + 1, len(tgt_words))):
                word_spans[wi].append(span)

        inner_parts: List[str] = []
        active: List[Dict] = []

        for wi, word in enumerate(tgt_words):
            wanted = word_spans[wi]
            wanted_ids = {id(s) for s in wanted}

            to_close_ids = {id(s) for s in active if id(s) not in wanted_ids}
            if to_close_ids:
                first_close_pos = next(
                    (pos for pos, s in enumerate(active) if id(s) in to_close_ids),
                    len(active),
                )
                to_reopen: List[Dict] = []
                for s in reversed(active[first_close_pos:]):
                    inner_parts.append(f'</{s["tag"]}>')
                    if id(s) not in to_close_ids:
                        to_reopen.insert(0, s)
                active = active[:first_close_pos]
                for s in to_reopen:
                    attrs_str = _build_attrs_str(s['attrs'])
                    inner_parts.append(f'<{s["tag"]}{attrs_str}>')
                    active.append(s)

            if wi > 0:
                inner_parts.append(' ')

            active_ids = {id(s) for s in active}
            to_open = [s for s in wanted if id(s) not in active_ids]
            to_open_sorted = sorted(
                to_open,
                key=lambda s: (-(s['word_end'] - s['word_start']), s['word_start']),
            )
            for s in to_open_sorted:
                attrs_str = _build_attrs_str(s['attrs'])
                inner_parts.append(f'<{s["tag"]}{attrs_str}>')
                active.append(s)

            inner_parts.append(_escape_xml(word))

        for s in reversed(active):
            inner_parts.append(f'</{s["tag"]}>')

        inner_html = ''.join(inner_parts)

        outer_tag = etree.QName(orig_elem).localname
        outer_attrs = _build_attrs_str({
            k: v for k, v in orig_elem.attrib.items()
            if not k.startswith('{')
        })

        full_html = (
            f'<{outer_tag} xmlns="{XHTML_NS}"{outer_attrs}>'
            f'{inner_html}'
            f'</{outer_tag}>'
        )

        try:
            etree.fromstring(full_html.encode('utf-8'))
            return full_html
        except etree.XMLSyntaxError as exc:
            logger.warning(
                f"[FormatAlignment] Generated HTML is invalid: {exc}\n"
                f"Fragment: {full_html[:300]}"
            )
            return ''

    def _itermax(
        self,
        sim,
        S: int,
        T: int,
    ) -> List[Tuple[int, int]]:
        SIM_THRESHOLD = 0.05
        pairs: List[Tuple[int, int]] = []
        sim_work = sim.clone().float()
        max_pairs = min(S, T)

        for _ in range(max_pairs):
            best_val = sim_work.max().item()
            if best_val < SIM_THRESHOLD:
                break

            flat_idx = sim_work.argmax().item()
            s = int(flat_idx) // T
            t = int(flat_idx) % T

            pairs.append((s, t))
            sim_work[s, :] = -2.0
            sim_work[:, t] = -2.0

        return sorted(pairs)

    def _itermax_bidirectional(
        self,
        sim,
        S: int,
        T: int,
    ) -> List[Tuple[int, int]]:
        """Bidirectional alignment: intersect forward and backward itermax pairs.

        A pair (s, t) is considered reliable only if:
          - forward pass: s is best match for t  AND
          - backward pass: t is best match for s
        Src words with no intersection pair fall back to their forward match.
        This dramatically reduces false positives from greedy competition.
        """
        fwd_pairs = set(self._itermax(sim, S, T))
        bwd_raw   = self._itermax(sim.t().contiguous(), T, S)
        bwd_pairs = {(s, t) for (t, s) in bwd_raw}

        intersection = fwd_pairs & bwd_pairs

        covered_src = {s for (s, _) in intersection}
        for (s, t) in fwd_pairs:
            if s not in covered_src:
                intersection.add((s, t))

        return sorted(intersection)

    def _compute_alignments(
        self,
        src_words: List[str],
        tgt_words: List[str],
    ) -> List[Tuple[int, int]]:
        self._last_src_len = len(src_words)
        S = len(src_words)
        T = len(tgt_words)

        if S == 1:
            return [(0, t) for t in range(T)]
        if T == 1:
            return [(s, 0) for s in range(S)]

        def _normalize(token: str) -> str:
            t = token.lower()
            t = _RE_APOSTROPHES.sub('', t)
            t = _RE_PUNCT_STRIP.sub('', t)
            return t

        _MIN_MATCH_LEN = 3

        src_norm = [_normalize(w) for w in src_words]
        tgt_norm = [_normalize(w) for w in tgt_words]

        forced_s2t: Dict[int, int] = {}
        forced_used_tgt: set = set()

        for s_idx, s_n in enumerate(src_norm):
            if len(s_n) < _MIN_MATCH_LEN:
                continue
            for t_idx, t_n in enumerate(tgt_norm):
                if t_idx in forced_used_tgt:
                    continue
                if s_n == t_n:
                    forced_s2t[s_idx] = t_idx
                    forced_used_tgt.add(t_idx)
                    logger.debug(
                        f"[FormatAlignment] Forced: '{src_words[s_idx]}' "
                        f"(norm='{s_n}') → '{tgt_words[t_idx]}' (idx {s_idx}→{t_idx})"
                    )
                    break

        if forced_s2t:
            monotonic: Dict[int, int] = {}
            last_t = -1
            for s_a, t_a in sorted(forced_s2t.items()):
                if t_a > last_t:
                    monotonic[s_a] = t_a
                    last_t = t_a
            if len(monotonic) != len(forced_s2t):
                dropped = len(forced_s2t) - len(monotonic)
                logger.debug(
                    f"[FormatAlignment] Monotonicity fix: dropped {dropped} "
                    f"crossing forced anchor(s)"
                )
            forced_s2t = monotonic
            forced_used_tgt = set(monotonic.values())

        src_embs = self._get_word_embeddings(src_words)
        tgt_embs = self._get_word_embeddings(tgt_words)

        neural_pairs: List[Tuple[int, int]] = []

        if src_embs is not None and tgt_embs is not None:
            src_norm_emb = F.normalize(src_embs, p=2, dim=-1)
            tgt_norm_emb = F.normalize(tgt_embs, p=2, dim=-1)
            sim = torch.mm(src_norm_emb, tgt_norm_emb.t())

            for t_idx, tw in enumerate(tgt_words):
                if not _RE_CONTENT_WORD.search(tw):
                    sim[:, t_idx] = -2.0
                    logger.debug(
                        f"[FormatAlignment] Masking pure-punct tgt[{t_idx}]='{tw}'"
                    )

            if forced_s2t:
                forced_s_list = list(forced_s2t.keys())
                forced_t_list = list(forced_s2t.values())
                sim[forced_s_list, :] = -2.0
                sim[:, forced_t_list] = -2.0

            neural_pairs = self._itermax_bidirectional(sim, S, T)

        else:
            logger.warning("[FormatAlignment] No embeddings available – using proportional fallback")
            for s_idx in range(S):
                if s_idx in forced_s2t:
                    continue
                t_idx = min(T - 1, int(s_idx / max(1, S - 1) * (T - 1)))
                if t_idx not in forced_used_tgt:
                    neural_pairs.append((s_idx, t_idx))

        combined: Dict[int, int] = {}

        for s, t in neural_pairs:
            if s in forced_s2t:
                continue
            if t in forced_used_tgt:
                continue
            combined[s] = t

        combined.update(forced_s2t)

        result = sorted(combined.items())

        logger.debug(
            f"[FormatAlignment] Alignments: {len(forced_s2t)} forced + "
            f"{len(result) - len(forced_s2t)} neural = {len(result)} pairs total "
            f"(S={S}, T={T})"
        )

        return result

    def _get_word_embeddings(self, words: List[str]):
        if not words:
            return None
        if self._adapter is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        result = self._adapter.embed_words(words)
        if result is None:
            return None
        return result.vectors

    def _build_word_ids(
        self,
        words: List[str],
        tokens: List[str],
    ) -> List[Optional[int]]:
        if self._adapter is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        return self._adapter.build_word_ids(words, tokens)
        
BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        'html': '<p>The old <i>wooden door</i> creaked loudly in the wind.</p>',
        'translation': 'La vieja puerta de madera crujió con fuerza en el viento.',
        'expected': [('i', 'puerta de madera')],
    },
    {
        'html': '<p>He bought <b>three red apples</b> at the market yesterday.</p>',
        'translation': 'Er hat gestern auf dem Markt drei rote Äpfel gekauft.',
        'expected': [('b', 'drei rote Äpfel')],
    },
    {
        'html': '<p>They finally reached the <em>ancient castle</em> before sunset.</p>',
        'translation': 'Ils ont enfin atteint le vieux château avant le coucher du soleil.',
        'expected': [('em', 'vieux château')],
    },
    {
        'html': (
            '<p>The scientist described the <i>extraordinarily complicated</i> '
            'mechanism in detail.</p>'
        ),
        'translation': (
            'Der Wissenschaftler beschrieb den außerordentlich komplizierten '
            'Mechanismus ausführlich.'
        ),
        'expected': [('i', 'außerordentlich komplizierten')],
    },
    {
        'html': '<p><b>Chapter One</b> began with a <i>strange</i> dream about the sea.</p>',
        'translation': 'Capítulo Uno comenzó con un extraño sueño sobre el mar.',
        'expected': [('b', 'Capítulo Uno'), ('i', 'extraño')],
    },
]


def _extract_tagged_spans(result_html: str) -> List[Tuple[str, str]]:
    if not result_html:
        return []
    try:
        root = etree.fromstring(result_html.encode('utf-8'))
    except etree.XMLSyntaxError:
        return []
    spans: List[Tuple[str, str]] = []
    for el in root.iter():
        if el is root or callable(el.tag):
            continue
        name = etree.QName(el).localname
        if name in INLINE_TRANSFER_TAGS:
            spans.append((name, ' '.join(''.join(el.itertext()).split())))
    return spans


def run_alignment_benchmark(
    model_names: Optional[Sequence[str]] = None,
    device: str = 'cpu',
    local_models_dir: str = '',
    cases: Optional[List[Dict[str, Any]]] = None,
    word_id_sources: Sequence[str] = ('legacy', 'auto'),
) -> List[Dict[str, Any]]:
    names = list(model_names) if model_names else list(AVAILABLE_MODELS.keys())
    test_cases = cases if cases is not None else BENCHMARK_CASES
    if not test_cases:
        raise ValueError("No benchmark cases provided.")

    rows: List[Dict[str, Any]] = []

    for name in names:
        engine = FormatAlignmentEngine(
            model_name=name,
            device=device,
            local_models_dir=local_models_dir,
        )
        try:
            started = time.perf_counter()
            engine.load_model()
            load_seconds = time.perf_counter() - started
        except Exception as exc:
            rows.append({'model': name, 'error': str(exc)})
            continue

        base_spec = engine._adapter.spec
        try:
            engine.align_element(test_cases[0]['html'], test_cases[0]['translation'])

            for source in word_id_sources:
                engine._adapter.spec = replace(base_spec, word_id_source=source)
                durations: List[float] = []
                hits = 0
                total = 0
                valid = 0

                for case in test_cases:
                    started = time.perf_counter()
                    result = engine.align_element(case['html'], case['translation'])
                    durations.append(time.perf_counter() - started)

                    spans = _extract_tagged_spans(result)
                    if result and spans:
                        valid += 1
                    for item in case.get('expected', []):
                        total += 1
                        if item in spans:
                            hits += 1

                rows.append({
                    'model': name,
                    'word_id_source': source,
                    'load_s': load_seconds,
                    'avg_ms': 1000.0 * sum(durations) / len(durations),
                    'span_accuracy': (hits / total) if total else None,
                    'valid': f"{valid}/{len(test_cases)}",
                })
        finally:
            engine._adapter.spec = base_spec
            engine.unload_model()

    header = f"{'model':34} {'src':7} {'load[s]':>8} {'avg[ms]':>9} {'spans':>7} {'valid':>6}"
    print(header)
    print('-' * len(header))
    for row in rows:
        if 'error' in row:
            print(f"{row['model']:34} ERROR: {row['error']}")
            continue
        accuracy = row['span_accuracy']
        accuracy_text = f"{accuracy:.2f}" if accuracy is not None else 'n/a'
        print(
            f"{row['model']:34} {row['word_id_source']:7} {row['load_s']:8.1f} "
            f"{row['avg_ms']:9.1f} {accuracy_text:>7} {row['valid']:>6}"
        )

    return rows


if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    run_alignment_benchmark(model_names=sys.argv[1:] or None)
