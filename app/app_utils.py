# app_utils.py
import copy
import json
import logging
import os
import re
import functools as _functools
from lxml import etree
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

WRAPPER_TAG_FALLBACKS = ("translated", "translation")
WRAPPER_TAG_NAME_RE = re.compile(r'^[^\W\d_][^\s<>/&"=\x27]*$')


def is_valid_wrapper_variant_name(name: Any) -> bool:
    return isinstance(name, str) and bool(WRAPPER_TAG_NAME_RE.match(name.strip()))


def normalize_wrapper_variants(variants: Any, primary: str = "") -> List[str]:
    if not isinstance(variants, (list, tuple)):
        return []
    primary_key = (primary or "").strip().lower()
    reserved = set(WRAPPER_TAG_FALLBACKS)
    result: List[str] = []
    seen = set()
    for item in variants:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not is_valid_wrapper_variant_name(name):
            continue
        key = name.lower()
        if key == primary_key or key in reserved or key in seen:
            continue
        seen.add(key)
        result.append(name)
    return result


def wrapper_tag_names(primary: str, variants: Any = None) -> List[str]:
    names: List[str] = []
    seen = set()
    candidates = [(primary or "").strip() or "translated"]
    candidates.extend(normalize_wrapper_variants(variants, primary))
    candidates.extend(WRAPPER_TAG_FALLBACKS)
    for name in candidates:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def build_wrapper_tags_pattern(primary: str, variants: Any = None) -> str:
    return "(?:" + "|".join(re.escape(n) for n in wrapper_tag_names(primary, variants)) + ")"

class SessionManager:
    @staticmethod
    def save_session(
        path: str,
        paragraphs: List[Dict],
        original_file_path: str,
        file_type: str,
        app_settings: Dict,
        context_before: int,
        context_after: int,
        temperature: float,
        custom_prompts: Dict[str, str],
        single_prompt_mode: bool,
        processing_mode: str,
        prompt_variant: str,
        json_payload_mode: bool = False,
        json_payload_template: str = "",
        json_response_field: str = "",
        sentence_batch_enabled: bool = False,
        sentence_batch_size: int = 5,
        auto_fix_enabled: bool = False,
        auto_fix_attempts: int = 3,
        top_p: float = 0.0,
        top_k: int = 0,
        timeout_minutes: int = 10,
        enable_thinking: bool = True,
        source_lang: str = "",
        target_lang: str = "",
        quick_translate_service: str = "DeepL Free",
        source_format: str = "",
        tag_profile_dict: Optional[Dict] = None,
        response_wrapper_tag: str = "",
        batch_json_payload_template: str = "",
        json_payload_batch: bool = False,
    ) -> None:
        _NON_SERIALISABLE = frozenset({
            'auto_fix_history',
            '_had_newlines',
            '_original_parts',
            '_original_separators',
            '_used_ps_marker',
            'use_ps_markers',
            'index',
            'llm_prompt',
        })

        paragraphs_to_save = []
        for para in paragraphs:
            para_copy = {}
            for k, v in para.items():
                if k in _NON_SERIALISABLE:
                    continue
                try:
                    from lxml import etree as _etree
                    if isinstance(v, _etree._Element):
                        continue
                except Exception:
                    pass
                para_copy[k] = v

            para_copy['processing_mode'] = processing_mode
            paragraphs_to_save.append(para_copy)

        skip_inline_tags = app_settings.get('skip_inline_tags', {})

        effective_tag_profile = tag_profile_dict
        if not isinstance(effective_tag_profile, dict) or not effective_tag_profile:
            effective_tag_profile = app_settings.get('tag_profile', {}) or {}

        effective_wrapper = (response_wrapper_tag or '').strip()
        if not effective_wrapper:
            effective_wrapper = app_settings.get('response_wrapper_tag', 'translated') or 'translated'

        effective_variants = normalize_wrapper_variants(
            app_settings.get('response_wrapper_tag_variants'),
            effective_wrapper
        )

        session_data = {
            'session_version': 2,
            'original_file_path': original_file_path,
            'file_type': file_type,
            'source_format': source_format or file_type,
            'paragraphs': paragraphs_to_save,
            'context_before': context_before,
            'context_after': context_after,
            'temperature': temperature,
            'custom_ollama_prompt': custom_prompts.get('ollama'),
            'custom_system_prompt': custom_prompts.get('system'),
            'custom_assistant_prompt': custom_prompts.get('assistant'),
            'custom_batch_assistant_prompt': custom_prompts.get('batch_assistant'),
            'custom_user_prompt': custom_prompts.get('user'),
            'custom_json_payload_prompt': custom_prompts.get('json_payload'),
            'custom_batch_json_payload_prompt': custom_prompts.get('batch_json_payload'),
            'custom_batch_hint_prompt': custom_prompts.get('batch_hint'),
            'single_prompt_mode': single_prompt_mode,
            'json_payload_mode': json_payload_mode,
            'json_payload_template': json_payload_template,
            'batch_json_payload_template': batch_json_payload_template,
            'json_payload_batch': json_payload_batch,
            'json_response_field': json_response_field,
            'sentence_batch_enabled': sentence_batch_enabled,
            'sentence_batch_size': sentence_batch_size,
            'auto_fix_enabled': auto_fix_enabled,
            'auto_fix_attempts': auto_fix_attempts,
            'top_p': top_p,
            'top_k': top_k,
            'timeout_minutes': timeout_minutes,
            'enable_thinking': enable_thinking,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'quick_translate_service': quick_translate_service,
            'metadata': {
                'processing_mode': processing_mode,
                'prompt_variant': prompt_variant,
                'skip_inline_tags': skip_inline_tags,
                'tag_profile': effective_tag_profile,
                'response_wrapper_tag': effective_wrapper,
                'response_wrapper_tag_variants': effective_variants,
            }
        }

        def _json_default(obj):
            try:
                from lxml import etree as _et
                if isinstance(obj, _et._Element):
                    return _et.tostring(obj, encoding='unicode')
            except Exception:
                pass
            return str(obj)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=4, default=_json_default)
            logger.info(f"Session saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            raise

    @staticmethod
    def load_session(session_path: str) -> Dict:
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            required_fields = ['original_file_path', 'file_type', 'paragraphs']
            for field in required_fields:
                if field not in session_data:
                    raise KeyError(f"Missing required field: {field}")

            metadata = session_data.get('metadata', {})
            processing_mode = metadata.get('processing_mode', 'inline')
            prompt_variant = metadata.get('prompt_variant', None)

            skip_inline_tags = metadata.get('skip_inline_tags')

            if skip_inline_tags is None and 'skip_spans' in metadata:
                old_skip_spans = metadata.get('skip_spans', False)
                skip_inline_tags = {'span': old_skip_spans} if old_skip_spans else {}
                logger.info(f"Converted old skip_spans={old_skip_spans} to skip_inline_tags={skip_inline_tags}")
            elif skip_inline_tags is None:
                skip_inline_tags = {}

            tag_profile = metadata.get('tag_profile')
            if not isinstance(tag_profile, dict) or not tag_profile:
                tag_profile = None

            response_wrapper_tag = metadata.get('response_wrapper_tag')
            if not isinstance(response_wrapper_tag, str) or not response_wrapper_tag.strip():
                response_wrapper_tag = None

            raw_wrapper_variants = metadata.get('response_wrapper_tag_variants')
            if isinstance(raw_wrapper_variants, list):
                response_wrapper_tag_variants = normalize_wrapper_variants(
                    raw_wrapper_variants,
                    response_wrapper_tag or ''
                )
            else:
                response_wrapper_tag_variants = None

            custom_prompts = {
                'ollama': session_data.get('custom_ollama_prompt'),
                'system': session_data.get('custom_system_prompt'),
                'assistant': session_data.get('custom_assistant_prompt'),
                'batch_assistant': session_data.get('custom_batch_assistant_prompt'),
                'user': session_data.get('custom_user_prompt'),
                'json_payload': session_data.get('custom_json_payload_prompt'),
                'batch_json_payload': session_data.get('custom_batch_json_payload_prompt'),
                'batch_hint': session_data.get('custom_batch_hint_prompt'),
            }

            if 'context_size' in session_data and 'context_before' not in session_data:
                context_before = session_data['context_size']
                context_after = 0
            else:
                context_before = session_data.get('context_before', 3)
                context_after = session_data.get('context_after', 2)

            paragraphs = session_data['paragraphs']
            _TRANSIENT_FIELDS = (
                '_had_newlines',
                '_original_parts',
                '_original_separators',
                '_used_ps_marker',
                'use_ps_markers',
                'index',
                'llm_prompt',
            )
            for para in paragraphs:
                for field in _TRANSIENT_FIELDS:
                    para.pop(field, None)

            result = {
                'session_version': session_data.get('session_version', 1),
                'original_file_path': session_data['original_file_path'],
                'file_type': session_data['file_type'],
                'source_format': session_data.get('source_format', ''),
                'paragraphs': paragraphs,
                'context_before': context_before,
                'context_after': context_after,
                'temperature': session_data.get('temperature', 0.8),
                'custom_prompts': custom_prompts,
                'single_prompt_mode': session_data.get('single_prompt_mode', False),
                'json_payload_mode': session_data.get('json_payload_mode', False),
                'json_payload_template': session_data.get('json_payload_template', ''),
                'batch_json_payload_template': session_data.get('batch_json_payload_template', ''),
                'json_payload_batch': session_data.get('json_payload_batch', False),
                'json_response_field': session_data.get('json_response_field', ''),
                'sentence_batch_enabled': session_data.get('sentence_batch_enabled', False),
                'sentence_batch_size': session_data.get('sentence_batch_size', 5),
                'auto_fix_enabled': session_data.get('auto_fix_enabled', False),
                'auto_fix_attempts': session_data.get('auto_fix_attempts', 3),
                'top_p': session_data.get('top_p', 0.0),
                'top_k': session_data.get('top_k', 0),
                'timeout_minutes': session_data.get('timeout_minutes', 10),
                'enable_thinking': session_data.get('enable_thinking', True),
                'source_lang': session_data.get('source_lang', ''),
                'target_lang': session_data.get('target_lang', ''),
                'quick_translate_service': session_data.get('quick_translate_service', 'DeepL Free'),
                'metadata': {
                    'processing_mode': processing_mode,
                    'prompt_variant': prompt_variant,
                    'skip_inline_tags': skip_inline_tags,
                    'tag_profile': tag_profile,
                    'response_wrapper_tag': response_wrapper_tag,
                    'response_wrapper_tag_variants': response_wrapper_tag_variants,
                }
            }

            _INT_KEY_FIELDS = ('inline_formatting_map', 'non_translatable_placeholders')
            for para in result['paragraphs']:
                for field in _INT_KEY_FIELDS:
                    raw = para.get(field)
                    if isinstance(raw, dict) and raw:
                        converted = {}
                        for k, v in raw.items():
                            try:
                                converted[int(k)] = v
                            except (ValueError, TypeError):
                                converted[k] = v
                        para[field] = converted

            logger.info(f"Session loaded from: {session_path}")
            logger.info(f"Mode: {processing_mode}, Variant: {prompt_variant}, skip_inline_tags: {skip_inline_tags}")

            return result

        except FileNotFoundError:
            logger.error(f"Session file not found: {session_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in session file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            raise

    @staticmethod
    def restore_epub_ids(
        elements_tree: Any,
        item_href: str,
        session_map: Dict[Tuple[str, str], Dict],
        namespaces: Dict[str, str]
    ) -> None:
        for element in elements_tree:
            if not isinstance(element, etree._Element):
                continue

            text = SessionManager._get_element_text_lxml(element)
            if not text or not text.strip():
                continue

            key = (item_href, text)
            if key in session_map:
                para = session_map[key]
                element_id = para.get('id')
                if element_id is not None:
                    element.set('id', element_id)
                    logger.debug(f"Restored ID: {element_id} for text: {text[:50]}...")

            SessionManager.restore_epub_ids(element, item_href, session_map, namespaces)

    @staticmethod
    def _get_element_text_lxml(element: Any) -> str:
        text_content = etree.tostring(
            element,
            encoding='unicode',
            method='text'
        )

        return ' '.join(text_content.split())

class TagProfile:
    def __init__(
        self,
        reserve_prefix: str = "id_",
        inline_prefix: str = "p_",
        non_trans_prefix: str = "nt_",
        paragraph_separator: str = "ps",
        batch_separator: str = "z",
        core_text_var: str = "core_text",
        context_before_var: str = "context_before",
        context_after_var: str = "context_after",
    ):
        self.reserve_prefix = reserve_prefix
        self.inline_prefix = inline_prefix
        self.non_trans_prefix = non_trans_prefix
        self.paragraph_separator = paragraph_separator
        self.batch_separator = batch_separator
        self.core_text_var = core_text_var
        self.context_before_var = context_before_var
        self.context_after_var = context_after_var

    @staticmethod
    def from_settings(settings_dict: Optional[Dict]) -> "TagProfile":
        defaults = TagProfile()
        raw = settings_dict.get("tag_profile") if isinstance(settings_dict, dict) else None
        if not isinstance(raw, dict):
            raw = {}

        def pick(key: str, default: str) -> str:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            return default

        return TagProfile(
            reserve_prefix=pick("reserve_prefix", defaults.reserve_prefix),
            inline_prefix=pick("inline_prefix", defaults.inline_prefix),
            non_trans_prefix=pick("non_trans_prefix", defaults.non_trans_prefix),
            paragraph_separator=pick("paragraph_separator", defaults.paragraph_separator),
            batch_separator=pick("batch_separator", defaults.batch_separator),
            core_text_var=pick("core_text_var", defaults.core_text_var),
            context_before_var=pick("context_before_var", defaults.context_before_var),
            context_after_var=pick("context_after_var", defaults.context_after_var),
        )

    def reserve_open_re(self) -> re.Pattern:
        return re.compile(rf'<{re.escape(self.reserve_prefix)}(\d{{2}})>')

    def reserve_close_re(self) -> re.Pattern:
        return re.compile(rf'</{re.escape(self.reserve_prefix)}(\d{{2}})>')

    def inline_open_re(self) -> re.Pattern:
        return re.compile(rf'<{re.escape(self.inline_prefix)}(\d{{2}})>')

    def inline_close_re(self) -> re.Pattern:
        return re.compile(rf'</{re.escape(self.inline_prefix)}(\d{{2}})>')

    def non_trans_open_re(self) -> re.Pattern:
        return re.compile(rf'<{re.escape(self.non_trans_prefix)}(\d{{2}})/>')

    def paragraph_sep_re(self) -> re.Pattern:
        return re.compile(rf'<{re.escape(self.paragraph_separator)}>')

    def batch_sep_re(self) -> re.Pattern:
        return re.compile(rf'<{re.escape(self.batch_separator)}>')

    def all_placeholders_re(self) -> re.Pattern:
        reserve = re.escape(self.reserve_prefix)
        inline = re.escape(self.inline_prefix)
        non_trans = re.escape(self.non_trans_prefix)
        paragraph = re.escape(self.paragraph_separator)
        batch = re.escape(self.batch_separator)
        return re.compile(
            rf'</?{reserve}\d{{2}}>'
            rf'|</?{inline}\d{{2}}>'
            rf'|<{non_trans}\d{{2}}/>'
            rf'|</?{paragraph}>'
            rf'|<{batch}>'
        )

    def reserve_placeholder(self, n: int) -> str:
        return f'<{self.reserve_prefix}{n:02d}>'

    def inline_open_placeholder(self, n: int) -> str:
        return f'<{self.inline_prefix}{n:02d}>'

    def inline_close_placeholder(self, n: int) -> str:
        return f'</{self.inline_prefix}{n:02d}>'

    def non_trans_placeholder(self, n: int) -> str:
        return f'<{self.non_trans_prefix}{n:02d}/>'

    def fill_template(
        self,
        template: str,
        core_text: str,
        context_before: str,
        context_after: str,
    ) -> str:
        mapping = {
            self.core_text_var: core_text,
            self.context_before_var: context_before,
            self.context_after_var: context_after,
        }
        pattern = re.compile(
            r'\{(' + '|'.join(re.escape(name) for name in mapping) + r')\}'
        )
        return pattern.sub(lambda m: mapping[m.group(1)], template)

class AppSettingsManager:
    SETTINGS_FILE = "app_settings.json"

    DEFAULT_SETTINGS = {
        "llm_choice": "LM Studio",
        "server_url": "http://localhost:1234/v1/chat/completions",
        "json_response_field": "",
        "ollama_model_name": "",
        "ollama_server_url": "http://localhost:11434",
        "openrouter_api_key": "",
        "openrouter_model_name": "",
        "ollama_endpoint": "http://localhost:11434",
        "deepl_free_api_key": "",
        "deepl_pro_api_key": "",
        "use_inline_formatting": True,
        "restore_paragraph_structure": True,
        "show_ps_in_ui": False,
        "quote_style": "neutral",
        "skip_inline_tags": {},
        "auto_save_interval": 10,
        "enable_thinking": False,
        "response_wrapper_tag": "translated",
        "response_wrapper_tag_variants": [],
        "prompt_tag_profile": None,
        "mismatch_checks": {
            "paragraphs": True,
            "first_char": True,
            "last_char": True,
            "length": True,
            "quote_parity": True,
            "untranslated": True,
            "reserve_elements": True,
            "nt_markers": True,
            "inline_formatting": True,
        },
        "mismatch_thresholds": {
            "length_ratio_short": 1.6,
            "length_ratio_medium": 1.4,
            "length_ratio_long": 1.3,
            "untranslated_ratio": 0.3,
            "position_shift_threshold": 0.15,
            "inline_position_shift_threshold": 0.30,
        },
        "alignment_settings": {
            "device": "cpu",
            "model_name": "microsoft/xlm-align-base",
        },
        "tag_profile": {
            "reserve_prefix": "id_",
            "inline_prefix": "p_",
            "non_trans_prefix": "nt_",
            "paragraph_separator": "ps",
            "batch_separator": "z",
            "core_text_var": "core_text",
            "context_before_var": "context_before",
            "context_after_var": "context_after",
        },
    }

    @staticmethod
    def load_settings() -> Dict:
        try:
            with open(AppSettingsManager.SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            logger.info("App settings loaded successfully")
        except FileNotFoundError:
            logger.info("App settings file not found, using defaults")
            settings = {}
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load app settings: {e}, using defaults")
            settings = {}

        for key, val in AppSettingsManager.DEFAULT_SETTINGS.items():
            settings.setdefault(key, val)

        settings['response_wrapper_tag_variants'] = normalize_wrapper_variants(
            settings.get('response_wrapper_tag_variants'),
            settings.get('response_wrapper_tag') or 'translated'
        )

        if 'use_ps_markers' in settings:
            if 'restore_paragraph_structure' not in settings:
                settings['restore_paragraph_structure'] = settings['use_ps_markers']
            del settings['use_ps_markers']
            logger.info("Migrated use_ps_markers -> restore_paragraph_structure")

        for obsolete in ('restore_paragraph_epub', 'restore_paragraph_txt'):
            if obsolete in settings:
                del settings[obsolete]
                logger.info(f"Removed obsolete key: {obsolete}")

        _aliases = {'swiss': 'french', 'russian': 'french'}
        if settings.get('quote_style') in _aliases:
            settings['quote_style'] = _aliases[settings['quote_style']]

        for nested_key in (
            "mismatch_checks",
            "mismatch_thresholds",
            "skip_inline_tags",
            "alignment_settings",
            "tag_profile",
        ):
            default_nested = AppSettingsManager.DEFAULT_SETTINGS.get(nested_key, {})
            if not isinstance(settings.get(nested_key), dict):
                settings[nested_key] = default_nested.copy()
            else:
                for k, v in default_nested.items():
                    settings[nested_key].setdefault(k, v)

                settings[nested_key].pop('ps_markers', None)

        try:
            with open(AppSettingsManager.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save merged settings: {e}")

        return settings

    @staticmethod
    def save_settings(settings: Dict) -> None:
        validated_settings = {}

        for key, default_val in AppSettingsManager.DEFAULT_SETTINGS.items():
            if isinstance(default_val, dict):
                src = settings.get(key, {})
                merged = default_val.copy()
                merged.update(src)

                for obsolete_key in list(merged.keys()):
                    if obsolete_key not in default_val:
                        merged.pop(obsolete_key, None)
                validated_settings[key] = merged
            else:
                validated_settings[key] = settings.get(key, default_val)

        validated_settings['response_wrapper_tag_variants'] = normalize_wrapper_variants(
            settings.get('response_wrapper_tag_variants'),
            validated_settings.get('response_wrapper_tag') or 'translated'
        )

        for obsolete in ('use_ps_markers', 'restore_paragraph_epub', 'restore_paragraph_txt'):
            validated_settings.pop(obsolete, None)

        _aliases = {'swiss': 'french', 'russian': 'french'}
        if validated_settings.get('quote_style') in _aliases:
            validated_settings['quote_style'] = _aliases[validated_settings['quote_style']]

        try:
            with open(AppSettingsManager.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(validated_settings, f, ensure_ascii=False, indent=2)
            logger.info("App settings saved successfully")
        except Exception as e:
            logger.error(f"Failed to save app settings: {e}")
            raise

    @staticmethod
    def get_llm_client_config(settings: Dict) -> Dict:
        llm_choice = settings.get('llm_choice', 'LM Studio')

        if llm_choice == 'Ollama':
            model_name = settings.get('ollama_model_name', '')
            api_key = None
            endpoint = (
                settings.get('ollama_endpoint', '').strip()
                or 'http://localhost:11434'
            )

        elif llm_choice == 'Openrouter':
            model_name = settings.get('openrouter_model_name', '')
            api_key = settings.get('openrouter_api_key', '')
            endpoint = 'https://openrouter.ai/api/v1'

        else:
            model_name = 'local-model'
            api_key = None
            endpoint = (
                settings.get('server_url', '').strip()
                or 'http://localhost:1234/v1/chat/completions'
            )

        return {
            'llm_choice': llm_choice,
            'model_name': model_name,
            'api_key': api_key,
            'endpoint': endpoint
        }

class PromptManager:
    PROMPT_VARIANTS = ("txt", "srt", "epub_legacy", "epub_inline")
    PROMPT_ROLES = (
        "system",
        "assistant",
        "batch_assistant",
        "user",
        "ollama",
        "json_payload",
        "batch_json_payload",
        "batch_hint",
    )
    DEFAULT_PROMPT_PROFILE = {
        "reserve_prefix": "id_",
        "inline_prefix": "p_",
        "non_trans_prefix": "nt_",
        "paragraph_separator": "ps",
        "batch_separator": "z",
        "core_text_var": "core_text",
        "context_before_var": "context_before",
        "context_after_var": "context_after",
        "response_wrapper_tag": "translated",
    }
    _active_profile = dict(DEFAULT_PROMPT_PROFILE)

    @staticmethod
    def normalize_prompt_profile(profile: Optional[Dict]) -> Dict[str, str]:
        result = dict(PromptManager.DEFAULT_PROMPT_PROFILE)
        if isinstance(profile, dict):
            for key in result:
                value = profile.get(key)
                if isinstance(value, str) and value.strip():
                    result[key] = value.strip()
        return result

    @staticmethod
    def profile_from_settings(settings: Optional[Dict]) -> Dict[str, str]:
        flat = {}
        if isinstance(settings, dict):
            tag_profile = settings.get("tag_profile")
            if isinstance(tag_profile, dict):
                flat.update(tag_profile)
            flat["response_wrapper_tag"] = settings.get("response_wrapper_tag")
        return PromptManager.normalize_prompt_profile(flat)

    @staticmethod
    def resolve_prompt_profile(settings: Optional[Dict]) -> Dict[str, str]:
        if isinstance(settings, dict):
            saved = settings.get("prompt_tag_profile")
            if isinstance(saved, dict) and saved:
                return PromptManager.normalize_prompt_profile(saved)
        return PromptManager.profile_from_settings(settings)

    @staticmethod
    def set_active_profile(profile: Optional[Dict]) -> None:
        PromptManager._active_profile = PromptManager.normalize_prompt_profile(profile)

    @staticmethod
    def convert_prompt_text(text: str, from_profile: Dict, to_profile: Dict) -> str:
        if not isinstance(text, str) or not text:
            return text

        src = PromptManager.normalize_prompt_profile(from_profile)
        dst = PromptManager.normalize_prompt_profile(to_profile)
        if src == dst:
            return text

        esc = re.escape
        number = r'(?:\d{2}|XX|NN)'
        parts = [
            r'(?P<batch><' + esc(src["batch_separator"]) + r'>)',
            r'(?P<para><(?P<para_slash>/?)' + esc(src["paragraph_separator"]) + r'>)',
            r'(?P<wrap><(?P<wrap_slash>/?)' + esc(src["response_wrapper_tag"]) + r'>)',
            r'(?P<nt><' + esc(src["non_trans_prefix"]) + r'(?P<nt_num>' + number + r')/>)',
            r'(?P<res><(?P<res_slash>/?)' + esc(src["reserve_prefix"]) + r'(?P<res_num>' + number + r')>)',
            r'(?P<inl><(?P<inl_slash>/?)' + esc(src["inline_prefix"]) + r'(?P<inl_num>' + number + r')>)',
            r'(?P<core>\{' + esc(src["core_text_var"]) + r'\})',
            r'(?P<before>\{' + esc(src["context_before_var"]) + r'\})',
            r'(?P<after>\{' + esc(src["context_after_var"]) + r'\})',
        ]
        pattern = re.compile('|'.join(parts))

        def replace(match):
            if match.group('batch') is not None:
                return '<' + dst["batch_separator"] + '>'
            if match.group('para') is not None:
                return '<' + match.group('para_slash') + dst["paragraph_separator"] + '>'
            if match.group('wrap') is not None:
                return '<' + match.group('wrap_slash') + dst["response_wrapper_tag"] + '>'
            if match.group('nt') is not None:
                return '<' + dst["non_trans_prefix"] + match.group('nt_num') + '/>'
            if match.group('res') is not None:
                return '<' + match.group('res_slash') + dst["reserve_prefix"] + match.group('res_num') + '>'
            if match.group('inl') is not None:
                return '<' + match.group('inl_slash') + dst["inline_prefix"] + match.group('inl_num') + '>'
            if match.group('core') is not None:
                return '{' + dst["core_text_var"] + '}'
            if match.group('before') is not None:
                return '{' + dst["context_before_var"] + '}'
            return '{' + dst["context_after_var"] + '}'

        return pattern.sub(replace, text)

    @staticmethod
    def adapt_default_text(text: str) -> str:
        return PromptManager.convert_prompt_text(
            text,
            PromptManager.DEFAULT_PROMPT_PROFILE,
            PromptManager._active_profile
        )

    @staticmethod
    def convert_prompt_cache(cache: Dict, from_profile: Dict, to_profile: Dict) -> Dict:
        converted = {}
        for variant, prompts in (cache or {}).items():
            if not isinstance(prompts, dict):
                converted[variant] = prompts
                continue
            converted[variant] = {}
            for role, text in prompts.items():
                converted[variant][role] = PromptManager.convert_prompt_text(
                    text, from_profile, to_profile
                )
        return converted

    @staticmethod
    def migrate_prompt_files(from_profile: Dict, to_profile: Dict) -> List[str]:
        src = PromptManager.normalize_prompt_profile(from_profile)
        dst = PromptManager.normalize_prompt_profile(to_profile)
        if src == dst:
            return []

        pending = []
        for variant in PromptManager.PROMPT_VARIANTS:
            for role in PromptManager.PROMPT_ROLES:
                filename = f"llm_prompt_{role}_{variant}.txt"
                if not os.path.exists(filename):
                    continue
                with open(filename, 'r', encoding='utf-8') as f:
                    original = f.read()
                converted = PromptManager.convert_prompt_text(original, src, dst)
                if converted != original:
                    pending.append((filename, original, converted))

        if not pending:
            return []

        temp_files = []
        try:
            for filename, _original, converted in pending:
                temp_path = filename + ".tmp"
                temp_files.append(temp_path)
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(converted)
        except Exception:
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
            raise

        replaced = []
        try:
            for (filename, original, _converted), temp_path in zip(pending, temp_files):
                os.replace(temp_path, filename)
                replaced.append((filename, original))
                logger.info(f"Prompt tags renamed in: {filename}")
        except Exception:
            for filename, original in replaced:
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(original)
                except Exception as restore_error:
                    logger.error(f"Failed to restore {filename}: {restore_error}")
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
            raise

        return [filename for filename, _original, _converted in pending]
    
    @staticmethod
    def get_prompt(variant: str, role: str) -> str:
        filename = f"llm_prompt_{role}_{variant}.txt"

        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                logger.info(f"Loaded custom prompt: {filename}")
                return content
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}, using factory")

        if role == 'system':
            return PromptManager.get_default_system_prompt(variant)
        elif role == 'assistant':
            return PromptManager.get_default_assistant_prompt(variant)
        elif role == 'user':
            return PromptManager.get_default_user_prompt(variant)
        elif role == 'ollama':
            return PromptManager.get_default_ollama_prompt(variant)
        else:
            logger.error(f"Unknown prompt role: {role}")
            return ""

    @staticmethod
    def save_prompt(variant: str, role: str, content: str) -> str:
        filename = f"llm_prompt_{role}_{variant}.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved prompt: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            raise

    @staticmethod
    def reset_to_factory(variant: str) -> None:
        logger.info(f"Factory reset requested for variant: {variant} (files preserved)")

    @staticmethod
    def hard_reset(variant: str) -> int:
        return PromptManager.delete_prompts_for_variant(variant)

    @staticmethod
    def load_prompts_for_variant(variant: str) -> Dict[str, str]:
        result = {}

        system_file = f"llm_prompt_system_{variant}.txt"
        if os.path.exists(system_file):
            try:
                with open(system_file, 'r', encoding='utf-8') as f:
                    result['system'] = f.read().strip()
                logger.info(f"Loaded custom system prompt: {system_file}")
            except Exception as e:
                logger.warning(f"Failed to load {system_file}: {e}, using factory")
                result['system'] = PromptManager.get_default_system_prompt(variant)
        else:
            result['system'] = PromptManager.get_default_system_prompt(variant)

        assistant_file = f"llm_prompt_assistant_{variant}.txt"
        if os.path.exists(assistant_file):
            try:
                with open(assistant_file, 'r', encoding='utf-8') as f:
                    result['assistant'] = f.read().strip()
                logger.info(f"Loaded custom assistant prompt: {assistant_file}")
            except Exception as e:
                logger.warning(f"Failed to load {assistant_file}: {e}, using factory")
                result['assistant'] = PromptManager.get_default_assistant_prompt(variant)
        else:
            result['assistant'] = PromptManager.get_default_assistant_prompt(variant)

        batch_assistant_file = f"llm_prompt_batch_assistant_{variant}.txt"
        if os.path.exists(batch_assistant_file):
            try:
                with open(batch_assistant_file, 'r', encoding='utf-8') as f:
                    result['batch_assistant'] = f.read().strip() or PromptManager.get_default_batch_assistant_prompt(variant)
                logger.info(f"Loaded custom batch assistant prompt: {batch_assistant_file}")
            except Exception as e:
                logger.warning(f"Failed to load {batch_assistant_file}: {e}, using factory")
                result['batch_assistant'] = PromptManager.get_default_batch_assistant_prompt(variant)
        else:
            result['batch_assistant'] = PromptManager.get_default_batch_assistant_prompt(variant)

        user_file = f"llm_prompt_user_{variant}.txt"
        if os.path.exists(user_file):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    result['user'] = f.read().strip()
                logger.info(f"Loaded custom user prompt: {user_file}")
            except Exception as e:
                logger.warning(f"Failed to load {user_file}: {e}, using factory")
                result['user'] = PromptManager.get_default_user_prompt(variant)
        else:
            result['user'] = PromptManager.get_default_user_prompt(variant)

        ollama_file = f"llm_prompt_ollama_{variant}.txt"
        if os.path.exists(ollama_file):
            try:
                with open(ollama_file, 'r', encoding='utf-8') as f:
                    result['ollama'] = f.read().strip()
                logger.info(f"Loaded custom Ollama prompt: {ollama_file}")
            except Exception as e:
                logger.warning(f"Failed to load {ollama_file}: {e}, using factory")
                result['ollama'] = PromptManager.get_default_ollama_prompt(variant)
        else:
            result['ollama'] = PromptManager.get_default_ollama_prompt(variant)

        json_payload_file = f"llm_prompt_json_payload_{variant}.txt"
        if os.path.exists(json_payload_file):
            try:
                with open(json_payload_file, 'r', encoding='utf-8') as f:
                    result['json_payload'] = f.read().strip()
                logger.info(f"Loaded JSON payload template: {json_payload_file}")
            except Exception as e:
                logger.warning(f"Failed to load {json_payload_file}: {e}, using factory")
                result['json_payload'] = PromptManager.get_default_json_payload_prompt(variant)
        else:
            result['json_payload'] = PromptManager.get_default_json_payload_prompt(variant)

        batch_json_payload_file = f"llm_prompt_batch_json_payload_{variant}.txt"
        if os.path.exists(batch_json_payload_file):
            try:
                with open(batch_json_payload_file, 'r', encoding='utf-8') as f:
                    result['batch_json_payload'] = f.read().strip()
                logger.info(f"Loaded batch JSON payload template: {batch_json_payload_file}")
            except Exception as e:
                logger.warning(f"Failed to load {batch_json_payload_file}: {e}, using factory")
                result['batch_json_payload'] = PromptManager.get_default_batch_json_payload_prompt(variant)
        else:
            result['batch_json_payload'] = PromptManager.get_default_batch_json_payload_prompt(variant)

        batch_hint_file = f"llm_prompt_batch_hint_{variant}.txt"
        if os.path.exists(batch_hint_file):
            try:
                with open(batch_hint_file, 'r', encoding='utf-8') as f:
                    result['batch_hint'] = f.read().strip()
                logger.info(f"Loaded batch hint template: {batch_hint_file}")
            except Exception as e:
                logger.warning(f"Failed to load {batch_hint_file}: {e}, using factory")
                result['batch_hint'] = PromptManager.get_default_batch_hint_prompt()
        else:
            result['batch_hint'] = PromptManager.get_default_batch_hint_prompt()

        return result

    @staticmethod
    def get_default_system_prompt(variant: str) -> str:
        base = """You are a professional translator from English to Polish.

1. Return ONLY translated text in this format:
<translated>Your translation here</translated>

2. CRITICAL: No text before/after tags. No explanations.

3. Translate ONLY text inside <text_to_translate> tags.
Context in <context_before>/<context_after> is for reference only."""

        if variant == "txt":
            return base + """

4. Preserve natural paragraph breaks and text structure.

5. Keep URLs, emails, proper nouns as-is.

6. Maintain author's meaning and natural Polish flow."""

        elif variant == "srt":
            return base + """

4. LINE PRESERVATION:
- Maintain line breaks (\\n) from original
- If original has 2 lines, translation must have 2 lines
- Split text naturally at phrase boundaries

5. DIALOGUE DASHES:
- Keep dialogue dashes (- ) at line start if present in original

6. Keep URLs, emails, proper nouns as-is.

7. LINE PRESERVATION:
- If the text contains <ps> markers, preserve them EXACTLY — do not remove, add, or move them
- Each <ps> represents a line break; the translation must have the same number of <ps> markers as the original
- Do NOT translate, rename, or omit <ps>

8. Maintain natural Polish subtitle flow."""

        elif variant == "epub_legacy":
            return base + """

4. RESERVE ELEMENTS (structural):
Format: <id_00>, <id_01>, etc.
These mark images, <br> tags, non-translatable elements.
- Copy EXACTLY as shown
- Keep approximate position in text
- Do NOT modify or remove

5. Keep URLs, emails, proper nouns as-is.

6. Maintain author's meaning and natural Polish flow."""

        elif variant == "epub_inline":
            return base + """

4. PLACEHOLDERS — copy all exactly, preserve positions:
- <id_00>, <id_01>… — images/<br>/structural (do NOT modify)
- <p_00>…</p_00>… — inline formatting (<i>/<b>/etc.); every opening needs matching closing, keep around same words
- <nt_00/>, <nt_01/>… — non-translatable spaces/anchors (self-closing, do NOT move)
- <ps> — paragraph break; count in translation must match original EXACTLY

5. Keep URLs, emails, proper nouns as-is.
6. Maintain author's meaning and natural Polish flow."""

        return base

    @staticmethod
    def get_default_assistant_prompt(variant: str) -> str:
        base = """Context for reference only. Do NOT translate.

<context_before>{context_before}</context_before>

<context_after>{context_after}</context_after>

I understand. I will translate ONLY <text_to_translate>, using context for narrative consistency."""

        if variant == "txt":
            return base

        elif variant == "srt":
            return base

        elif variant == "epub_legacy":
            return base

        elif variant == "epub_inline":
            return base

        return base

    @staticmethod
    def get_default_batch_assistant_prompt(variant: str) -> str:
        return """Context for reference only. Do NOT translate.

<context_before>{context_before}</context_before>

<context_after>{context_after}</context_after>

I understand. I will translate every segment inside <text_to_translate>, preserving all batch separator markers exactly, and use the context only for narrative consistency."""

    @staticmethod
    def get_default_user_prompt(variant: str) -> str:
        base = """<text_to_translate>{core_text}</text_to_translate>

Translate FULL text inside <text_to_translate> tags. Do not skip, shorten, or omit any part. Keep all placeholders and formatting intact, and preserve the tone, including any strong language.

Return format:
<translated>Translation here</translated>"""

        if variant == "txt":
            return base

        elif variant == "srt":
            return base

        elif variant == "epub_inline":
            return base

        return base

    @staticmethod
    def get_default_ollama_prompt(variant: str) -> str:
        system = PromptManager.get_default_system_prompt(variant)
        assistant = PromptManager.get_default_assistant_prompt(variant)
        user = PromptManager.get_default_user_prompt(variant)

        return f"""{system}

{assistant}

{user}"""

    @staticmethod
    def get_default_json_payload_prompt(variant: str) -> str:
        if variant == "txt":
            variant_instructions = (
                "Translate the following text to Polish.\\n"
                "Preserve natural paragraph breaks and text structure.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n"
                "Maintain the author's meaning and natural Polish flow.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text to translate:\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
        elif variant == "srt":
            variant_instructions = (
                "Translate the following text to Polish.\\n"
                "LINE PRESERVATION: maintain line breaks (\\\\n) from the original.\\n"
                "If the original has 2 lines, the translation must have exactly 2 lines.\\n"
                "Keep dialogue dashes (- ) at line start if present in original.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text to translate:\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
        elif variant == "epub_legacy":
            variant_instructions = (
                "Translate the following text to Polish.\\n"
                "RESERVE ELEMENTS — copy exactly, keep approximate position:\\n"
                "  <id_00>, <id_01>, ... — images, <br> tags, non-translatable elements.\\n"
                "Do NOT modify or remove them.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text to translate:\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
        elif variant == "epub_inline":
            variant_instructions = (
                "Translate the following text to Polish.\\n"
                "PLACEHOLDERS — copy all exactly and preserve their positions:\\n"
                "  <id_00>, <id_01>... — images/<br>/structural (do NOT modify)\\n"
                "  <p_00>...</p_00>... — inline formatting (<i>/<b>/etc.); every opening needs matching closing\\n"
                "  <nt_00/>, <nt_01/>... — non-translatable anchors (self-closing, do NOT move)\\n"
                "  <ps> — paragraph break; count in translation must match original EXACTLY\\n"
                "NEVER use raw HTML tags like <i>, <b>, <em> — use only <p_XX> placeholders.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text to translate:\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
        else:
            variant_instructions = (
                "Translate the following text to Polish.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text to translate:\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
 
        return (
            '[\n'
            '  {\n'
            '    "type": "text",\n'
            '    "text": "' + variant_instructions + '"\n'
            '  }\n'
            ']'
        )
 
    @staticmethod
    def get_default_json_response_field(variant: str) -> str:
        return ""
 
    @staticmethod
    def get_default_batch_json_payload_prompt(variant: str) -> str:
        if variant == "txt":
            variant_instructions = (
                "Translate the following text segments to Polish.\\n"
                "The segments are separated by <z> markers.\\n"
                "Preserve ALL <z> markers in the same positions — "
                "there must be exactly the same number of <z> markers as in the input.\\n"
                "Preserve natural paragraph breaks and text structure.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n"
                "Maintain the author's meaning and natural Polish flow.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text segments to translate (separated by <z>):\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
        elif variant == "srt":
            variant_instructions = (
                "Translate the following text segments to Polish.\\n"
                "The segments are separated by <z> markers.\\n"
                "Preserve ALL <z> markers in the same positions — "
                "there must be exactly the same number of <z> markers as in the input.\\n"
                "LINE PRESERVATION: maintain line breaks (\\\\n) from the original.\\n"
                "Keep dialogue dashes (- ) at line start if present in original.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text segments to translate (separated by <z>):\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
        elif variant == "epub_legacy":
            variant_instructions = (
                "Translate the following text segments to Polish.\\n"
                "The segments are separated by <z> markers.\\n"
                "Preserve ALL <z> markers in the same positions — "
                "there must be exactly the same number of <z> markers as in the input.\\n"
                "RESERVE ELEMENTS — copy exactly, keep approximate position:\\n"
                "  <id_00>, <id_01>, ... — images, <br> tags, non-translatable elements.\\n"
                "Do NOT modify or remove them.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text segments to translate (separated by <z>):\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
        elif variant == "epub_inline":
            variant_instructions = (
                "Translate the following text segments to Polish.\\n"
                "The segments are separated by <z> markers.\\n"
                "Preserve ALL <z> markers in the same positions — "
                "there must be exactly the same number of <z> markers as in the input.\\n"
                "PLACEHOLDERS — copy all exactly and preserve their positions:\\n"
                "  <id_00>, <id_01>... — images/<br>/structural (do NOT modify)\\n"
                "  <p_00>...</p_00>... — inline formatting (<i>/<b>/etc.); every opening needs matching closing\\n"
                "  <nt_00/>, <nt_01/>... — non-translatable anchors (self-closing, do NOT move)\\n"
                "  <ps> — paragraph break; count in translation must match original EXACTLY\\n"
                "NEVER use raw HTML tags like <i>, <b>, <em> — use only <p_XX> placeholders.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text segments to translate (separated by <z>):\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
        else:
            variant_instructions = (
                "Translate the following text segments to Polish.\\n"
                "The segments are separated by <z> markers.\\n"
                "Preserve ALL <z> markers in the same positions — "
                "there must be exactly the same number of <z> markers as in the input.\\n\\n"
                "Context before (for reference only, do NOT translate):\\n"
                "{context_before}\\n\\n"
                "Text segments to translate (separated by <z>):\\n"
                "{core_text}\\n\\n"
                "Context after (for reference only, do NOT translate):\\n"
                "{context_after}"
            )
 
        return (
            '[\n'
            '  {\n'
            '    "type": "text",\n'
            '    "text": "' + variant_instructions + '"\n'
            '  }\n'
            ']'
        )
    
    @staticmethod
    def get_default_batch_hint_prompt() -> str:
        return (
            "=== CRITICAL BATCH INSTRUCTION ===\n"
            "The text below contains exactly {marker_count} separator marker(s) <z> dividing it into {n} separate paragraphs.\n"
            "RULES — you MUST follow ALL of them:\n"
            "1. Preserve every <z> marker exactly as-is — do NOT remove, add, move, or translate them.\n"
            "2. Output EXACTLY {marker_count} <z> marker(s) in your translation — no more, no fewer.\n"
            "3. The <z> marker is NOT an HTML tag — treat it as a literal paragraph boundary separator.\n"
            "4. Translate each segment between <z> markers independently.\n"
            "==================================="
        )

    @staticmethod
    def save_prompts_for_variant(
        variant: str,
        system: str = None,
        assistant: str = None,
        batch_assistant: str = None,
        user: str = None,
        ollama: str = None,
        json_payload: str = None,
        batch_json_payload: str = None,
        batch_hint: str = None,
    ) -> List[str]:
        saved_files = []

        prompts_map = {
            'system':             system,
            'assistant':          assistant,
            'batch_assistant':    batch_assistant,
            'user':               user,
            'ollama':             ollama,
            'json_payload':       json_payload,
            'batch_json_payload': batch_json_payload,
            'batch_hint':         batch_hint,
        }

        try:
            for role, content in prompts_map.items():
                if content is not None:
                    filename = f"llm_prompt_{role}_{variant}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    saved_files.append(filename)
                    logger.info(f"Saved: {filename}")

            return saved_files

        except Exception as e:
            logger.error(f"Failed to save prompts for variant {variant}: {e}")
            raise

    @staticmethod
    def delete_prompts_for_variant(variant: str) -> int:
        files_to_delete = [
            f"llm_prompt_system_{variant}.txt",
            f"llm_prompt_assistant_{variant}.txt",
            f"llm_prompt_batch_assistant_{variant}.txt",
            f"llm_prompt_user_{variant}.txt",
            f"llm_prompt_ollama_{variant}.txt",
            f"llm_prompt_json_payload_{variant}.txt",
            f"llm_prompt_batch_json_payload_{variant}.txt",
            f"llm_prompt_batch_hint_{variant}.txt",
        ]

        deleted_count = 0
        for filename in files_to_delete:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    deleted_count += 1
                    logger.info(f"Deleted: {filename}")
            except Exception as e:
                logger.error(f"Failed to delete {filename}: {e}")

        return deleted_count

def _adapt_default_prompt(original):
    @_functools.wraps(original)
    def wrapper(*args, **kwargs):
        return PromptManager.adapt_default_text(original(*args, **kwargs))
    return wrapper


for _method_name in (
    'get_default_system_prompt',
    'get_default_assistant_prompt',
    'get_default_batch_assistant_prompt',
    'get_default_user_prompt',
    'get_default_json_payload_prompt',
    'get_default_batch_json_payload_prompt',
    'get_default_batch_hint_prompt',
):
    setattr(
        PromptManager,
        _method_name,
        staticmethod(_adapt_default_prompt(getattr(PromptManager, _method_name)))
    )

class EPUBConstants:
    PRIORITY_TAGS = ['p', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote']

    NON_INLINE_ELEMENTS = {
        'address', 'blockquote', 'dialog', 'div', 'figure', 'figcaption',
        'footer', 'header', 'legend', 'main', 'p', 'pre', 'search', 'article',
        'aside', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hgroup', 'nav',
        'section', 'dd', 'dl', 'dt', 'menu', 'ol', 'ul', 'table', 'caption',
        'colgroup', 'col', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th', 'li'
    }

    CONTAINER_TAGS = {
        'ul', 'ol', 'dl',
        'table', 'tbody', 'thead', 'tfoot', 'tr',
        'div', 'section', 'article', 'aside', 'nav', 'main',
        'header', 'footer', 'figure', 'body'
    }

    NOISE_TAGS = ['rt', 'rp']

    RESERVE_TAGS = [
        'img', 'code', 'br', 'hr', 'sub', 'sup', 'kbd',
        'abbr', 'wbr', 'var', 'canvas', 'svg', 'script',
        'style', 'math'
    ]

    INLINE_FORMATTING_TAGS = ['i', 'b', 'em', 'strong', 'u', 'sup', 'sub', 'small', 'span']

    STRUCTURAL_SPAN_CLASSES = {
        'first-letter',
        'last-word',
        'item-number',
        'element-number',
    }

class SRTConstants:
    MAX_LINE_LENGTH = 42

    MAX_LINES_PER_BLOCK = 2

    DIALOGUE_DASHES = ['-', '–', '—']

class QuoteConstants:
    ALL_QUOTES = r'["\'\u2018\u2019\u201C\u201D\u201E\u201F\u00AB\u00BB\u2039\u203A\u201A\u201B\u2032\u2033\u301D\u301E\u301F\uFF02]'

    DOUBLE_QUOTES = r'["\u201C\u201D\u201E\u201F\u00AB\u00BB\u301D\u301E\u301F\uFF02]'

    SINGLE_QUOTES = r'[\'\u2018\u2019\u201A\u201B\u2032\u2039\u203A]'

    POLISH_OPENING = '\u201e'
    POLISH_CLOSING = '\u201d'

class LanguageConstants:
    # Format: (Display Name, language code)
    # Codes follow DeepL conventions.
    # Google Translate uses lowercase codes; DeepL uses uppercase.
    # The combo stores the code as UserData; display shows the full name.

    SOURCE_LANGUAGES = [
        ("Auto Detect",                 "auto"),
        # — European —
        ("Albanian",                    "sq"),
        ("Basque",                      "eu"),
        ("Belarusian",                  "be"),
        ("Bosnian",                     "bs"),
        ("Bulgarian",                   "bg"),
        ("Catalan",                     "ca"),
        ("Croatian",                    "hr"),
        ("Czech",                       "cs"),
        ("Danish",                      "da"),
        ("Dutch",                       "nl"),
        ("English",                     "en"),
        ("Estonian",                    "et"),
        ("Finnish",                     "fi"),
        ("French",                      "fr"),
        ("Galician",                    "gl"),
        ("German",                      "de"),
        ("Greek",                       "el"),
        ("Hungarian",                   "hu"),
        ("Icelandic",                   "is"),
        ("Irish",                       "ga"),
        ("Italian",                     "it"),
        ("Latvian",                     "lv"),
        ("Lithuanian",                  "lt"),
        ("Luxembourgish",               "lb"),
        ("Macedonian",                  "mk"),
        ("Maltese",                     "mt"),
        ("Norwegian (Bokmål)",          "nb"),
        ("Polish",                      "pl"),
        ("Portuguese",                  "pt"),
        ("Romanian",                    "ro"),
        ("Russian",                     "ru"),
        ("Serbian",                     "sr"),
        ("Slovak",                      "sk"),
        ("Slovenian",                   "sl"),
        ("Spanish",                     "es"),
        ("Swedish",                     "sv"),
        ("Turkish",                     "tr"),
        ("Ukrainian",                   "uk"),
        ("Welsh",                       "cy"),
        # — Middle East / Central Asia —
        ("Arabic",                      "ar"),
        ("Armenian",                    "hy"),
        ("Azerbaijani",                 "az"),
        ("Georgian",                    "ka"),
        ("Hebrew",                      "he"),
        ("Kazakh",                      "kk"),
        ("Kurdish (Kurmanji)",          "ku"),
        ("Kyrgyz",                      "ky"),
        ("Persian (Farsi)",             "fa"),
        ("Tajik",                       "tg"),
        ("Turkmen",                     "tk"),
        ("Uzbek",                       "uz"),
        # — South / Southeast Asia —
        ("Bengali",                     "bn"),
        ("Burmese",                     "my"),
        ("Filipino (Tagalog)",          "tl"),
        ("Gujarati",                    "gu"),
        ("Hindi",                       "hi"),
        ("Indonesian",                  "id"),
        ("Javanese",                    "jv"),
        ("Kannada",                     "kn"),
        ("Khmer",                       "km"),
        ("Lao",                         "lo"),
        ("Malay",                       "ms"),
        ("Malayalam",                   "ml"),
        ("Marathi",                     "mr"),
        ("Nepali",                      "ne"),
        ("Odia (Oriya)",                "or"),
        ("Punjabi",                     "pa"),
        ("Sinhala",                     "si"),
        ("Tamil",                       "ta"),
        ("Telugu",                      "te"),
        ("Thai",                        "th"),
        ("Urdu",                        "ur"),
        ("Vietnamese",                  "vi"),
        # — East Asia —
        ("Chinese (Simplified)",        "zh-CN"),
        ("Chinese (Traditional)",       "zh-TW"),
        ("Japanese",                    "ja"),
        ("Korean",                      "ko"),
        ("Mongolian",                   "mn"),
        # — Africa —
        ("Afrikaans",                   "af"),
        ("Amharic",                     "am"),
        ("Hausa",                       "ha"),
        ("Igbo",                        "ig"),
        ("Malagasy",                    "mg"),
        ("Sesotho",                     "st"),
        ("Shona",                       "sn"),
        ("Somali",                      "so"),
        ("Swahili",                     "sw"),
        ("Xhosa",                       "xh"),
        ("Yoruba",                      "yo"),
        ("Zulu",                        "zu"),
        # — Americas / Other —
        ("Cebuano",                     "ceb"),
        ("Corsican",                    "co"),
        ("Esperanto",                   "eo"),
        ("Frisian",                     "fy"),
        ("Hawaiian",                    "haw"),
        ("Hmong",                       "hmn"),
        ("Latin",                       "la"),
        ("Maori",                       "mi"),
        ("Samoan",                      "sm"),
        ("Scots Gaelic",                "gd"),
        ("Sindhi",                      "sd"),
        ("Sundanese",                   "su"),
        ("Uyghur",                      "ug"),
        ("Yiddish",                     "yi"),
    ]

    TARGET_LANGUAGES = [
        # — European —
        ("Albanian",                    "sq"),
        ("Basque",                      "eu"),
        ("Belarusian",                  "be"),
        ("Bosnian",                     "bs"),
        ("Bulgarian",                   "bg"),
        ("Catalan",                     "ca"),
        ("Croatian",                    "hr"),
        ("Czech",                       "cs"),
        ("Danish",                      "da"),
        ("Dutch",                       "nl"),
        ("English",                     "en"),
        ("English (British)",           "en-GB"),
        ("English (American)",          "en-US"),
        ("Estonian",                    "et"),
        ("Finnish",                     "fi"),
        ("French",                      "fr"),
        ("Galician",                    "gl"),
        ("German",                      "de"),
        ("Greek",                       "el"),
        ("Hungarian",                   "hu"),
        ("Icelandic",                   "is"),
        ("Irish",                       "ga"),
        ("Italian",                     "it"),
        ("Latvian",                     "lv"),
        ("Lithuanian",                  "lt"),
        ("Luxembourgish",               "lb"),
        ("Macedonian",                  "mk"),
        ("Maltese",                     "mt"),
        ("Norwegian (Bokmål)",          "nb"),
        ("Polish",                      "pl"),
        ("Portuguese",                  "pt"),
        ("Portuguese (Portugal)",       "pt-PT"),
        ("Portuguese (Brazil)",         "pt-BR"),
        ("Romanian",                    "ro"),
        ("Russian",                     "ru"),
        ("Serbian",                     "sr"),
        ("Slovak",                      "sk"),
        ("Slovenian",                   "sl"),
        ("Spanish",                     "es"),
        ("Swedish",                     "sv"),
        ("Turkish",                     "tr"),
        ("Ukrainian",                   "uk"),
        ("Welsh",                       "cy"),
        # — Middle East / Central Asia —
        ("Arabic",                      "ar"),
        ("Armenian",                    "hy"),
        ("Azerbaijani",                 "az"),
        ("Georgian",                    "ka"),
        ("Hebrew",                      "he"),
        ("Kazakh",                      "kk"),
        ("Kurdish (Kurmanji)",          "ku"),
        ("Kyrgyz",                      "ky"),
        ("Persian (Farsi)",             "fa"),
        ("Tajik",                       "tg"),
        ("Turkmen",                     "tk"),
        ("Uzbek",                       "uz"),
        # — South / Southeast Asia —
        ("Bengali",                     "bn"),
        ("Burmese",                     "my"),
        ("Filipino (Tagalog)",          "tl"),
        ("Gujarati",                    "gu"),
        ("Hindi",                       "hi"),
        ("Indonesian",                  "id"),
        ("Javanese",                    "jv"),
        ("Kannada",                     "kn"),
        ("Khmer",                       "km"),
        ("Lao",                         "lo"),
        ("Malay",                       "ms"),
        ("Malayalam",                   "ml"),
        ("Marathi",                     "mr"),
        ("Nepali",                      "ne"),
        ("Odia (Oriya)",                "or"),
        ("Punjabi",                     "pa"),
        ("Sinhala",                     "si"),
        ("Tamil",                       "ta"),
        ("Telugu",                      "te"),
        ("Thai",                        "th"),
        ("Urdu",                        "ur"),
        ("Vietnamese",                  "vi"),
        # — East Asia —
        ("Chinese (Simplified)",        "zh-CN"),
        ("Chinese (Traditional)",       "zh-TW"),
        ("Japanese",                    "ja"),
        ("Korean",                      "ko"),
        ("Mongolian",                   "mn"),
        # — Africa —
        ("Afrikaans",                   "af"),
        ("Amharic",                     "am"),
        ("Hausa",                       "ha"),
        ("Igbo",                        "ig"),
        ("Malagasy",                    "mg"),
        ("Sesotho",                     "st"),
        ("Shona",                       "sn"),
        ("Somali",                      "so"),
        ("Swahili",                     "sw"),
        ("Xhosa",                       "xh"),
        ("Yoruba",                      "yo"),
        ("Zulu",                        "zu"),
        # — Americas / Other —
        ("Cebuano",                     "ceb"),
        ("Corsican",                    "co"),
        ("Esperanto",                   "eo"),
        ("Frisian",                     "fy"),
        ("Hawaiian",                    "haw"),
        ("Hmong",                       "hmn"),
        ("Latin",                       "la"),
        ("Maori",                       "mi"),
        ("Samoan",                      "sm"),
        ("Scots Gaelic",                "gd"),
        ("Sindhi",                      "sd"),
        ("Sundanese",                   "su"),
        ("Uyghur",                      "ug"),
        ("Yiddish",                     "yi"),
    ]

__all__ = [
    'SessionManager',
    'AppSettingsManager',
    'PromptManager',
    'EPUBConstants',
    'SRTConstants',
    'QuoteConstants',
    'LanguageConstants'
]