from __future__ import annotations

import os
from pathlib import Path

_GEMINI_KEY_NAME = "GEMINI_API_KEY"
_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _normalize_key(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().strip('"').strip("'")
    return normalized or None


def _get_key_from_streamlit_secrets() -> str | None:
    try:
        import streamlit as st

        key_value = _normalize_key(st.secrets.get(_GEMINI_KEY_NAME))
        if key_value:
            return key_value

        # Also support common nested secret sections.
        for section_name in ("general", "secrets", "api_keys"):
            section = st.secrets.get(section_name)
            if section is None:
                continue

            value = None
            try:
                value = section.get(_GEMINI_KEY_NAME)
            except Exception:
                value = None

            key_value = _normalize_key(value)
            if key_value:
                return key_value

        return None
    except Exception:
        return None


def _get_key_from_env_file() -> str | None:
    if not _ENV_PATH.exists():
        return None

    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        if key.strip() == _GEMINI_KEY_NAME:
            return _normalize_key(value)

    return None


def get_gemini_api_key() -> str | None:
    key_from_secrets = _get_key_from_streamlit_secrets()
    if key_from_secrets:
        os.environ[_GEMINI_KEY_NAME] = key_from_secrets
        return key_from_secrets

    key_from_env = _normalize_key(os.getenv(_GEMINI_KEY_NAME))
    if key_from_env:
        os.environ[_GEMINI_KEY_NAME] = key_from_env
        return key_from_env

    key_from_env_file = _get_key_from_env_file()
    if key_from_env_file:
        os.environ[_GEMINI_KEY_NAME] = key_from_env_file
    return key_from_env_file


def has_gemini_api_key() -> bool:
    return bool(get_gemini_api_key())
