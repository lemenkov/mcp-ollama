# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from mcp_ollama.config import (
    AppConfig,
    ConfigError,
    load_config,
    normalize_base_url,
    redact_url,
    split_userinfo,
)


def make_config(**overrides):
    data = {
        "default_backend": "local",
        "backends": {
            "local": {
                "base_url": "http://127.0.0.1:11434",
                "default_model": "llama3.2:3b",
            }
        },
    }
    data.update(overrides)
    return AppConfig.model_validate(data)


def test_valid_config():
    config = make_config()
    assert config.default_backend == "local"
    assert config.backends["local"].default_model == "llama3.2:3b"


def test_password_file_relative_to_config(tmp_path):
    (tmp_path / "ollama.pass").write_text("s3cret\n", encoding="utf-8")
    config_path = tmp_path / "backends.json"
    config_path.write_text(
        json.dumps(
            {
                "default_backend": "local",
                "backends": {
                    "local": {
                        "base_url": "http://127.0.0.1:11434",
                        "username": "user",
                        "password_file": "ollama.pass",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config, path = load_config(str(config_path))
    assert path == config_path
    assert config.backends["local"].password == "s3cret"


def test_password_and_password_file_conflict():
    with pytest.raises(ValidationError):
        make_config(
            backends={
                "local": {
                    "base_url": "http://127.0.0.1:11434",
                    "username": "u",
                    "password": "a",
                    "password_file": "b",
                }
            }
        )


def test_default_backend_must_exist():
    with pytest.raises(ValidationError):
        make_config(default_backend="missing")


def test_backend_name_validation():
    with pytest.raises(ValidationError):
        make_config(
            backends={
                "bad name": {"base_url": "http://127.0.0.1:11434"}
            }
        )


def test_base_url_validation():
    with pytest.raises(ValidationError):
        make_config(
            backends={"local": {"base_url": "ftp://127.0.0.1:11434"}}
        )


def test_load_missing_config(tmp_path):
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "missing.json"))


def test_split_userinfo():
    user, password, clean = split_userinfo("http://alice:p%40ss@example.com:11434/ollama")
    assert user == "alice"
    assert password == "p@ss"
    assert clean == "http://example.com:11434/ollama"


def test_normalize_base_url():
    assert normalize_base_url("http://example.com/ollama") == "http://example.com/ollama/"
    assert normalize_base_url("http://example.com") == "http://example.com/"


def test_redact_url():
    assert redact_url("https://user:pass@example.com/ollama/") == "https://example.com/ollama/"
