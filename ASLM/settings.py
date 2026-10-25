# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import sys
from pathlib import Path


# Build project root paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Expose project root for local package imports
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Settings.settings import get_engine_url, get_llm_engine, get_openai_api_key, is_engine_enabled, load_settings


# Load persisted project settings
_cfg = load_settings()


# Core Django settings
SECRET_KEY = _cfg.get("secret_key") or "django-insecure-fallback-key-change-me"
DEBUG = bool(_cfg.get("debug", False))
ALLOWED_HOSTS = _cfg.get("allowed_hosts", ["127.0.0.1", "localhost"])


# Runtime engine settings
LLM_ENGINE = get_llm_engine()
OLLAMA_URL = get_engine_url("ollama-service")
OLLAMA_ENABLED = is_engine_enabled("ollama-service")
LMSTUDIO_URL = get_engine_url("lms")
OPENAI_COMPAT_URL = get_engine_url("openai")
OPENAI_COMPAT_API_KEY = get_openai_api_key() or os.environ.get("OPENAI_API_KEY", "not-needed")


# CORS and CSRF settings
CORS_ALLOWED_ORIGINS = [
    "https://localhost",
    "https://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1",
]
CSRF_TRUSTED_ORIGINS = [
    "https://localhost",
    "https://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1",
]
CORS_ALLOW_METHODS = (
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
)


# Installed apps
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "martor",
    "Apps.Data",
    "Apps.API",
    "Apps.UI",
    "Apps.Tests",
]


# Middleware pipeline
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


# Password hashing and validation
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
    "django.contrib.auth.hashers.ScryptPasswordHasher",
]
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# URL, template, and application entry points
ROOT_URLCONF = "ASLM.urls"
WSGI_APPLICATION = "ASLM.wsgi.application"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


# Database settings
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Localization and encoding
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
DEFAULT_CHARSET = "utf-8"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# Static files
STATIC_ROOT = "staticfiles/"
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static/"]

# WhiteNoise serves static files directly from source directories in development.
WHITENOISE_USE_FINDERS = True
WHITENOISE_AUTOREFRESH = DEBUG


# Upload limits
# Large JSON requests are needed for base64-encoded image uploads.
DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024 * 256
