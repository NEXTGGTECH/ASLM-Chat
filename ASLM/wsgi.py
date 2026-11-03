# Copyright NEXTGGTECH. Elastic License 2.0.

import os

from Settings.proxy_policy import apply_loopback_proxy_bypass

# Install the loopback proxy bypass at import time, before Django and its HTTP clients start,
# so local service traffic never goes through a system proxy.
apply_loopback_proxy_bypass()

from django.core.wsgi import get_wsgi_application


# Set the default Django settings module for WSGI startup unless the environment already defines one.
def _configure_settings_module() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ASLM.settings")


# Create the Django WSGI application after the settings module is configured.
def _create_application():
    _configure_settings_module()
    return get_wsgi_application()


# WSGI callable that servers import by name (ASLM.wsgi:application).
application = _create_application()
