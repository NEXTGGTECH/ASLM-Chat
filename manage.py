# Copyright NEXTGGTECH. Elastic License 2.0.

import os
import sys

from Settings.proxy_policy import apply_loopback_proxy_bypass


# Set the default Django settings module for management commands unless the environment already defines one.
def _configure_settings_module() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ASLM.settings")


# Run Django administrative tasks: install the loopback proxy bypass, configure settings and
# dispatch the command line, with a friendlier error when Django is not installed.
def main() -> None:
    apply_loopback_proxy_bypass()
    _configure_settings_module()

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    execute_from_command_line(sys.argv)


# Run: python manage.py <command>
if __name__ == "__main__":
    main()
