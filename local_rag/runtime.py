"""Child launch contracts for standalone Python and the native Hermes host."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys


def python_command(mode: str, target: str, *args: str) -> list[str]:
    """Run -c/-m in this host's selected dependency environment.

    Native Hermes uses a bare interpreter plus bootstrap activation, not a venv.
    Re-run that bootstrap from the already imported core; never snapshot sys.path
    or retain a dependency-generation path. Only our own package root is added
    afterward, so children execute the same plugin source as their parent.
    """
    if mode not in {'-c', '-m'}:
        raise ValueError('Only Python -c and -m launches are supported')
    bootstrap = sys.modules.get('hermes_bootstrap')
    if bootstrap is None:
        return [sys.executable, mode, target, *args]
    core = str(Path(bootstrap.__file__).resolve().parent)
    plugin_root = str(Path(__file__).resolve().parent.parent)
    code = (
        "import os, sys, runpy; "
        "[os.environ.pop(k, None) for k in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')]; "
        "os.environ['HERMES_DISABLE_LAZY_INSTALLS'] = '1'; "
        f"sys.path.insert(0, {core!r}); import hermes_bootstrap; "
        f"sys.path.insert(0, {plugin_root!r}); "
    )
    if mode == '-m':
        code += f"sys.argv = [{target!r}, *sys.argv[1:]]; runpy.run_module({target!r}, run_name='__main__', alter_sys=True)"
    else:
        code += f"exec(compile({target!r}, '<string>', 'exec'))"
    return [sys.executable, '-I', '-c', code, *args]


def hermes_command(args=()) -> list[str]:
    """Prefer the public SDK launcher; standalone installations use PATH."""
    try:
        from hermes_cli.chat_launch import hermes_command as sdk_command
    except ImportError:
        executable = shutil.which('hermes')
        if executable is None:
            raise FileNotFoundError('Hermes launcher not found on PATH') from None
        return [executable, *args]
    return sdk_command(args)


def hf_command(*args: str) -> list[str]:
    # Same entry point as the hf console script, including versions without __main__.
    return python_command('-c', 'from huggingface_hub.cli.hf import main; main()', *args)
