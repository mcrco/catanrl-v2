"""Test 8: importing PufferLib's trainer keeps catanrl's signal handlers and
warning filters (pufferl turns RuntimeWarning into errors and installs an
os._exit(0) SIGINT handler at import)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

CHECK = textwrap.dedent("""
    import signal, sys, warnings
    def mine(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, mine)
    signal.signal(signal.SIGTERM, mine)
    filters = list(warnings.filters)
    {importer}
    assert signal.getsignal(signal.SIGINT) is mine, "SIGINT handler replaced"
    assert signal.getsignal(signal.SIGTERM) is mine, "SIGTERM handler replaced"
    assert not any(f[0] == "error" and f[2] is RuntimeWarning for f in warnings.filters), "RuntimeWarning is an error"
    warnings.warn("still a warning", RuntimeWarning)
    print("ok")
""")


def run(importer: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", CHECK.format(importer=importer)],
                          check=False, capture_output=True, text=True, timeout=300)


def test_training_module_import_has_no_side_effects():
    result = run("import catanrl.train.puffer")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


def test_catanrl_puffer_import_has_no_side_effects():
    try:
        import catanrl._puffer  # noqa: F401
    except ImportError:
        pytest.skip("catanrl._puffer is not written yet")
    result = run("from catanrl._puffer import import_pufferl; import_pufferl()")
    assert result.returncode == 0, result.stderr[-2000:]
