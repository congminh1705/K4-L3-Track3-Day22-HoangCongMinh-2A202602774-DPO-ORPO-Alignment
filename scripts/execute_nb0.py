"""Execute NB0 on CPU and save a real notebook with its outputs.

Requires torch, nbformat, nbclient, and ipykernel (no training dependencies).
python scripts/execute_nb0.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Keep Jupyter caches inside the workspace on restricted Windows machines.
    runtime = REPO / ".venv" / "notebook-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime))
    os.environ.setdefault("IPYTHONDIR", str(runtime / "ipython"))
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    from build_colab import percent_cells

    source = REPO / "notebooks" / "00_dpo_loss_from_scratch.py"
    cells = []
    for cell in percent_cells(source):
        factory = nbformat.v4.new_code_cell if cell["cell_type"] == "code" else nbformat.v4.new_markdown_cell
        cells.append(factory("".join(cell["source"])))
    nb = nbformat.v4.new_notebook(cells=cells)
    manager = KernelManager(kernel_name="python3")
    # Use the invoking virtualenv rather than an unrelated globally installed kernel.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, km=manager, timeout=120, resources={"metadata": {"path": str(REPO)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    destination = source.with_suffix(".ipynb")
    nbformat.write(nb, destination)
    executed = [c for c in nb.cells if c.cell_type == "code"]
    print(f"Executed {len(executed)} code cells; saved {destination}")
    for cell in executed:
        for output in cell.outputs:
            if output.output_type == "stream":
                print(output.text.rstrip())


if __name__ == "__main__":
    main()
