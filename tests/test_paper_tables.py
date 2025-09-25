from __future__ import annotations

from pathlib import Path

import pandas as pd

import benchmarks.paper_tables as paper_tables


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_generate_tables_creates_latex(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    showcase_dir = results_dir / "showcase"
    showcase_dir.mkdir(parents=True)
    _write_csv(
        showcase_dir / "showcase_raw.csv",
        [
            {
                "circuit": "clustered_ghz_random",
                "qubits": 30,
                "framework": "STATEVECTOR",
                "backend": "STATEVECTOR",
                "run_time_mean": 0.6,
                "run_peak_memory_mean": 2048.0,
            },
            {
                "circuit": "clustered_ghz_random",
                "qubits": 30,
                "framework": "MPS",
                "backend": "MPS",
                "run_time_mean": 0.8,
                "run_peak_memory_mean": 1024.0,
            },
            {
                "circuit": "clustered_ghz_random",
                "qubits": 30,
                "framework": "quasar",
                "backend": "MPS",
                "run_time_mean": 0.4,
                "run_peak_memory_mean": 1536.0,
            },
        ],
    )

    _write_csv(
        results_dir / "staged_rank_summary.csv",
        [
            {
                "scenario": "staged_rank",
                "variant": "staged_rank_1",
                "qubits": 9,
                "total_qubits": 9,
                "quasar_backend": "mps",
                "runtime_speedup": 0.28,
            }
        ],
    )
    _write_csv(
        results_dir / "staged_sparsity_summary.csv",
        [
            {
                "scenario": "staged_sparsity",
                "variant": "staged_sparsity_1",
                "qubits": 10,
                "total_qubits": 10,
                "quasar_backend": "mps",
                "runtime_speedup": 0.27,
            }
        ],
    )
    _write_csv(
        results_dir / "tableau_boundary_summary.csv",
        [
            {
                "scenario": "tableau_boundary",
                "variant": "tableau_boundary_1",
                "qubits": 7,
                "total_qubits": 7,
                "quasar_backend": "dd",
                "runtime_speedup": 0.25,
            }
        ],
    )
    _write_csv(
        results_dir / "dual_magic_injection_summary.csv",
        [
            {
                "scenario": "dual_magic_injection",
                "variant": "dual_magic_injection_1",
                "qubits": 36,
                "total_qubits": 36,
                "quasar_backend": "dd",
                "runtime_speedup": 0.88,
            }
        ],
    )
    _write_csv(
        results_dir / "w_state_oracle_summary.csv",
        [
            {
                "scenario": "w_state_oracle",
                "variant": "w_state_oracle_1",
                "oracle_layers": 2,
                "oracle_rotation_gate_count": 8,
                "oracle_rotation_unique": "RZ,RY",
                "oracle_entangling_count": 4,
                "quasar_backend": "dd",
                "runtime_speedup": 1.1,
            }
        ],
    )

    output_dir = tmp_path / "tables"
    written = paper_tables.generate_tables(
        results_dir=results_dir, output_dir=output_dir
    )

    assert set(written) == {
        "showcase_runtime",
        "showcase_memory",
        "partitioning_summary",
        "w_state_oracle",
    }
    runtime_table = output_dir / "showcase_runtime.tex"
    runtime_content = runtime_table.read_text()
    assert "\\begin{tabular}" in runtime_content
    assert "Optimal backend" in runtime_content
    assert "\\times" in runtime_content

    partitioning_table = output_dir / "partitioning_summary.tex"
    partition_content = partitioning_table.read_text()
    assert "Staged Rank" in partition_content
    assert "speedup" in partition_content.lower()

    oracle_table = output_dir / "w_state_oracle.tex"
    oracle_content = oracle_table.read_text()
    assert "\\{" in oracle_content

    memory_table = output_dir / "showcase_memory.tex"
    memory_content = memory_table.read_text()
    assert "Minimum-memory backend" in memory_content
    assert "\\mathrm" in memory_content
