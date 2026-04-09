"""
Space Link Protocol Analyzer — Vulnerability Detection
========================================================
Deep packet inspection and security analysis for space
communication protocols. Identifies authentication gaps,
encryption weaknesses, and command injection vectors.

Supports:
- CCSDS Space Packet Protocol (133.0-B-2)
- TC Space Data Link Protocol (232.0-B-4)
- AOS Transfer Frames (732.0-B-4)
- Proximity-1 (211.0-B-5)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

console = Console()


class VulnSeverity(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()


@dataclass
class SecurityFinding:
    id: str
    severity: VulnSeverity
    title: str
    description: str
    affected_field: str
    offset: int
    recommendation: str
    cwe: Optional[str] = None
    mitre: Optional[str] = None


class ProtocolAnalyzer:
    """Analyze space protocol packets for security vulnerabilities."""

    def __init__(self):
        self.findings: list[SecurityFinding] = []

    def analyze_ccsds_packet(self, data: bytes) -> list[SecurityFinding]:
        """Analyze a CCSDS space packet for vulnerabilities."""
        findings = []

        if len(data) < 6:
            findings.append(SecurityFinding(
                id="CCSDS-001", severity=VulnSeverity.HIGH,
                title="Truncated packet header",
                description="Packet shorter than minimum 6-byte CCSDS primary header. "
                            "May trigger buffer under-read in parsers.",
                affected_field="primary_header", offset=0,
                recommendation="Validate minimum packet length before parsing.",
                cwe="CWE-125: Out-of-bounds Read",
            ))
            return findings

        word1, word2, data_length = struct.unpack(">HHH", data[:6])
        version = (word1 >> 13) & 0x07
        pkt_type = (word1 >> 12) & 0x01
        sec_hdr = (word1 >> 11) & 0x01
        apid = word1 & 0x7FF
        seq_flags = (word2 >> 14) & 0x03
        seq_count = word2 & 0x3FFF

        # Version check
        if version != 0:
            findings.append(SecurityFinding(
                id="CCSDS-002", severity=VulnSeverity.MEDIUM,
                title=f"Non-standard version number ({version})",
                description="CCSDS version should be 000. Non-standard versions may "
                            "indicate crafted packets or protocol confusion attacks.",
                affected_field="version", offset=0,
                recommendation="Reject packets with version != 0.",
                cwe="CWE-20: Improper Input Validation",
            ))

        # Length consistency
        expected_total = data_length + 7  # header(6) + data_length + 1
        if expected_total != len(data):
            findings.append(SecurityFinding(
                id="CCSDS-003", severity=VulnSeverity.HIGH,
                title="Data length field inconsistency",
                description=f"Header claims {data_length + 1} data bytes (total {expected_total}), "
                            f"but actual packet is {len(data)} bytes. Length mismatches can cause "
                            f"buffer overflows or information leaks.",
                affected_field="data_length", offset=4,
                recommendation="Validate data_length against actual packet size before processing.",
                cwe="CWE-131: Incorrect Calculation of Buffer Size",
            ))

        # APID analysis
        if apid == 0x7FF:
            findings.append(SecurityFinding(
                id="CCSDS-004", severity=VulnSeverity.INFO,
                title="Idle APID (0x7FF) detected",
                description="APID 0x7FF is reserved for idle packets. These should be "
                            "discarded but some implementations process them.",
                affected_field="apid", offset=0,
                recommendation="Ensure idle packets are properly discarded.",
            ))
        elif apid == 0:
            findings.append(SecurityFinding(
                id="CCSDS-005", severity=VulnSeverity.LOW,
                title="APID 0 — potential default/unconfigured endpoint",
                description="APID 0 may indicate a default configuration that should "
                            "not be accepting commands in production.",
                affected_field="apid", offset=0,
                recommendation="Audit APID 0 handler for unintended functionality.",
            ))

        # Sequence flag analysis
        if seq_flags == 0b00:  # continuation
            findings.append(SecurityFinding(
                id="CCSDS-006", severity=VulnSeverity.MEDIUM,
                title="Continuation segment without preceding first segment",
                description="Standalone continuation segment. Reassembly logic may be "
                            "vulnerable to state confusion if segments arrive out of order.",
                affected_field="sequence_flags", offset=2,
                recommendation="Implement strict segment reassembly state machine.",
                cwe="CWE-362: Race Condition",
            ))

        # Check for no secondary header on telecommand
        if pkt_type == 1 and not sec_hdr:
            findings.append(SecurityFinding(
                id="CCSDS-007", severity=VulnSeverity.HIGH,
                title="Telecommand without secondary header",
                description="TC packet lacks secondary header (PUS). This means no "
                            "service type validation, acknowledgement tracking, or "
                            "source authentication is possible.",
                affected_field="sec_header_flag", offset=0,
                recommendation="Require PUS secondary header on all telecommands.",
                mitre="T1599.001 — Network Boundary Bridging",
            ))

        # PUS analysis (if secondary header present)
        if sec_hdr and len(data) >= 10 and pkt_type == 1:
            pus_byte = data[6]
            pus_version = (pus_byte >> 4) & 0x0F
            ack_flags = pus_byte & 0x0F
            svc_type = data[7]
            svc_subtype = data[8]

            # Dangerous service types
            dangerous_services = {
                6: "Memory Management — can read/write arbitrary memory",
                8: "Function Management — can invoke arbitrary on-board functions",
                11: "Time-based Scheduling — can schedule deferred commands",
                19: "Event-Action — can set up autonomous trigger-response pairs",
            }
            if svc_type in dangerous_services:
                findings.append(SecurityFinding(
                    id="CCSDS-008", severity=VulnSeverity.CRITICAL,
                    title=f"Dangerous PUS service: {svc_type} ({dangerous_services[svc_type].split(' — ')[0]})",
                    description=dangerous_services[svc_type],
                    affected_field="pus_service_type", offset=7,
                    recommendation="Restrict access to dangerous services via APID-based ACLs.",
                    mitre="T1059 — Command and Scripting Interpreter",
                ))

            if ack_flags == 0:
                findings.append(SecurityFinding(
                    id="CCSDS-009", severity=VulnSeverity.MEDIUM,
                    title="No acknowledgement requested on TC",
                    description="Command requests no acknowledgement. Operator has no "
                                "confirmation of execution. Could be used to inject "
                                "commands stealthily.",
                    affected_field="ack_flags", offset=6,
                    recommendation="Require at least completion acknowledgement on all TCs.",
                ))

        self.findings.extend(findings)
        return findings

    def analyze_tc_frame(self, data: bytes) -> list[SecurityFinding]:
        """Analyze a TC Transfer Frame for vulnerabilities."""
        findings = []

        if len(data) < 5:
            findings.append(SecurityFinding(
                id="TCTF-001", severity=VulnSeverity.HIGH,
                title="Truncated TC Transfer Frame",
                description="Frame shorter than minimum header.",
                affected_field="frame_header", offset=0,
                recommendation="Validate minimum frame length.",
                cwe="CWE-125: Out-of-bounds Read",
            ))
            return findings

        word1, = struct.unpack(">H", data[:2])
        bypass = (word1 >> 13) & 0x01

        if bypass:
            findings.append(SecurityFinding(
                id="TCTF-002", severity=VulnSeverity.CRITICAL,
                title="BD (Bypass) mode — no FARM sequence check",
                description="Frame uses BD mode, bypassing the Frame Acceptance and "
                            "Reporting Mechanism. This means no sequence validation, "
                            "making replay attacks trivial.",
                affected_field="bypass_flag", offset=0,
                recommendation="Use AD mode with FARM for command sequence validation.",
                mitre="T1499.003 — Application Exhaustion Flood",
            ))

        # CRC check
        if len(data) >= 4:
            frame_data = data[:-2]
            declared_crc = struct.unpack(">H", data[-2:])[0]
            computed_crc = self._crc16_ccitt(frame_data)
            if declared_crc != computed_crc:
                findings.append(SecurityFinding(
                    id="TCTF-003", severity=VulnSeverity.LOW,
                    title="Frame CRC mismatch",
                    description=f"FECF mismatch: declared=0x{declared_crc:04X}, "
                                f"computed=0x{computed_crc:04X}. May indicate corruption or tampering.",
                    affected_field="fecf", offset=len(data) - 2,
                    recommendation="Reject frames with CRC errors.",
                ))

        self.findings.extend(findings)
        return findings

    def generate_report(self) -> dict:
        """Generate a summary security report."""
        severity_counts = {}
        for f in self.findings:
            severity_counts[f.severity.name] = severity_counts.get(f.severity.name, 0) + 1

        return {
            "total_findings": len(self.findings),
            "by_severity": severity_counts,
            "findings": [
                {
                    "id": f.id,
                    "severity": f.severity.name,
                    "title": f.title,
                    "description": f.description,
                    "affected_field": f.affected_field,
                    "offset": f.offset,
                    "recommendation": f.recommendation,
                    "cwe": f.cwe,
                    "mitre": f.mitre,
                }
                for f in self.findings
            ],
        }

    @staticmethod
    def _crc16_ccitt(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
                crc &= 0xFFFF
        return crc


@click.command()
@click.option("--demo", is_flag=True, help="Run demo analysis on sample packets")
@click.option("--packet-hex", default=None, help="Analyze a hex-encoded packet")
@click.option("--output-json", default=None, help="Export findings to JSON")
def main(demo, packet_hex, output_json):
    """Space Protocol Analyzer — Vulnerability detection for space link protocols."""
    console.print(Panel.fit(
        "[bold magenta]SPACE PROTOCOL ANALYZER[/bold magenta]\n"
        "[dim]Vulnerability detection for CCSDS/AOS/Proximity-1[/dim]",
        border_style="magenta",
    ))

    analyzer = ProtocolAnalyzer()

    if demo:
        from satcom_fuzzer.ccsds import build_tc_packet, TCTransferFrame

        console.print("\n[yellow]Analyzing sample packets...[/yellow]\n")

        # 1. Valid TC packet with dangerous service
        pkt1 = build_tc_packet(apid=1, service_type=6, service_subtype=5,
                                payload=struct.pack(">IH", 0x08000000, 256))
        f1 = analyzer.analyze_ccsds_packet(pkt1)
        console.print(f"[cyan]Packet 1:[/cyan] Memory Management TC ({len(pkt1)} bytes) → {len(f1)} findings")

        # 2. TC without secondary header
        pkt2 = bytes.fromhex("08010000000401020304")
        f2 = analyzer.analyze_ccsds_packet(pkt2)
        console.print(f"[cyan]Packet 2:[/cyan] TC without PUS header ({len(pkt2)} bytes) → {len(f2)} findings")

        # 3. Length mismatch
        pkt3 = bytes.fromhex("18010000FFFF0102")
        f3 = analyzer.analyze_ccsds_packet(pkt3)
        console.print(f"[cyan]Packet 3:[/cyan] Length overflow ({len(pkt3)} bytes) → {len(f3)} findings")

        # 4. BD-mode TC frame
        frame = TCTransferFrame(bypass_flag=True, spacecraft_id=42, data=pkt1)
        f4 = analyzer.analyze_tc_frame(frame.pack())
        console.print(f"[cyan]Frame 1:[/cyan] BD-mode TC Frame → {len(f4)} findings")

    elif packet_hex:
        pkt = bytes.fromhex(packet_hex)
        findings = analyzer.analyze_ccsds_packet(pkt)
        console.print(f"\n[cyan]Analyzed {len(pkt)} bytes → {len(findings)} findings[/cyan]")

    # Display findings
    if analyzer.findings:
        console.print()
        table = Table(title="Security Findings")
        table.add_column("ID", style="cyan")
        table.add_column("Severity")
        table.add_column("Title")
        table.add_column("Field")
        table.add_column("CWE/MITRE", max_width=30)

        for f in analyzer.findings:
            sev_color = {
                "CRITICAL": "bold red", "HIGH": "red",
                "MEDIUM": "yellow", "LOW": "blue", "INFO": "dim"
            }.get(f.severity.name, "white")
            ref = f.cwe or f.mitre or "-"
            table.add_row(f.id, f"[{sev_color}]{f.severity.name}[/{sev_color}]",
                          f.title, f.affected_field, ref)
        console.print(table)

        # Summary
        report = analyzer.generate_report()
        console.print(f"\n[bold]Summary:[/bold] {report['total_findings']} findings")
        for sev, count in report["by_severity"].items():
            console.print(f"  {sev}: {count}")

    if output_json:
        import json
        from pathlib import Path
        Path(output_json).write_text(json.dumps(analyzer.generate_report(), indent=2))
        console.print(f"\n[green]Report saved to {output_json}[/green]")


if __name__ == "__main__":
    main()
