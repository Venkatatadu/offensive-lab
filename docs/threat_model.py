"""
Space Systems Threat Model & ATT&CK Mapping
=============================================
Maps offensive techniques to the space system kill chain,
combining MITRE ATT&CK with space-specific threat categories
from NIST IR 8401 and CISA Space Security guidance.

This module provides:
- Space-specific threat taxonomy
- Kill chain for satellite system compromise
- Mapping of tools to threat categories
- Red team engagement planning templates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import json


class SpaceKillChainPhase(Enum):
    """Space system compromise kill chain phases."""
    RECONNAISSANCE = "Reconnaissance"
    RF_ACCESS = "RF Access & Interception"
    GROUND_SEGMENT_COMPROMISE = "Ground Segment Compromise"
    LINK_LAYER_ATTACK = "Link Layer Attack"
    COMMAND_INJECTION = "Command Injection"
    PAYLOAD_MANIPULATION = "Payload/Data Manipulation"
    PERSISTENCE = "Persistence & Stealth"
    MISSION_IMPACT = "Mission Impact"


class ThreatCategory(Enum):
    """NIST IR 8401 space threat categories."""
    SPOOFING = "Spoofing"
    JAMMING = "Jamming"
    EAVESDROPPING = "Eavesdropping"
    HIJACKING = "Hijacking"
    DATA_CORRUPTION = "Data Corruption"
    SUPPLY_CHAIN = "Supply Chain"
    INSIDER = "Insider Threat"
    CYBER_PHYSICAL = "Cyber-Physical"
    AI_ML = "AI/ML Exploitation"


@dataclass
class SpaceThreat:
    """A space-specific threat with full context."""
    id: str
    name: str
    phase: SpaceKillChainPhase
    category: ThreatCategory
    description: str
    mitre_techniques: list[str]
    affected_segments: list[str]  # space, ground, link, user
    lab_tool: Optional[str] = None
    difficulty: str = "medium"  # low, medium, high, nation-state
    detection: str = "medium"  # easy, medium, hard
    impact: str = "high"  # low, medium, high, catastrophic
    prerequisites: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# COMPREHENSIVE SPACE THREAT CATALOG
# ═══════════════════════════════════════════════════════════════════

THREAT_CATALOG: list[SpaceThreat] = [
    # ── RECONNAISSANCE ──
    SpaceThreat(
        id="ST-001",
        name="TLE-Based Orbital Prediction",
        phase=SpaceKillChainPhase.RECONNAISSANCE,
        category=ThreatCategory.EAVESDROPPING,
        description="Use publicly available Two-Line Element sets to predict "
                    "satellite positions and plan signal interception windows.",
        mitre_techniques=["T1595 — Active Scanning", "T1596 — Search Open Technical Databases"],
        affected_segments=["link"],
        lab_tool="orbital_recon",
        difficulty="low",
        detection="hard",
        prerequisites=["Internet access", "CelesTrak/Space-Track account"],
    ),
    SpaceThreat(
        id="ST-002",
        name="Ground Station RF Enumeration",
        phase=SpaceKillChainPhase.RECONNAISSANCE,
        category=ThreatCategory.EAVESDROPPING,
        description="Map ground station locations, antenna types, and operating "
                    "frequencies via OSINT, satellite imagery, and ITU filings.",
        mitre_techniques=["T1592 — Gather Victim Host Information"],
        affected_segments=["ground"],
        lab_tool="orbital_recon",
        difficulty="low",
        detection="hard",
        prerequisites=["OSINT tools", "Google Earth access"],
    ),
    SpaceThreat(
        id="ST-003",
        name="ML Model Fingerprinting",
        phase=SpaceKillChainPhase.RECONNAISSANCE,
        category=ThreatCategory.AI_ML,
        description="Query cloud-based satellite imagery classification APIs "
                    "to fingerprint the underlying model architecture and "
                    "training data characteristics.",
        mitre_techniques=["T1592.004 — Client Configurations"],
        affected_segments=["ground", "user"],
        lab_tool="ai_adversarial (model extraction)",
        difficulty="medium",
        detection="medium",
    ),

    # ── RF ACCESS ──
    SpaceThreat(
        id="ST-010",
        name="Satellite Downlink Interception",
        phase=SpaceKillChainPhase.RF_ACCESS,
        category=ThreatCategory.EAVESDROPPING,
        description="Receive and demodulate satellite telemetry using consumer "
                    "SDR hardware (RTL-SDR, HackRF). Many LEO satellites transmit "
                    "unencrypted telemetry in UHF/S-band.",
        mitre_techniques=["T1040 — Network Sniffing"],
        affected_segments=["link"],
        lab_tool="satcom_fuzzer (rf_sigint)",
        difficulty="low",
        detection="hard",
        impact="medium",
        prerequisites=["SDR hardware ($25-300)", "Antenna", "GNU Radio"],
    ),
    SpaceThreat(
        id="ST-011",
        name="GNSS Spoofing",
        phase=SpaceKillChainPhase.RF_ACCESS,
        category=ThreatCategory.SPOOFING,
        description="Generate fake GPS/GNSS signals to manipulate the position "
                    "reported by a receiver. Can affect spacecraft navigation, "
                    "drone operations, and time synchronization.",
        mitre_techniques=["T1557 — Adversary-in-the-Middle"],
        affected_segments=["link", "space", "user"],
        lab_tool="satcom_fuzzer (gnss_spoofing)",
        difficulty="medium",
        detection="medium",
        impact="catastrophic",
        prerequisites=["SDR with TX capability", "GNSS signal knowledge"],
        references=["CISA GPS Spoofing Advisory", "Humphreys et al. (UT Austin)"],
    ),
    SpaceThreat(
        id="ST-012",
        name="Uplink Jamming",
        phase=SpaceKillChainPhase.RF_ACCESS,
        category=ThreatCategory.JAMMING,
        description="Transmit interference on the satellite's command uplink "
                    "frequency to prevent ground operators from sending commands.",
        mitre_techniques=["T1499 — Endpoint Denial of Service"],
        affected_segments=["link"],
        difficulty="medium",
        detection="easy",
        impact="high",
        prerequisites=["High-power transmitter", "Directional antenna", "Frequency knowledge"],
    ),

    # ── GROUND SEGMENT ──
    SpaceThreat(
        id="ST-020",
        name="Ground Station Software Exploitation",
        phase=SpaceKillChainPhase.GROUND_SEGMENT_COMPROMISE,
        category=ThreatCategory.HIJACKING,
        description="Exploit vulnerabilities in ground station command & control "
                    "software. Parsing bugs in CCSDS packet handlers are common — "
                    "many are written in C/C++ with minimal bounds checking.",
        mitre_techniques=["T1190 — Exploit Public-Facing Application"],
        affected_segments=["ground"],
        lab_tool="satcom_fuzzer",
        difficulty="medium",
        detection="medium",
        impact="catastrophic",
    ),
    SpaceThreat(
        id="ST-021",
        name="AI/ML Pipeline Poisoning (Ground)",
        phase=SpaceKillChainPhase.GROUND_SEGMENT_COMPROMISE,
        category=ThreatCategory.AI_ML,
        description="Poison training data or models used in ground-based "
                    "satellite imagery processing, anomaly detection, or "
                    "autonomous mission planning systems.",
        mitre_techniques=["T1195.003 — Supply Chain Compromise: Software"],
        affected_segments=["ground"],
        lab_tool="ai_adversarial (poisoning + supply_chain)",
        difficulty="high",
        detection="hard",
        impact="high",
    ),
    SpaceThreat(
        id="ST-022",
        name="LLM-Augmented Ops Prompt Injection",
        phase=SpaceKillChainPhase.GROUND_SEGMENT_COMPROMISE,
        category=ThreatCategory.AI_ML,
        description="Inject malicious instructions into LLM-augmented satellite "
                    "operations systems via telemetry data, operator manuals, "
                    "or metadata fields processed by RAG pipelines.",
        mitre_techniques=["T1059 — Command and Scripting Interpreter"],
        affected_segments=["ground"],
        lab_tool="ai_adversarial (prompt_inject)",
        difficulty="medium",
        detection="hard",
        impact="high",
        references=["OWASP LLM01", "Advisory GHSA-mcww-4hxq-hfr3"],
    ),

    # ── LINK LAYER ──
    SpaceThreat(
        id="ST-030",
        name="TC Frame Replay Attack",
        phase=SpaceKillChainPhase.LINK_LAYER_ATTACK,
        category=ThreatCategory.HIJACKING,
        description="Capture and replay valid TC Transfer Frames. BD-mode frames "
                    "bypass sequence validation (FARM), making replay trivial.",
        mitre_techniques=["T1557 — Adversary-in-the-Middle"],
        affected_segments=["link"],
        lab_tool="space_protocol_analyzer",
        difficulty="medium",
        detection="medium",
        impact="high",
        prerequisites=["Captured valid TC frame", "Uplink capability"],
    ),
    SpaceThreat(
        id="ST-031",
        name="AOS Insert Zone Injection",
        phase=SpaceKillChainPhase.LINK_LAYER_ATTACK,
        category=ThreatCategory.DATA_CORRUPTION,
        description="Inject data into AOS Transfer Frame Insert Zone. The Insert "
                    "Zone is typically unauthenticated and can carry arbitrary data "
                    "that may be processed by ground systems.",
        mitre_techniques=["T1565 — Data Manipulation"],
        affected_segments=["link"],
        lab_tool="space_protocol_analyzer",
        difficulty="high",
        detection="hard",
        impact="medium",
    ),

    # ── COMMAND INJECTION ──
    SpaceThreat(
        id="ST-040",
        name="Unauthorized Telecommand Execution",
        phase=SpaceKillChainPhase.COMMAND_INJECTION,
        category=ThreatCategory.HIJACKING,
        description="Send crafted CCSDS telecommand packets to the spacecraft. "
                    "Without encryption and authentication, any entity with "
                    "uplink capability can command the satellite.",
        mitre_techniques=["T1059 — Command and Scripting Interpreter"],
        affected_segments=["space", "link"],
        lab_tool="satcom_fuzzer",
        difficulty="high",
        detection="easy",
        impact="catastrophic",
        prerequisites=["Uplink capability", "Frequency/protocol knowledge"],
    ),
    SpaceThreat(
        id="ST-041",
        name="PUS Memory Management Abuse",
        phase=SpaceKillChainPhase.COMMAND_INJECTION,
        category=ThreatCategory.HIJACKING,
        description="Abuse PUS Service 6 (Memory Management) to read/write "
                    "arbitrary on-board memory. Can modify flight software, "
                    "extract cryptographic keys, or disable safety checks.",
        mitre_techniques=["T1055 — Process Injection"],
        affected_segments=["space"],
        lab_tool="space_protocol_analyzer",
        difficulty="high",
        detection="medium",
        impact="catastrophic",
    ),

    # ── AI/ML SPECIFIC ──
    SpaceThreat(
        id="ST-050",
        name="Adversarial Evasion of Satellite Imagery Classifier",
        phase=SpaceKillChainPhase.PAYLOAD_MANIPULATION,
        category=ThreatCategory.AI_ML,
        description="Apply adversarial perturbations to ground-level scenes to "
                    "evade detection by overhead satellite imagery classifiers. "
                    "Physical-world adversarial patches can fool military asset "
                    "detection systems.",
        mitre_techniques=["T1036 — Masquerading"],
        affected_segments=["space", "ground", "user"],
        lab_tool="ai_adversarial (evasion)",
        difficulty="medium",
        detection="hard",
        impact="high",
        references=["Brown et al. 2017 (Adversarial Patches)"],
    ),
    SpaceThreat(
        id="ST-051",
        name="Model Supply Chain Attack (Pickle RCE)",
        phase=SpaceKillChainPhase.GROUND_SEGMENT_COMPROMISE,
        category=ThreatCategory.SUPPLY_CHAIN,
        description="Distribute trojaned ML model files (.pkl, .pt) via model "
                    "registries. Pickle deserialization achieves RCE on the "
                    "ground station when the model is loaded.",
        mitre_techniques=["T1195 — Supply Chain Compromise"],
        affected_segments=["ground"],
        lab_tool="ai_adversarial (supply_chain)",
        difficulty="medium",
        detection="easy",
        impact="catastrophic",
        references=["Trail of Bits ML Security", "HuggingFace scanning"],
    ),
    SpaceThreat(
        id="ST-052",
        name="Federated Learning Poisoning (Constellation)",
        phase=SpaceKillChainPhase.PAYLOAD_MANIPULATION,
        category=ThreatCategory.AI_ML,
        description="In satellite constellations using federated learning for "
                    "on-orbit model updates, a compromised node can poison the "
                    "global model via gradient manipulation.",
        mitre_techniques=["T1565.001 — Stored Data Manipulation"],
        affected_segments=["space"],
        lab_tool="ai_adversarial (poisoning)",
        difficulty="high",
        detection="hard",
        impact="high",
    ),
]


# ═══════════════════════════════════════════════════════════════════
# RED TEAM ENGAGEMENT PLANNER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EngagementScenario:
    """A red team engagement scenario for space systems."""
    name: str
    objective: str
    scope: list[str]
    phases: list[dict]
    tools_required: list[str]
    estimated_duration: str
    risk_level: str


ENGAGEMENT_TEMPLATES: list[EngagementScenario] = [
    EngagementScenario(
        name="Ground Station Resilience Assessment",
        objective="Test ground station software's resilience to malformed space protocol traffic",
        scope=["Ground station C2 software", "CCSDS packet parsers", "Mission control interfaces"],
        phases=[
            {"phase": 1, "name": "Protocol Enumeration", "tool": "space_protocol_analyzer", "duration": "2 days"},
            {"phase": 2, "name": "Fuzzing Campaign", "tool": "satcom_fuzzer", "duration": "5 days"},
            {"phase": 3, "name": "Crash Triage & Exploitation", "tool": "manual", "duration": "3 days"},
            {"phase": 4, "name": "Reporting", "tool": "N/A", "duration": "2 days"},
        ],
        tools_required=["satcom_fuzzer", "space_protocol_analyzer"],
        estimated_duration="2 weeks",
        risk_level="LOW (isolated test environment)",
    ),
    EngagementScenario(
        name="AI/ML Pipeline Security Assessment",
        objective="Assess security of ML models and pipelines used in satellite data processing",
        scope=["Model training pipelines", "Model registries", "Inference APIs", "RAG/LLM systems"],
        phases=[
            {"phase": 1, "name": "Model Supply Chain Audit", "tool": "supply_chain analyzer", "duration": "2 days"},
            {"phase": 2, "name": "Adversarial Robustness Testing", "tool": "ai_adversarial (evasion)", "duration": "3 days"},
            {"phase": 3, "name": "Prompt Injection Testing", "tool": "ai_adversarial (prompt_inject)", "duration": "2 days"},
            {"phase": 4, "name": "Data Poisoning Feasibility", "tool": "ai_adversarial (poison)", "duration": "2 days"},
            {"phase": 5, "name": "Model Extraction Attempt", "tool": "ai_adversarial (extraction)", "duration": "3 days"},
            {"phase": 6, "name": "Reporting", "tool": "N/A", "duration": "2 days"},
        ],
        tools_required=["ai_adversarial", "supply_chain"],
        estimated_duration="2 weeks",
        risk_level="LOW-MEDIUM",
    ),
    EngagementScenario(
        name="Full Satellite System Red Team",
        objective="End-to-end assessment of satellite system security across all segments",
        scope=["Ground segment", "Link segment (simulated)", "User segment", "AI/ML components"],
        phases=[
            {"phase": 1, "name": "OSINT & Orbital Recon", "tool": "orbital_recon", "duration": "3 days"},
            {"phase": 2, "name": "Ground Station Assessment", "tool": "satcom_fuzzer + analyzer", "duration": "5 days"},
            {"phase": 3, "name": "Protocol Vulnerability Analysis", "tool": "space_protocol_analyzer", "duration": "3 days"},
            {"phase": 4, "name": "AI/ML Attack Surface", "tool": "ai_adversarial suite", "duration": "5 days"},
            {"phase": 5, "name": "RF Attack Simulation", "tool": "rf_sigint", "duration": "3 days"},
            {"phase": 6, "name": "Comprehensive Reporting", "tool": "N/A", "duration": "3 days"},
        ],
        tools_required=["orbital_recon", "satcom_fuzzer", "space_protocol_analyzer", "ai_adversarial", "rf_sigint"],
        estimated_duration="3-4 weeks",
        risk_level="MEDIUM",
    ),
]


def generate_threat_matrix() -> dict:
    """Generate a threat matrix summary."""
    matrix = {}
    for threat in THREAT_CATALOG:
        phase = threat.phase.value
        if phase not in matrix:
            matrix[phase] = []
        matrix[phase].append({
            "id": threat.id,
            "name": threat.name,
            "category": threat.category.value,
            "difficulty": threat.difficulty,
            "impact": threat.impact,
            "detection": threat.detection,
            "tool": threat.lab_tool,
        })
    return matrix


def export_catalog(output_path: str):
    """Export full threat catalog to JSON."""
    data = {
        "metadata": {
            "title": "Space Systems Threat Catalog",
            "version": "1.0",
            "framework": "MITRE ATT&CK + NIST IR 8401",
            "author": "Offensive Lab",
        },
        "threats": [
            {
                "id": t.id,
                "name": t.name,
                "phase": t.phase.value,
                "category": t.category.value,
                "description": t.description,
                "mitre_techniques": t.mitre_techniques,
                "affected_segments": t.affected_segments,
                "lab_tool": t.lab_tool,
                "difficulty": t.difficulty,
                "detection": t.detection,
                "impact": t.impact,
                "prerequisites": t.prerequisites,
                "references": t.references,
            }
            for t in THREAT_CATALOG
        ],
        "engagement_templates": [
            {
                "name": e.name,
                "objective": e.objective,
                "scope": e.scope,
                "phases": e.phases,
                "tools_required": e.tools_required,
                "estimated_duration": e.estimated_duration,
                "risk_level": e.risk_level,
            }
            for e in ENGAGEMENT_TEMPLATES
        ],
    }

    from pathlib import Path
    Path(output_path).write_text(json.dumps(data, indent=2))
    return data
