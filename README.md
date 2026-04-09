# Offensive Lab

> Offensive security research at the edge — satellite protocols, adversarial AI, and novel attack surfaces.

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Overview

An open-source collection of offensive security tools targeting underexplored attack surfaces — satellite communication protocols, AI/ML systems, and the intersection of autonomous systems with space infrastructure. Built for authorized security research, red teaming, and vulnerability discovery.

---

## Tools

### `satcom_fuzzer` — CCSDS Space Protocol Fuzzer
AI-guided protocol fuzzer for CCSDS telecommand and telemetry packets. Uses mutation strategies informed by ML to maximize code coverage and discover parsing vulnerabilities in ground station software.

- CCSDS TC/TM packet generation & mutation
- AI-guided fuzzing with coverage feedback loop
- Supports AOS, Proximity-1, and custom frame types
- Crash triage and deduplication
- Export to PCAP for Wireshark analysis

### `orbital_recon` — Satellite OSINT & Attack Surface Mapper
Passive reconnaissance framework for satellite infrastructure. Pulls public TLE data, calculates orbital passes, enumerates known ground station locations, and maps RF attack surfaces.

- TLE fetching & orbital propagation (SGP4)
- Ground station geolocation database
- RF frequency band enumeration per satellite
- Pass prediction for signal interception windows
- Link budget calculator for SDR intercept feasibility

### `ai_adversarial` — Adversarial ML Attack Framework
Toolkit for attacking ML models deployed in autonomous and space systems — satellite imagery classifiers, anomaly detectors, and AI-driven C2 systems.

- Evasion attacks (FGSM, PGD, C&W) against image classifiers
- Data poisoning payload generator
- Model extraction via query API
- Prompt injection payloads for LLM-augmented systems
- Membership inference attacks

### `space_protocol_analyzer` — Space Link Protocol Dissector
Deep packet inspection and vulnerability analysis for space communication protocols.

- CCSDS Space Packet Protocol parser
- Proximity-1 frame dissector
- AOS frame analysis
- Authentication bypass detection
- Command injection vector mapping

### `ai_adversarial/supply_chain` — ML Supply Chain Attacker
Offensive toolkit targeting the ML model distribution pipeline — pickle RCE, ONNX graph injection, safetensors fuzzing, and model registry typosquatting.

- Pickle deserialization RCE payload generator
- ONNX graph backdoor injection
- Safetensors header fuzzing
- Model file static analyzer
- HuggingFace Hub typosquatting templates

### `satcom_fuzzer/rf_sigint` — RF Signal Intelligence & SDR Simulator
Simulates software-defined radio attacks against satellite links.

- BPSK/QPSK IQ sample generation
- Jamming effectiveness calculator (J/S ratio)
- GNSS spoofing parameter generator
- RF fingerprinting for transmitter authentication/spoofing
- Multiple jamming modes: barrage, spot, sweep, pulse

---

## Quick Start

```bash
git clone https://github.com/Venkatatadu/offensive-lab.git
cd offensive-lab
pip install -r requirements.txt

# Run the CCSDS fuzzer
python -m satcom_fuzzer --target localhost:9999 --mode ai-guided --frames 10000

# Satellite recon
python -m orbital_recon --sat "ISS" --passes 5 --location "46.1,-64.8"

# Adversarial ML
python -m ai_adversarial --attack fgsm --model target_model.onnx --input sample.png

# Protocol analysis
python -m space_protocol_analyzer --pcap capture.pcap --detect-vulns
```

---

## Legal

These tools are for **authorized security research only**. Usage against systems without explicit written authorization is illegal and unethical. All satellite data used comes from publicly available sources (CelesTrak, Space-Track.org).

## Contributing

PRs welcome. See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
