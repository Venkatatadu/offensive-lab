# Contributing to Offensive Lab

## Code of Conduct

This project produces **offensive security research tools**. All contributions must:

1. Be designed for **authorized testing only**
2. Include appropriate warnings and ethical usage notes
3. Not include weaponized exploits targeting specific production systems
4. Follow responsible disclosure for any real vulnerabilities discovered

## How to Contribute

### Adding a New Tool

1. Create a new directory under the project root
2. Include `__init__.py`, `__main__.py` (CLI), and core modules
3. Add Rich-based CLI output (consistent with existing tools)
4. Add entry to the main `README.md`
5. Map your tool's capabilities to the threat model in `docs/threat_model.py`

### Adding Attack Techniques

1. Document the technique with academic/industry references
2. Include MITRE ATT&CK and CWE mappings where applicable
3. Provide both the attack implementation and detection guidance
4. Add to the appropriate tool module

### Code Style

- Python 3.10+ with type hints
- Dataclasses over plain dicts for structured data
- Rich for CLI output
- Click for CLI argument parsing
- Docstrings on all public classes and functions

## Areas We're Looking For

- **Space protocol implementations** — CCSDS extensions, DTN (Delay-Tolerant Networking), SpaceWire
- **On-orbit AI attacks** — adversarial attacks specific to edge ML on satellites
- **Quantum-safe crypto analysis** — post-quantum implications for space link security
- **Digital twin attacks** — compromising satellite digital twins used in mission planning
- **Autonomous system attacks** — targeting AI-driven spacecraft autonomy

## License

All contributions are under MIT license.
