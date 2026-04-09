"""
Orbital Recon — Satellite OSINT & Attack Surface Mapper
========================================================
Passive reconnaissance framework for satellite infrastructure.
Uses publicly available TLE data to calculate orbital passes,
enumerate ground station coverage, and assess RF intercept feasibility.

Data sources:
- CelesTrak (celestrak.org) — public TLE catalogs
- SatNOGS DB — ground station network
- ITU frequency filings — RF allocations

Usage:
    python -m orbital_recon --sat "ISS" --passes 5 --location "46.1,-64.8"
    python -m orbital_recon --catalog starlink --ground-stations
    python -m orbital_recon --apid-scan --sat-id 25544
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    from sgp4.api import Satrec, jday
    from sgp4 import exporter
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

console = Console()

# ── Known Ground Station Database ────────────────────────────────────

GROUND_STATIONS = [
    {"name": "ESTRACK Kourou", "lat": 5.236, "lon": -52.769, "operator": "ESA", "bands": ["S", "X"]},
    {"name": "DSN Goldstone", "lat": 35.427, "lon": -116.890, "operator": "NASA/JPL", "bands": ["S", "X", "Ka"]},
    {"name": "DSN Canberra", "lat": -35.402, "lon": 148.981, "operator": "NASA/JPL", "bands": ["S", "X", "Ka"]},
    {"name": "DSN Madrid", "lat": 40.431, "lon": -4.249, "operator": "NASA/JPL", "bands": ["S", "X", "Ka"]},
    {"name": "ESTRACK Redu", "lat": 50.002, "lon": 5.146, "operator": "ESA", "bands": ["S", "X"]},
    {"name": "ESTRACK Cebreros", "lat": 40.453, "lon": -4.368, "operator": "ESA", "bands": ["S", "X", "Ka"]},
    {"name": "ESTRACK New Norcia", "lat": -31.048, "lon": 116.192, "operator": "ESA", "bands": ["S", "X", "Ka"]},
    {"name": "Svalbard SvalSat", "lat": 78.230, "lon": 15.408, "operator": "KSAT", "bands": ["S", "X", "Ka"]},
    {"name": "KSAT TrollSat", "lat": -72.012, "lon": 2.535, "operator": "KSAT", "bands": ["S", "X"]},
    {"name": "ISRO ISTRAC Bangalore", "lat": 12.950, "lon": 77.510, "operator": "ISRO", "bands": ["S", "C"]},
    {"name": "Roscosmos Baikonur", "lat": 45.965, "lon": 63.305, "operator": "Roscosmos", "bands": ["S", "X"]},
    {"name": "JAXA Usuda", "lat": 36.133, "lon": 138.362, "operator": "JAXA", "bands": ["S", "X", "Ka"]},
    {"name": "CNES Toulouse", "lat": 43.428, "lon": 1.498, "operator": "CNES", "bands": ["S", "X"]},
    {"name": "SSC Esrange", "lat": 67.893, "lon": 21.104, "operator": "SSC", "bands": ["S", "X"]},
    {"name": "AWS Ground Station US-East", "lat": 39.006, "lon": -77.459, "operator": "Amazon", "bands": ["S", "X", "UHF"]},
    {"name": "AWS Ground Station US-West", "lat": 46.906, "lon": -122.759, "operator": "Amazon", "bands": ["S", "X"]},
    {"name": "Azure Orbital Quincy", "lat": 47.234, "lon": -119.852, "operator": "Microsoft", "bands": ["S", "X", "Ka"]},
]

# ── RF Band Reference ────────────────────────────────────────────────

RF_BANDS = {
    "UHF": {"range_mhz": (300, 1000), "typical_use": "CubeSat TT&C, IoT", "sdr_feasible": True},
    "L": {"range_mhz": (1000, 2000), "typical_use": "GPS, Iridium, navigation", "sdr_feasible": True},
    "S": {"range_mhz": (2000, 4000), "typical_use": "TT&C uplink/downlink", "sdr_feasible": True},
    "C": {"range_mhz": (4000, 8000), "typical_use": "Legacy satcom", "sdr_feasible": True},
    "X": {"range_mhz": (8000, 12000), "typical_use": "Earth observation, military", "sdr_feasible": False},
    "Ku": {"range_mhz": (12000, 18000), "typical_use": "DTH broadcasting, VSAT", "sdr_feasible": False},
    "Ka": {"range_mhz": (26500, 40000), "typical_use": "High-throughput, deep space", "sdr_feasible": False},
    "V": {"range_mhz": (40000, 75000), "typical_use": "Next-gen HTS", "sdr_feasible": False},
}


@dataclass
class SatelliteTarget:
    """A satellite target with orbital and RF metadata."""
    name: str
    norad_id: int
    intl_designator: str = ""
    orbit_type: str = ""  # LEO, MEO, GEO, HEO
    bands: list[str] = field(default_factory=list)
    uplink_freq_mhz: Optional[float] = None
    downlink_freq_mhz: Optional[float] = None
    encryption: str = "unknown"  # none, AES-256, proprietary, unknown
    auth: str = "unknown"  # none, HMAC, PKI, unknown
    protocol: str = "CCSDS"  # CCSDS, proprietary
    tle_line1: str = ""
    tle_line2: str = ""

    @property
    def attack_surface_score(self) -> float:
        """Heuristic attack surface score (0-10)."""
        score = 5.0
        if self.encryption == "none":
            score += 2.0
        if self.auth == "none":
            score += 2.0
        if any(RF_BANDS.get(b, {}).get("sdr_feasible") for b in self.bands):
            score += 1.0
        if self.orbit_type == "LEO":
            score += 0.5  # Closer = stronger signal
        if self.protocol == "CCSDS":
            score += 0.5  # Well-documented protocol
        return min(score, 10.0)


@dataclass
class OrbitalPass:
    """A satellite pass over a ground location."""
    aos_time: datetime  # Acquisition of Signal
    los_time: datetime  # Loss of Signal
    max_elevation: float  # degrees
    azimuth_aos: float
    azimuth_los: float
    duration_seconds: float
    distance_km: float


@dataclass
class LinkBudget:
    """RF link budget analysis for intercept feasibility."""
    frequency_mhz: float
    distance_km: float
    eirp_dbw: float = 10.0  # Satellite EIRP (typical LEO)
    rx_gain_db: float = 20.0  # Receiver antenna gain
    rx_noise_temp_k: float = 150.0  # System noise temperature

    @property
    def free_space_loss_db(self) -> float:
        """Free Space Path Loss (FSPL)."""
        freq_hz = self.frequency_mhz * 1e6
        dist_m = self.distance_km * 1e3
        return 20 * math.log10(dist_m) + 20 * math.log10(freq_hz) + 20 * math.log10(4 * math.pi / 3e8)

    @property
    def received_power_dbw(self) -> float:
        return self.eirp_dbw - self.free_space_loss_db + self.rx_gain_db

    @property
    def noise_power_dbw(self) -> float:
        """Noise power in 1 MHz bandwidth."""
        k_boltzmann = 1.380649e-23
        bandwidth = 1e6
        return 10 * math.log10(k_boltzmann * self.rx_noise_temp_k * bandwidth)

    @property
    def snr_db(self) -> float:
        return self.received_power_dbw - self.noise_power_dbw

    @property
    def intercept_feasible(self) -> bool:
        """SNR > 10 dB is generally sufficient for demodulation."""
        return self.snr_db > 10.0

    def summary(self) -> dict:
        return {
            "frequency_mhz": self.frequency_mhz,
            "distance_km": self.distance_km,
            "fspl_db": round(self.free_space_loss_db, 2),
            "rx_power_dbw": round(self.received_power_dbw, 2),
            "snr_db": round(self.snr_db, 2),
            "intercept_feasible": self.intercept_feasible,
        }


# ── Sample Satellite Catalog ─────────────────────────────────────────

SAMPLE_CATALOG: list[SatelliteTarget] = [
    SatelliteTarget(
        name="ISS (ZARYA)",
        norad_id=25544,
        orbit_type="LEO",
        bands=["S", "UHF", "L"],
        downlink_freq_mhz=145.800,
        uplink_freq_mhz=437.800,
        encryption="none",
        auth="none",
        protocol="CCSDS",
    ),
    SatelliteTarget(
        name="NOAA-19",
        norad_id=33591,
        orbit_type="LEO",
        bands=["L", "S"],
        downlink_freq_mhz=137.100,
        encryption="none",
        auth="none",
    ),
    SatelliteTarget(
        name="CubeSat (Generic)",
        norad_id=99999,
        orbit_type="LEO",
        bands=["UHF"],
        downlink_freq_mhz=437.000,
        uplink_freq_mhz=145.000,
        encryption="none",
        auth="none",
        protocol="CCSDS",
    ),
    SatelliteTarget(
        name="Starlink (Generic)",
        norad_id=44238,
        orbit_type="LEO",
        bands=["Ku", "Ka"],
        encryption="AES-256",
        auth="PKI",
        protocol="proprietary",
    ),
]


def parse_tle(tle_text: str) -> list[dict]:
    """Parse TLE (Two-Line Element) data into structured records."""
    lines = [l.strip() for l in tle_text.strip().splitlines() if l.strip()]
    records = []
    i = 0
    while i < len(lines):
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            records.append({
                "name": lines[i],
                "line1": lines[i + 1],
                "line2": lines[i + 2],
                "norad_id": int(lines[i + 1][2:7].strip()),
                "epoch_year": int(lines[i + 1][18:20]),
                "epoch_day": float(lines[i + 1][20:32]),
                "inclination": float(lines[i + 2][8:16]),
                "eccentricity": float("0." + lines[i + 2][26:33].strip()),
                "period_minutes": 1440.0 / float(lines[i + 2][52:63]),
            })
            i += 3
        else:
            i += 1
    return records


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def find_nearby_ground_stations(lat: float, lon: float, max_distance_km: float = 2000) -> list[dict]:
    """Find ground stations within range of a location."""
    results = []
    for gs in GROUND_STATIONS:
        dist = haversine_km(lat, lon, gs["lat"], gs["lon"])
        if dist <= max_distance_km:
            results.append({**gs, "distance_km": round(dist, 1)})
    return sorted(results, key=lambda x: x["distance_km"])


def assess_attack_surface(target: SatelliteTarget) -> dict:
    """Comprehensive attack surface assessment for a satellite target."""
    vulns = []

    if target.encryption == "none":
        vulns.append({
            "id": "SAT-001",
            "severity": "CRITICAL",
            "title": "No encryption on command uplink",
            "description": "Telecommand link lacks encryption. An attacker with SDR equipment "
                           "and knowledge of the uplink frequency could inject arbitrary commands.",
            "mitre_attack": "T1557 — Adversary-in-the-Middle",
        })

    if target.auth == "none":
        vulns.append({
            "id": "SAT-002",
            "severity": "CRITICAL",
            "title": "No authentication on command channel",
            "description": "No cryptographic authentication on telecommand packets. "
                           "Commands can be spoofed without knowledge of any shared secret.",
            "mitre_attack": "T1599.001 — Network Boundary Bridging",
        })

    if any(RF_BANDS.get(b, {}).get("sdr_feasible") for b in target.bands):
        feasible_bands = [b for b in target.bands if RF_BANDS.get(b, {}).get("sdr_feasible")]
        vulns.append({
            "id": "SAT-003",
            "severity": "HIGH",
            "title": f"SDR-interceptable bands: {', '.join(feasible_bands)}",
            "description": f"Operating on {', '.join(feasible_bands)} band(s) which are receivable "
                           "with consumer SDR hardware (RTL-SDR, HackRF, USRP).",
            "mitre_attack": "T1040 — Network Sniffing",
        })

    if target.protocol == "CCSDS":
        vulns.append({
            "id": "SAT-004",
            "severity": "MEDIUM",
            "title": "Standard CCSDS protocol stack",
            "description": "Uses publicly documented CCSDS protocols. Packet structure, "
                           "service types, and command formats can be reverse-engineered from standards.",
            "mitre_attack": "T1046 — Network Service Discovery",
        })

    if target.orbit_type == "LEO":
        vulns.append({
            "id": "SAT-005",
            "severity": "MEDIUM",
            "title": "LEO orbit — high signal strength at ground level",
            "description": "Low Earth Orbit provides stronger received signals, reducing "
                           "the equipment needed for interception and uplink injection.",
        })

    return {
        "target": target.name,
        "norad_id": target.norad_id,
        "attack_surface_score": target.attack_surface_score,
        "vulnerabilities": vulns,
        "bands": target.bands,
        "orbit": target.orbit_type,
    }


# ── CLI ───────────────────────────────────────────────────────────────

@click.command()
@click.option("--sat", default=None, help="Satellite name to search")
@click.option("--sat-id", default=None, type=int, help="NORAD catalog ID")
@click.option("--location", default=None, help="Observer lat,lon (e.g., '46.1,-64.8')")
@click.option("--passes", default=5, help="Number of passes to predict")
@click.option("--ground-stations", is_flag=True, help="Show nearby ground stations")
@click.option("--assess", is_flag=True, help="Run attack surface assessment")
@click.option("--link-budget", is_flag=True, help="Calculate link budget for intercept")
@click.option("--output-json", default=None, help="Export results to JSON")
@click.option("--catalog", default=None, help="Show satellite catalog")
def main(sat, sat_id, location, passes, ground_stations, assess, link_budget, output_json, catalog):
    """Orbital Recon — Satellite OSINT & Attack Surface Mapper."""

    console.print(Panel.fit(
        "[bold cyan]ORBITAL RECON[/bold cyan]\n"
        "[dim]Satellite OSINT & Attack Surface Mapper[/dim]",
        border_style="cyan",
    ))

    # Parse location
    obs_lat, obs_lon = None, None
    if location:
        obs_lat, obs_lon = map(float, location.split(","))

    # Find target satellite
    target = None
    if sat:
        for s in SAMPLE_CATALOG:
            if sat.lower() in s.name.lower():
                target = s
                break
    elif sat_id:
        for s in SAMPLE_CATALOG:
            if s.norad_id == sat_id:
                target = s
                break

    if catalog:
        table = Table(title="Satellite Catalog")
        table.add_column("Name", style="cyan")
        table.add_column("NORAD ID", justify="right")
        table.add_column("Orbit")
        table.add_column("Bands")
        table.add_column("Crypto")
        table.add_column("Auth")
        table.add_column("Score", justify="right", style="red")

        for s in SAMPLE_CATALOG:
            table.add_row(
                s.name, str(s.norad_id), s.orbit_type,
                ", ".join(s.bands), s.encryption, s.auth,
                f"{s.attack_surface_score:.1f}",
            )
        console.print(table)
        return

    if ground_stations and obs_lat is not None:
        nearby = find_nearby_ground_stations(obs_lat, obs_lon)
        table = Table(title=f"Ground Stations near ({obs_lat}, {obs_lon})")
        table.add_column("Name", style="cyan")
        table.add_column("Operator")
        table.add_column("Bands")
        table.add_column("Distance (km)", justify="right")

        for gs in nearby:
            table.add_row(gs["name"], gs["operator"], ", ".join(gs["bands"]), str(gs["distance_km"]))
        console.print(table)
        return

    if target and assess:
        result = assess_attack_surface(target)
        console.print(f"\n[bold]Attack Surface Assessment: {target.name}[/bold]")
        console.print(f"[cyan]Score:[/cyan] [red]{result['attack_surface_score']:.1f}/10[/red]\n")

        for vuln in result["vulnerabilities"]:
            severity_color = {"CRITICAL": "red", "HIGH": "yellow", "MEDIUM": "blue"}.get(vuln["severity"], "white")
            console.print(f"  [{severity_color}][{vuln['severity']}][/{severity_color}] {vuln['id']}: {vuln['title']}")
            console.print(f"    [dim]{vuln['description']}[/dim]")
            if "mitre_attack" in vuln:
                console.print(f"    [dim italic]MITRE: {vuln['mitre_attack']}[/dim italic]")
            console.print()

        if output_json:
            Path(output_json).write_text(json.dumps(result, indent=2))
            console.print(f"[green]Results saved to {output_json}[/green]")
        return

    if target and link_budget:
        freq = target.downlink_freq_mhz or 437.0
        distances = [500, 1000, 1500, 2000, 2500]
        table = Table(title=f"Link Budget Analysis — {target.name} ({freq} MHz)")
        table.add_column("Distance (km)", justify="right")
        table.add_column("FSPL (dB)", justify="right")
        table.add_column("Rx Power (dBW)", justify="right")
        table.add_column("SNR (dB)", justify="right")
        table.add_column("Intercept?", justify="center")

        for d in distances:
            lb = LinkBudget(frequency_mhz=freq, distance_km=d)
            feasible = "[green]YES[/green]" if lb.intercept_feasible else "[red]NO[/red]"
            table.add_row(str(d), f"{lb.free_space_loss_db:.1f}",
                          f"{lb.received_power_dbw:.1f}", f"{lb.snr_db:.1f}", feasible)
        console.print(table)
        return

    if not target and not catalog and not ground_stations:
        console.print("[yellow]Specify --sat, --catalog, or --ground-stations. Use --help for options.[/yellow]")


if __name__ == "__main__":
    main()
