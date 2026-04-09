"""
RF Signal Intelligence & SDR Attack Simulator
===============================================
Simulates software-defined radio (SDR) attacks against satellite
downlinks. Generates IQ sample data, simulates demodulation,
and models jamming/spoofing scenarios.

This is a simulation framework — no actual RF transmission.
For use in red team planning and ground station security assessments.

Attack scenarios modeled:
- Satellite downlink interception (passive SIGINT)
- Uplink command injection (requires knowledge of protocol + freq)
- GPS/GNSS spoofing signal generation
- Jamming effectiveness modeling
- Meaconing (replay attacks on navigation signals)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SignalParams:
    """Parameters describing an RF signal."""
    frequency_hz: float
    bandwidth_hz: float
    modulation: str  # BPSK, QPSK, 8PSK, FSK, GMSK, OQPSK
    symbol_rate_sps: float  # symbols per second
    power_dbm: float = -80.0
    coding: str = "none"  # convolutional, turbo, ldpc, none
    fec_rate: str = "1/2"


@dataclass
class JammingScenario:
    """Model a jamming attack against a satellite link."""
    target_freq_hz: float
    target_bandwidth_hz: float
    target_power_dbm: float
    jammer_power_dbm: float
    jammer_distance_km: float
    jammer_type: str = "barrage"  # barrage, spot, sweep, pulse

    @property
    def jammer_eirp_dbw(self) -> float:
        return self.jammer_power_dbm / 1000.0  # simplified

    def jamming_margin_db(self) -> float:
        """
        Calculate Jamming-to-Signal ratio (J/S).
        J/S > 0 dB means the jammer overpowers the signal.
        """
        # Free space path loss for jammer
        freq_hz = self.target_freq_hz
        dist_m = self.jammer_distance_km * 1000
        fspl = 20 * math.log10(dist_m) + 20 * math.log10(freq_hz) - 147.55

        j_at_rx = self.jammer_power_dbm - fspl
        j_over_s = j_at_rx - self.target_power_dbm
        return j_over_s

    def effectiveness(self) -> dict:
        j_s = self.jamming_margin_db()
        if j_s > 20:
            effect = "DENIAL — signal completely unusable"
        elif j_s > 10:
            effect = "SEVERE DEGRADATION — high BER, frequent dropouts"
        elif j_s > 3:
            effect = "DEGRADATION — increased errors, reduced throughput"
        elif j_s > 0:
            effect = "MARGINAL — minor impact, FEC can likely compensate"
        else:
            effect = "INEFFECTIVE — jammer too weak to impact signal"

        return {
            "j_s_ratio_db": round(j_s, 2),
            "effect": effect,
            "jammer_type": self.jammer_type,
            "countermeasures": self._suggest_countermeasures(j_s),
        }

    def _suggest_countermeasures(self, j_s: float) -> list[str]:
        countermeasures = []
        if j_s > 0:
            countermeasures.append("Frequency hopping (FHSS) — forces jammer to sweep")
            countermeasures.append("Increase satellite EIRP (if possible)")
            countermeasures.append("Directional ground antenna to null jammer")
            countermeasures.append("Spread spectrum (DSSS) with higher processing gain")
        if j_s > 10:
            countermeasures.append("Switch to alternate ground station outside jammer range")
            countermeasures.append("Report interference to ITU / national spectrum authority")
        if self.jammer_type == "barrage":
            countermeasures.append("Narrowband filter — barrage jammers waste power across bandwidth")
        return countermeasures


class IQSampleGenerator:
    """
    Generate simulated IQ (In-phase/Quadrature) samples.

    IQ data is the raw representation of RF signals as captured
    by SDR hardware. Used for:
    - Testing demodulation pipelines
    - Generating spoofed satellite signals
    - Creating training data for ML-based signal classifiers
    """

    def __init__(self, sample_rate: float = 2.4e6):
        self.sample_rate = sample_rate

    def generate_carrier(self, frequency_hz: float, duration_s: float,
                          amplitude: float = 1.0) -> np.ndarray:
        """Generate a pure carrier tone as complex IQ samples."""
        n_samples = int(self.sample_rate * duration_s)
        t = np.arange(n_samples) / self.sample_rate
        signal = amplitude * np.exp(1j * 2 * np.pi * frequency_hz * t)
        return signal.astype(np.complex64)

    def generate_bpsk(self, data_bits: np.ndarray, symbol_rate: float,
                       carrier_freq: float = 0) -> np.ndarray:
        """
        Generate BPSK modulated IQ samples.
        BPSK is used in many satellite telecommand uplinks.
        """
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        n_samples = len(data_bits) * samples_per_symbol

        # Map bits to symbols: 0 → -1, 1 → +1
        symbols = 2.0 * data_bits.astype(np.float32) - 1.0

        # Upsample
        signal = np.repeat(symbols, samples_per_symbol)

        # Apply carrier (if non-zero)
        if carrier_freq != 0:
            t = np.arange(n_samples) / self.sample_rate
            carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
            signal = signal * carrier

        return signal.astype(np.complex64)

    def generate_qpsk(self, data_bits: np.ndarray, symbol_rate: float,
                       carrier_freq: float = 0) -> np.ndarray:
        """
        Generate QPSK modulated IQ samples.
        QPSK is standard for satellite telemetry downlinks.
        """
        # Ensure even number of bits
        if len(data_bits) % 2 != 0:
            data_bits = np.append(data_bits, 0)

        samples_per_symbol = int(self.sample_rate / symbol_rate)

        # Map bit pairs to QPSK constellation
        i_bits = data_bits[0::2]
        q_bits = data_bits[1::2]
        i_symbols = 2.0 * i_bits.astype(np.float32) - 1.0
        q_symbols = 2.0 * q_bits.astype(np.float32) - 1.0

        # Complex symbols
        symbols = (i_symbols + 1j * q_symbols) / np.sqrt(2)

        # Upsample
        signal = np.repeat(symbols, samples_per_symbol)

        if carrier_freq != 0:
            t = np.arange(len(signal)) / self.sample_rate
            carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
            signal = signal * carrier

        return signal.astype(np.complex64)

    def add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Additive White Gaussian Noise at specified SNR."""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
        return (signal + noise).astype(np.complex64)

    def generate_jamming_signal(self, duration_s: float,
                                  bandwidth_hz: float,
                                  jam_type: str = "barrage") -> np.ndarray:
        """
        Generate a jamming signal for simulation.

        Types:
        - barrage: Wideband noise across entire bandwidth
        - spot: Narrowband CW on center frequency
        - sweep: Frequency sweep across bandwidth
        - pulse: Pulsed high-power bursts
        """
        n_samples = int(self.sample_rate * duration_s)

        if jam_type == "barrage":
            # Wideband noise
            signal = np.sqrt(0.5) * (
                np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
            )
            # Bandlimit via crude filtering
            # (in practice, use proper FIR filter)

        elif jam_type == "spot":
            # CW tone at center
            t = np.arange(n_samples) / self.sample_rate
            signal = np.exp(1j * 2 * np.pi * 0 * t)  # baseband

        elif jam_type == "sweep":
            # Linear frequency sweep
            t = np.arange(n_samples) / self.sample_rate
            phase = 2 * np.pi * (
                -bandwidth_hz / 2 * t +
                (bandwidth_hz / (2 * duration_s)) * t ** 2
            )
            signal = np.exp(1j * phase)

        elif jam_type == "pulse":
            # Pulsed jamming (50% duty cycle)
            signal = np.zeros(n_samples, dtype=np.complex64)
            pulse_len = int(self.sample_rate * 0.001)  # 1ms pulses
            period = pulse_len * 2
            for i in range(0, n_samples, period):
                end = min(i + pulse_len, n_samples)
                signal[i:end] = np.random.randn(end - i) + 1j * np.random.randn(end - i)
        else:
            signal = np.zeros(n_samples, dtype=np.complex64)

        return signal.astype(np.complex64)

    def save_iq_file(self, signal: np.ndarray, filepath: str, fmt: str = "cf32"):
        """
        Save IQ samples to file.

        Formats:
        - cf32: Complex float32 (GNU Radio / SDR++)
        - cs16: Complex int16 (HackRF / SDR#)
        - cu8: Complex uint8 (RTL-SDR)
        """
        if fmt == "cf32":
            signal.astype(np.complex64).tofile(filepath)
        elif fmt == "cs16":
            interleaved = np.empty(len(signal) * 2, dtype=np.int16)
            interleaved[0::2] = (signal.real * 32767).astype(np.int16)
            interleaved[1::2] = (signal.imag * 32767).astype(np.int16)
            interleaved.tofile(filepath)
        elif fmt == "cu8":
            interleaved = np.empty(len(signal) * 2, dtype=np.uint8)
            interleaved[0::2] = ((signal.real + 1) * 127.5).clip(0, 255).astype(np.uint8)
            interleaved[1::2] = ((signal.imag + 1) * 127.5).clip(0, 255).astype(np.uint8)
            interleaved.tofile(filepath)


class GNSSSpoofingSimulator:
    """
    Simulate GPS/GNSS spoofing signal generation.

    GNSS spoofing is a well-documented threat to space and
    autonomous systems. This simulator generates the parameters
    needed to model a spoofing attack — no actual RF transmission.

    Reference: CISA Advisory on GPS Spoofing Threats
    """

    # GPS L1 C/A signal parameters
    GPS_L1_FREQ = 1575.42e6  # Hz
    GPS_L1_BANDWIDTH = 2.046e6  # Hz
    GPS_CHIP_RATE = 1.023e6  # chips/sec
    GPS_NAV_RATE = 50  # bps

    # GLONASS L1
    GLONASS_L1_FREQ = 1602.0e6

    # Galileo E1
    GALILEO_E1_FREQ = 1575.42e6

    @dataclass
    class SpoofingConfig:
        target_lat: float  # Target spoofed latitude
        target_lon: float  # Target spoofed longitude
        target_alt: float  # Target spoofed altitude (m)
        time_offset_ns: float = 0  # Time offset in nanoseconds
        num_satellites: int = 8  # Number of spoofed SVs
        power_advantage_db: float = 3.0  # Power over real signals

    def generate_spoof_parameters(self, config) -> dict:
        """
        Calculate the spoofing signal parameters needed to
        make a receiver report a fake position.

        Returns pseudorange offsets for each spoofed satellite.
        """
        C = 299792458.0  # speed of light m/s

        # Generate pseudorange offsets from target position
        # Each satellite needs a different delay to triangulate
        # to the target position
        sv_params = []
        for sv_id in range(1, config.num_satellites + 1):
            # Simulate satellite positions (simplified)
            sv_elevation = 20 + (sv_id * 10) % 70  # degrees
            sv_azimuth = (sv_id * 45) % 360  # degrees

            # Pseudorange from target position to satellite
            # (simplified — real implementation needs ephemeris)
            base_range_m = 20200e3  # ~GPS orbit altitude
            range_offset_m = config.target_alt * math.sin(math.radians(sv_elevation))

            # Time offset contribution
            time_offset_m = config.time_offset_ns * 1e-9 * C

            sv_params.append({
                "sv_id": sv_id,
                "prn": sv_id,
                "elevation_deg": sv_elevation,
                "azimuth_deg": sv_azimuth,
                "pseudorange_offset_m": round(range_offset_m + time_offset_m, 3),
                "doppler_hz": round(np.random.uniform(-5000, 5000), 1),
                "cn0_dbhz": round(35 + config.power_advantage_db + np.random.uniform(-3, 3), 1),
            })

        return {
            "target_position": {
                "latitude": config.target_lat,
                "longitude": config.target_lon,
                "altitude_m": config.target_alt,
            },
            "spoofed_time_offset_ns": config.time_offset_ns,
            "carrier_frequency_hz": self.GPS_L1_FREQ,
            "chip_rate_cps": self.GPS_CHIP_RATE,
            "power_advantage_db": config.power_advantage_db,
            "satellites": sv_params,
            "detection_risk": self._assess_detection_risk(config),
        }

    def _assess_detection_risk(self, config) -> dict:
        """Assess the likelihood of the spoofing being detected."""
        risks = []

        if config.power_advantage_db > 10:
            risks.append("HIGH: Excessive power advantage — obvious to power-monitoring receivers")
        elif config.power_advantage_db > 5:
            risks.append("MEDIUM: Noticeable power increase — sophisticated receivers may flag")

        if abs(config.time_offset_ns) > 100:
            risks.append("HIGH: Large time offset — NTP cross-check will detect inconsistency")

        if config.num_satellites < 6:
            risks.append("MEDIUM: Few spoofed SVs — RAIM may detect inconsistency with real signals")

        return {
            "overall_risk": "HIGH" if any("HIGH" in r for r in risks) else "MEDIUM" if risks else "LOW",
            "detection_vectors": risks,
            "countermeasures_to_evade": [
                "Match power levels to ambient signal strength",
                "Gradually introduce position offset (avoid step change)",
                "Spoof all visible SVs to prevent RAIM detection",
                "Maintain consistent Doppler shifts with spoofed geometry",
                "Account for multipath and atmospheric delays",
            ],
        }


# ── Satellite RF Fingerprinting ───────────────────────────────────────

class RFFingerprinter:
    """
    RF fingerprinting for satellite transmitter identification.

    Each transmitter has unique hardware imperfections (IQ imbalance,
    phase noise, frequency offset) that create a fingerprint.
    This can be used for:
    - Authenticating legitimate satellite signals
    - Detecting spoofed/replayed signals
    - Identifying rogue transmitters

    Offensive use: Extract fingerprint to improve spoofing fidelity.
    """

    @staticmethod
    def extract_features(iq_samples: np.ndarray) -> dict:
        """Extract RF fingerprint features from IQ samples."""
        # IQ imbalance
        i_power = np.mean(iq_samples.real ** 2)
        q_power = np.mean(iq_samples.imag ** 2)
        iq_imbalance_db = 10 * np.log10(i_power / (q_power + 1e-10))

        # Carrier frequency offset (CFO)
        # Estimate from phase progression
        phase = np.angle(iq_samples)
        phase_diff = np.diff(np.unwrap(phase))
        cfo_estimate = np.mean(phase_diff) / (2 * np.pi)

        # Phase noise (variance of instantaneous frequency)
        freq_noise = np.var(phase_diff)

        # Amplitude statistics
        amplitude = np.abs(iq_samples)
        amp_mean = float(np.mean(amplitude))
        amp_std = float(np.std(amplitude))
        amp_kurtosis = float(np.mean((amplitude - amp_mean) ** 4) / (amp_std ** 4 + 1e-10))

        # Spectral features
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(iq_samples[:4096])))
        spectral_centroid = float(np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-10))
        spectral_flatness = float(
            np.exp(np.mean(np.log(spectrum + 1e-10))) / (np.mean(spectrum) + 1e-10)
        )

        return {
            "iq_imbalance_db": round(iq_imbalance_db, 4),
            "carrier_freq_offset": round(float(cfo_estimate), 6),
            "phase_noise_var": round(float(freq_noise), 8),
            "amplitude_mean": round(amp_mean, 4),
            "amplitude_std": round(amp_std, 4),
            "amplitude_kurtosis": round(amp_kurtosis, 4),
            "spectral_centroid": round(spectral_centroid, 2),
            "spectral_flatness": round(spectral_flatness, 6),
        }

    @staticmethod
    def compare_fingerprints(fp1: dict, fp2: dict) -> dict:
        """Compare two RF fingerprints to assess if they're from the same transmitter."""
        keys = ["iq_imbalance_db", "carrier_freq_offset", "phase_noise_var",
                "amplitude_kurtosis", "spectral_flatness"]
        distances = {}
        total_distance = 0
        for key in keys:
            if key in fp1 and key in fp2:
                d = abs(fp1[key] - fp2[key])
                distances[key] = round(d, 6)
                # Normalize by typical range
                total_distance += d

        match_score = max(0, 1 - total_distance / 10)  # simplified
        return {
            "match_score": round(match_score, 4),
            "likely_same_transmitter": match_score > 0.85,
            "per_feature_distance": distances,
        }
