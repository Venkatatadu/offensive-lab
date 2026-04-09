"""
CCSDS Space Packet Protocol Implementation
===========================================
Implements CCSDS 133.0-B-2 (Space Packet Protocol) and
CCSDS 232.0-B-4 (TC Space Data Link Protocol) for
offensive security research.

Reference: https://public.ccsds.org/Pubs/133x0b2e1.pdf
"""

from __future__ import annotations

import struct
import enum
import hashlib
from dataclasses import dataclass, field
from typing import Optional


class PacketType(enum.IntEnum):
    TELEMETRY = 0
    TELECOMMAND = 1


class SequenceFlags(enum.IntEnum):
    CONTINUATION = 0b00
    FIRST = 0b01
    LAST = 0b10
    UNSEGMENTED = 0b11


class ServiceType(enum.IntEnum):
    """PUS (Packet Utilization Standard) service types - ECSS-E-ST-70-41C"""
    HOUSEKEEPING = 3
    EVENT_REPORTING = 5
    MEMORY_MGMT = 6
    FUNCTION_MGMT = 8
    TIME_MGMT = 9
    ON_BOARD_STORAGE = 15
    TEST = 17
    ON_BOARD_CONTROL = 18
    EVENT_ACTION = 19


@dataclass
class CCSDSPrimaryHeader:
    """
    CCSDS Space Packet Primary Header (6 bytes)

    Bits layout:
    [0-2]   Packet Version Number (3 bits, always 000)
    [3]     Packet Type (1 bit: 0=TM, 1=TC)
    [4]     Secondary Header Flag (1 bit)
    [5-15]  APID (11 bits)
    [16-17] Sequence Flags (2 bits)
    [18-31] Sequence Count (14 bits)
    [32-47] Data Length (16 bits) = (total_bytes_after_header - 1)
    """
    version: int = 0
    packet_type: PacketType = PacketType.TELECOMMAND
    sec_header_flag: bool = True
    apid: int = 0
    seq_flags: SequenceFlags = SequenceFlags.UNSEGMENTED
    seq_count: int = 0
    data_length: int = 0

    def pack(self) -> bytes:
        word1 = (
            (self.version & 0x07) << 13
            | (self.packet_type & 0x01) << 12
            | (int(self.sec_header_flag) & 0x01) << 11
            | (self.apid & 0x7FF)
        )
        word2 = (
            (self.seq_flags & 0x03) << 14
            | (self.seq_count & 0x3FFF)
        )
        return struct.pack(">HHH", word1, word2, self.data_length)

    @classmethod
    def unpack(cls, data: bytes) -> "CCSDSPrimaryHeader":
        if len(data) < 6:
            raise ValueError(f"Need 6 bytes, got {len(data)}")
        word1, word2, data_length = struct.unpack(">HHH", data[:6])
        return cls(
            version=(word1 >> 13) & 0x07,
            packet_type=PacketType((word1 >> 12) & 0x01),
            sec_header_flag=bool((word1 >> 11) & 0x01),
            apid=word1 & 0x7FF,
            seq_flags=SequenceFlags((word2 >> 14) & 0x03),
            seq_count=word2 & 0x3FFF,
            data_length=data_length,
        )


@dataclass
class PUSSecondaryHeader:
    """
    PUS-C Telecommand Secondary Header
    Used in European space missions (ESA standard ECSS-E-ST-70-41C)
    """
    pus_version: int = 2
    ack_flags: int = 0b1111  # Request all acknowledgements
    service_type: int = 0
    service_subtype: int = 0
    source_id: int = 0

    def pack(self) -> bytes:
        byte1 = ((self.pus_version & 0x0F) << 4) | (self.ack_flags & 0x0F)
        return struct.pack(">BBBB", byte1, self.service_type, self.service_subtype, self.source_id)

    @classmethod
    def unpack(cls, data: bytes) -> "PUSSecondaryHeader":
        if len(data) < 4:
            raise ValueError(f"Need 4 bytes for PUS header, got {len(data)}")
        byte1, stype, subtype, source = struct.unpack(">BBBB", data[:4])
        return cls(
            pus_version=(byte1 >> 4) & 0x0F,
            ack_flags=byte1 & 0x0F,
            service_type=stype,
            service_subtype=subtype,
            source_id=source,
        )


@dataclass
class TCTransferFrame:
    """
    TC Space Data Link Protocol Transfer Frame (CCSDS 232.0-B-4)

    This is the link-layer frame that wraps CCSDS space packets
    for uplink to spacecraft. Many ground stations parse these
    with minimal validation — a prime target for fuzzing.
    """
    version: int = 0
    bypass_flag: bool = False  # AD(0) vs BD(1) mode
    control_command_flag: bool = False
    spacecraft_id: int = 0
    virtual_channel_id: int = 0
    frame_length: int = 0
    frame_seq_num: int = 0
    data: bytes = b""
    fecf: Optional[int] = None  # Frame Error Control Field (CRC-16)

    def pack(self) -> bytes:
        word1 = (
            (self.version & 0x03) << 14
            | (int(self.bypass_flag) & 0x01) << 13
            | (int(self.control_command_flag) & 0x01) << 12
            | (self.spacecraft_id & 0x3FF) << 2
            | (self.virtual_channel_id & 0x03)
        )
        # Frame length = total octets in frame - 1
        frame_bytes = struct.pack(">HHB", word1, self.frame_length, self.frame_seq_num)
        frame_bytes += self.data

        if self.fecf is not None:
            frame_bytes += struct.pack(">H", self.fecf)
        else:
            # Calculate CRC-16-CCITT
            crc = self._crc16_ccitt(frame_bytes)
            frame_bytes += struct.pack(">H", crc)

        return frame_bytes

    @staticmethod
    def _crc16_ccitt(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc


@dataclass
class AOSTransferFrame:
    """
    Advanced Orbiting Systems (AOS) Transfer Frame — CCSDS 732.0-B-4
    Used for high-rate downlink (telemetry). Commonly found in
    Earth observation and deep-space missions.

    Security note: Many implementations lack authentication on the
    Insert Zone and lack encryption, making frame injection feasible
    with SDR equipment.
    """
    version: int = 1
    spacecraft_id: int = 0
    virtual_channel_id: int = 0
    frame_count: int = 0
    replay_flag: bool = False
    vc_frame_count_usage: bool = True
    vc_frame_count: int = 0
    insert_zone: bytes = b""
    data: bytes = b""

    def pack(self) -> bytes:
        word1 = (
            (self.version & 0x03) << 14
            | (self.spacecraft_id & 0xFF) << 6
            | (self.virtual_channel_id & 0x3F)
        )
        frame = struct.pack(">H", word1)
        frame += struct.pack(">I", self.frame_count & 0xFFFFFF)[1:]  # 24-bit
        frame += struct.pack(">B", (int(self.replay_flag) << 7) | (int(self.vc_frame_count_usage) << 6))
        frame += self.insert_zone
        frame += self.data
        return frame


@dataclass
class Proximity1Frame:
    """
    Proximity-1 Space Link Protocol — CCSDS 211.0-B-5
    Short-range protocol for surface-to-orbiter relay (e.g., Mars rovers).

    Security note: Proximity-1 negotiation phase has no authentication.
    A rogue transmitter could hijack the link during hailing sequence.
    """
    version: int = 0
    frame_type: int = 0  # 0=data, 1=ack, 2=directive
    spacecraft_id: int = 0
    phy_channel_id: int = 0
    sequence_num: int = 0
    data: bytes = b""

    def pack(self) -> bytes:
        header = struct.pack(
            ">BBH H",
            (self.version << 5) | (self.frame_type & 0x07),
            self.spacecraft_id & 0xFF,
            self.phy_channel_id,
            self.sequence_num,
        )
        return header + self.data


def build_tc_packet(
    apid: int,
    service_type: int,
    service_subtype: int,
    payload: bytes,
    seq_count: int = 0,
) -> bytes:
    """Build a complete CCSDS telecommand packet with PUS header."""
    pus = PUSSecondaryHeader(
        service_type=service_type,
        service_subtype=service_subtype,
    )
    pus_bytes = pus.pack()
    data_field = pus_bytes + payload

    header = CCSDSPrimaryHeader(
        packet_type=PacketType.TELECOMMAND,
        sec_header_flag=True,
        apid=apid,
        seq_flags=SequenceFlags.UNSEGMENTED,
        seq_count=seq_count,
        data_length=len(data_field) - 1,
    )
    return header.pack() + data_field


def build_tm_packet(
    apid: int,
    payload: bytes,
    seq_count: int = 0,
) -> bytes:
    """Build a CCSDS telemetry packet (no PUS secondary header)."""
    header = CCSDSPrimaryHeader(
        packet_type=PacketType.TELEMETRY,
        sec_header_flag=False,
        apid=apid,
        seq_flags=SequenceFlags.UNSEGMENTED,
        seq_count=seq_count,
        data_length=len(payload) - 1,
    )
    return header.pack() + payload
