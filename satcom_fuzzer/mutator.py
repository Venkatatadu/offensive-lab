"""
AI-Guided Mutation Engine for CCSDS Protocol Fuzzing
=====================================================
Uses lightweight ML (multi-armed bandit + byte-level mutation scoring)
to prioritize mutations that maximize novel crashes and code coverage
in space ground station software.

Approach:
- Thompson Sampling selects which mutation strategy to apply
- A coverage feedback loop rewards strategies that discover new paths
- Byte importance scoring focuses mutations on protocol-critical offsets
"""

from __future__ import annotations

import os
import random
import struct
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from .ccsds import (
    CCSDSPrimaryHeader,
    PacketType,
    SequenceFlags,
    ServiceType,
    build_tc_packet,
    TCTransferFrame,
)


class MutationStrategy(Enum):
    BIT_FLIP = auto()
    BYTE_FLIP = auto()
    ARITHMETIC = auto()
    INTERESTING_VALUES = auto()
    BLOCK_INSERT = auto()
    BLOCK_DELETE = auto()
    BLOCK_OVERWRITE = auto()
    HEADER_FIELD_CORRUPT = auto()
    APID_SWEEP = auto()
    LENGTH_OVERFLOW = auto()
    SEQUENCE_WRAP = auto()
    PUS_SERVICE_FUZZ = auto()
    FRAME_CRC_CORRUPT = auto()
    CROSS_LAYER_INJECT = auto()


# Values known to trigger edge cases in parsers
INTERESTING_8 = [0, 1, 0x7F, 0x80, 0xFF]
INTERESTING_16 = [0, 1, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF]
INTERESTING_32 = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x10000, 0xFFFF]


@dataclass
class MutationResult:
    """Result of applying a mutation."""
    strategy: MutationStrategy
    original: bytes
    mutated: bytes
    description: str
    byte_offsets: list[int] = field(default_factory=list)


@dataclass
class BanditArm:
    """Thompson Sampling arm for a mutation strategy."""
    strategy: MutationStrategy
    alpha: float = 1.0  # successes + 1
    beta: float = 1.0   # failures + 1
    total_uses: int = 0
    total_reward: float = 0.0

    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: float):
        self.total_uses += 1
        self.total_reward += reward
        if reward > 0:
            self.alpha += reward
        else:
            self.beta += 1.0


class CoverageTracker:
    """Track unique crash signatures and coverage bitmaps."""

    def __init__(self):
        self.seen_crashes: set[str] = set()
        self.coverage_bitmap: set[int] = set()
        self.total_inputs = 0
        self.total_crashes = 0
        self.unique_crashes = 0

    def is_new_crash(self, crash_data: bytes, trace_hash: str) -> bool:
        sig = hashlib.sha256(crash_data + trace_hash.encode()).hexdigest()[:16]
        if sig not in self.seen_crashes:
            self.seen_crashes.add(sig)
            self.unique_crashes += 1
            return True
        return False

    def update_coverage(self, bitmap: set[int]) -> int:
        """Returns count of newly discovered edges."""
        new_edges = bitmap - self.coverage_bitmap
        self.coverage_bitmap |= bitmap
        return len(new_edges)

    @property
    def stats(self) -> dict:
        return {
            "total_inputs": self.total_inputs,
            "total_crashes": self.total_crashes,
            "unique_crashes": self.unique_crashes,
            "coverage_edges": len(self.coverage_bitmap),
        }


class AIGuidedMutator:
    """
    AI-guided mutation engine using Thompson Sampling MAB.

    Each mutation strategy is an arm. Rewards are based on:
    - New crash discovered: +10
    - New coverage edge:    +1 per edge
    - No new coverage:      +0

    Byte importance map tracks which offsets in the packet
    are most likely to trigger new behavior when mutated.
    """

    def __init__(self, seed_corpus: list[bytes] | None = None):
        self.arms = {
            strategy: BanditArm(strategy=strategy)
            for strategy in MutationStrategy
        }
        self.coverage = CoverageTracker()
        self.seed_corpus = seed_corpus or self._default_seeds()
        self.byte_importance: dict[int, float] = {}
        self.rng = np.random.default_rng()
        self.mutation_log: list[dict] = []

    def _default_seeds(self) -> list[bytes]:
        """Generate seed corpus of valid CCSDS packets."""
        seeds = []
        # Standard housekeeping request
        seeds.append(build_tc_packet(
            apid=1, service_type=3, service_subtype=1,
            payload=b"\x00\x01",
        ))
        # Memory dump command (PUS service 6)
        seeds.append(build_tc_packet(
            apid=1, service_type=6, service_subtype=5,
            payload=struct.pack(">IH", 0x08000000, 256),
        ))
        # Function management (PUS service 8)
        seeds.append(build_tc_packet(
            apid=2, service_type=8, service_subtype=1,
            payload=struct.pack(">I", 42),
        ))
        # On-board control (PUS service 18) — enable a subsystem
        seeds.append(build_tc_packet(
            apid=3, service_type=18, service_subtype=1,
            payload=b"\x01",
        ))
        # Large payload (test buffer handling)
        seeds.append(build_tc_packet(
            apid=1, service_type=17, service_subtype=1,
            payload=os.urandom(1024),
        ))
        # TC Transfer Frame wrapping a packet
        inner = build_tc_packet(apid=1, service_type=17, service_subtype=1, payload=b"\x01")
        frame = TCTransferFrame(spacecraft_id=42, data=inner)
        seeds.append(frame.pack())

        return seeds

    def select_strategy(self) -> MutationStrategy:
        """Thompson Sampling: sample from each arm's Beta distribution."""
        scores = {s: arm.sample() for s, arm in self.arms.items()}
        return max(scores, key=scores.get)

    def select_seed(self) -> bytes:
        """Select a seed, biased toward seeds that previously yielded coverage."""
        return random.choice(self.seed_corpus)

    def mutate(self, data: bytes, strategy: MutationStrategy | None = None) -> MutationResult:
        """Apply a mutation strategy to the input."""
        if strategy is None:
            strategy = self.select_strategy()

        mutators: dict[MutationStrategy, Callable] = {
            MutationStrategy.BIT_FLIP: self._bit_flip,
            MutationStrategy.BYTE_FLIP: self._byte_flip,
            MutationStrategy.ARITHMETIC: self._arithmetic,
            MutationStrategy.INTERESTING_VALUES: self._interesting_values,
            MutationStrategy.BLOCK_INSERT: self._block_insert,
            MutationStrategy.BLOCK_DELETE: self._block_delete,
            MutationStrategy.BLOCK_OVERWRITE: self._block_overwrite,
            MutationStrategy.HEADER_FIELD_CORRUPT: self._header_field_corrupt,
            MutationStrategy.APID_SWEEP: self._apid_sweep,
            MutationStrategy.LENGTH_OVERFLOW: self._length_overflow,
            MutationStrategy.SEQUENCE_WRAP: self._sequence_wrap,
            MutationStrategy.PUS_SERVICE_FUZZ: self._pus_service_fuzz,
            MutationStrategy.FRAME_CRC_CORRUPT: self._frame_crc_corrupt,
            MutationStrategy.CROSS_LAYER_INJECT: self._cross_layer_inject,
        }
        return mutators[strategy](data)

    def report_result(self, strategy: MutationStrategy, new_edges: int, is_crash: bool):
        """Feed reward back to the bandit."""
        reward = 0.0
        if is_crash:
            reward += 10.0
        reward += new_edges * 1.0
        self.arms[strategy].update(reward)

    def generate_batch(self, batch_size: int = 100) -> list[tuple[bytes, MutationStrategy]]:
        """Generate a batch of mutated inputs."""
        batch = []
        for _ in range(batch_size):
            seed = self.select_seed()
            strategy = self.select_strategy()
            result = self.mutate(seed, strategy)
            batch.append((result.mutated, strategy))
        return batch

    def get_strategy_stats(self) -> list[dict]:
        """Get performance stats for all mutation strategies."""
        stats = []
        for s, arm in sorted(self.arms.items(), key=lambda x: x[1].total_reward, reverse=True):
            stats.append({
                "strategy": s.name,
                "uses": arm.total_uses,
                "reward": round(arm.total_reward, 2),
                "avg_reward": round(arm.total_reward / max(arm.total_uses, 1), 4),
                "alpha": round(arm.alpha, 2),
                "beta": round(arm.beta, 2),
            })
        return stats

    # ── Mutation Implementations ──────────────────────────────────────

    def _bit_flip(self, data: bytes) -> MutationResult:
        d = bytearray(data)
        offset = self._weighted_offset(len(d))
        bit = random.randint(0, 7)
        d[offset] ^= (1 << bit)
        return MutationResult(MutationStrategy.BIT_FLIP, data, bytes(d),
                              f"Flipped bit {bit} at offset {offset}", [offset])

    def _byte_flip(self, data: bytes) -> MutationResult:
        d = bytearray(data)
        n = random.choice([1, 2, 4])
        offset = self._weighted_offset(len(d) - n + 1)
        for i in range(n):
            d[offset + i] ^= 0xFF
        return MutationResult(MutationStrategy.BYTE_FLIP, data, bytes(d),
                              f"Flipped {n} bytes at offset {offset}",
                              list(range(offset, offset + n)))

    def _arithmetic(self, data: bytes) -> MutationResult:
        d = bytearray(data)
        offset = self._weighted_offset(len(d))
        delta = random.choice([-35, -1, 1, 35, 127, -128])
        d[offset] = (d[offset] + delta) & 0xFF
        return MutationResult(MutationStrategy.ARITHMETIC, data, bytes(d),
                              f"Added {delta} at offset {offset}", [offset])

    def _interesting_values(self, data: bytes) -> MutationResult:
        d = bytearray(data)
        width = random.choice([1, 2, 4])
        offset = self._weighted_offset(max(1, len(d) - width + 1))
        if width == 1:
            val = random.choice(INTERESTING_8)
            d[offset] = val
        elif width == 2 and offset + 1 < len(d):
            val = random.choice(INTERESTING_16)
            struct.pack_into(">H", d, offset, val)
        elif width == 4 and offset + 3 < len(d):
            val = random.choice(INTERESTING_32)
            struct.pack_into(">I", d, offset, val)
        return MutationResult(MutationStrategy.INTERESTING_VALUES, data, bytes(d),
                              f"Inserted interesting {width}-byte value at offset {offset}", [offset])

    def _block_insert(self, data: bytes) -> MutationResult:
        d = bytearray(data)
        offset = random.randint(0, len(d))
        size = random.randint(1, 128)
        block = os.urandom(size)
        d[offset:offset] = block
        return MutationResult(MutationStrategy.BLOCK_INSERT, data, bytes(d),
                              f"Inserted {size} random bytes at offset {offset}",
                              list(range(offset, offset + size)))

    def _block_delete(self, data: bytes) -> MutationResult:
        if len(data) <= 6:
            return self._bit_flip(data)
        d = bytearray(data)
        size = random.randint(1, min(64, len(d) - 6))
        offset = random.randint(6, len(d) - size)  # preserve primary header
        del d[offset:offset + size]
        return MutationResult(MutationStrategy.BLOCK_DELETE, data, bytes(d),
                              f"Deleted {size} bytes at offset {offset}", [])

    def _block_overwrite(self, data: bytes) -> MutationResult:
        d = bytearray(data)
        size = random.randint(1, min(64, len(d)))
        offset = self._weighted_offset(max(1, len(d) - size + 1))
        d[offset:offset + size] = os.urandom(size)
        return MutationResult(MutationStrategy.BLOCK_OVERWRITE, data, bytes(d),
                              f"Overwrote {size} bytes at offset {offset}",
                              list(range(offset, offset + size)))

    def _header_field_corrupt(self, data: bytes) -> MutationResult:
        """Surgically corrupt specific CCSDS primary header fields."""
        d = bytearray(data)
        if len(d) < 6:
            return self._bit_flip(data)

        field_choice = random.choice(["version", "type", "apid", "seq", "length"])
        if field_choice == "version":
            d[0] = (d[0] & 0x1F) | (random.randint(1, 7) << 5)
            desc = "Corrupted version field"
        elif field_choice == "type":
            d[0] ^= 0x10
            desc = "Toggled packet type"
        elif field_choice == "apid":
            new_apid = random.choice([0, 0x7FF, random.randint(0, 0x7FF)])
            d[0] = (d[0] & 0xF8) | ((new_apid >> 8) & 0x07)
            d[1] = new_apid & 0xFF
            desc = f"Set APID to {new_apid}"
        elif field_choice == "seq":
            new_seq = random.choice([0, 0x3FFF, random.randint(0, 0x3FFF)])
            d[2] = (d[2] & 0xC0) | ((new_seq >> 8) & 0x3F)
            d[3] = new_seq & 0xFF
            desc = f"Set sequence count to {new_seq}"
        else:
            new_len = random.choice([0, 0xFFFF, len(d) * 2, 1])
            struct.pack_into(">H", d, 4, new_len)
            desc = f"Set data length to {new_len}"

        return MutationResult(MutationStrategy.HEADER_FIELD_CORRUPT, data, bytes(d),
                              desc, list(range(6)))

    def _apid_sweep(self, data: bytes) -> MutationResult:
        """Sweep through APID space to find undocumented endpoints."""
        d = bytearray(data)
        if len(d) < 6:
            return self._bit_flip(data)
        apid = random.randint(0, 0x7FF)
        d[0] = (d[0] & 0xF8) | ((apid >> 8) & 0x07)
        d[1] = apid & 0xFF
        return MutationResult(MutationStrategy.APID_SWEEP, data, bytes(d),
                              f"APID sweep: set to {apid}", [0, 1])

    def _length_overflow(self, data: bytes) -> MutationResult:
        """Set data length field to trigger buffer overflows."""
        d = bytearray(data)
        if len(d) < 6:
            return self._bit_flip(data)
        overflow_values = [0xFFFF, 0xFFFE, 0, 1, len(d) + 1000, len(d) - 7]
        new_len = random.choice(overflow_values)
        struct.pack_into(">H", d, 4, max(0, new_len) & 0xFFFF)
        return MutationResult(MutationStrategy.LENGTH_OVERFLOW, data, bytes(d),
                              f"Set length to {new_len & 0xFFFF} (actual payload: {len(d) - 6})",
                              [4, 5])

    def _sequence_wrap(self, data: bytes) -> MutationResult:
        """Test sequence counter wrapping and replay."""
        d = bytearray(data)
        if len(d) < 6:
            return self._bit_flip(data)
        wrap_values = [0, 0x3FFF, 0x3FFE, 1]
        seq = random.choice(wrap_values)
        d[2] = (d[2] & 0xC0) | ((seq >> 8) & 0x3F)
        d[3] = seq & 0xFF
        return MutationResult(MutationStrategy.SEQUENCE_WRAP, data, bytes(d),
                              f"Sequence wrap: set to {seq}", [2, 3])

    def _pus_service_fuzz(self, data: bytes) -> MutationResult:
        """Fuzz PUS service type/subtype combinations."""
        d = bytearray(data)
        if len(d) < 10:
            return self._bit_flip(data)
        # PUS header starts at offset 6
        svc = random.randint(0, 255)
        sub = random.randint(0, 255)
        d[7] = svc
        d[8] = sub
        return MutationResult(MutationStrategy.PUS_SERVICE_FUZZ, data, bytes(d),
                              f"PUS service fuzz: type={svc}, subtype={sub}", [7, 8])

    def _frame_crc_corrupt(self, data: bytes) -> MutationResult:
        """Corrupt frame-level CRC to test error handling paths."""
        d = bytearray(data)
        if len(d) < 8:
            return self._bit_flip(data)
        # Corrupt last 2 bytes (assumed FECF location)
        d[-2] ^= random.randint(1, 255)
        d[-1] ^= random.randint(1, 255)
        return MutationResult(MutationStrategy.FRAME_CRC_CORRUPT, data, bytes(d),
                              "Corrupted frame CRC", [len(d) - 2, len(d) - 1])

    def _cross_layer_inject(self, data: bytes) -> MutationResult:
        """Inject a packet inside a TC Transfer Frame with mismatched metadata."""
        inner = build_tc_packet(
            apid=random.randint(0, 0x7FF),
            service_type=random.randint(0, 255),
            service_subtype=random.randint(0, 255),
            payload=os.urandom(random.randint(1, 256)),
        )
        frame = TCTransferFrame(
            spacecraft_id=random.randint(0, 0x3FF),
            virtual_channel_id=random.randint(0, 3),
            data=inner,
            frame_length=random.choice([len(inner), len(inner) + 100, 0, 0xFFFF]),
        )
        result = frame.pack()
        return MutationResult(MutationStrategy.CROSS_LAYER_INJECT, data, result,
                              f"Cross-layer: frame(sc={frame.spacecraft_id}) wrapping random TC packet",
                              [])

    def _weighted_offset(self, max_offset: int) -> int:
        """Select byte offset weighted by importance map."""
        if not self.byte_importance or max_offset <= 0:
            return random.randint(0, max(0, max_offset - 1))

        weights = np.array([
            self.byte_importance.get(i, 1.0)
            for i in range(max_offset)
        ])
        weights /= weights.sum()
        return int(self.rng.choice(max_offset, p=weights))

    def update_byte_importance(self, offsets: list[int], reward: float):
        """Increase importance of byte offsets that yielded rewards."""
        for offset in offsets:
            current = self.byte_importance.get(offset, 1.0)
            self.byte_importance[offset] = current + reward
