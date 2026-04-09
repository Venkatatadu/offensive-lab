"""
CCSDS Space Protocol Fuzzer — Main Entry Point
================================================
AI-guided fuzzer for space ground station software.
Sends mutated CCSDS packets over TCP/UDP and monitors for crashes.

Usage:
    python -m satcom_fuzzer --target localhost:9999 --mode ai-guided --frames 10000
    python -m satcom_fuzzer --target localhost:9999 --mode dumb --protocol udp
    python -m satcom_fuzzer --generate-corpus --output seeds/
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

from .ccsds import build_tc_packet, build_tm_packet, TCTransferFrame
from .mutator import AIGuidedMutator, MutationStrategy, CoverageTracker

console = Console()


def send_payload(target: str, data: bytes, protocol: str = "tcp", timeout: float = 2.0) -> dict:
    """Send a fuzzed payload and return result metadata."""
    host, port = target.rsplit(":", 1)
    port = int(port)
    result = {"sent": len(data), "crash": False, "error": None, "response": b""}

    try:
        if protocol == "tcp":
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, port))
                s.sendall(data)
                try:
                    result["response"] = s.recv(4096)
                except socket.timeout:
                    pass
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(timeout)
                s.sendto(data, (host, port))
                try:
                    result["response"], _ = s.recvfrom(4096)
                except socket.timeout:
                    pass
    except ConnectionRefusedError:
        result["error"] = "connection_refused"
        result["crash"] = True  # Potential crash indicator
    except ConnectionResetError:
        result["error"] = "connection_reset"
        result["crash"] = True
    except BrokenPipeError:
        result["error"] = "broken_pipe"
        result["crash"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


def make_stats_table(mutator: AIGuidedMutator, elapsed: float, total: int) -> Table:
    """Build a rich table showing mutation strategy performance."""
    table = Table(title="🛰️  SATCOM Fuzzer — Strategy Performance")
    table.add_column("Strategy", style="cyan")
    table.add_column("Uses", justify="right")
    table.add_column("Reward", justify="right", style="green")
    table.add_column("Avg", justify="right")
    table.add_column("α", justify="right", style="yellow")
    table.add_column("β", justify="right", style="red")

    for stat in mutator.get_strategy_stats()[:10]:
        table.add_row(
            stat["strategy"],
            str(stat["uses"]),
            str(stat["reward"]),
            str(stat["avg_reward"]),
            str(stat["alpha"]),
            str(stat["beta"]),
        )

    return table


@click.command()
@click.option("--target", default="localhost:9999", help="Target host:port")
@click.option("--protocol", type=click.Choice(["tcp", "udp"]), default="tcp")
@click.option("--mode", type=click.Choice(["ai-guided", "dumb", "corpus-only"]), default="ai-guided")
@click.option("--frames", default=10000, help="Number of frames to send")
@click.option("--batch-size", default=50, help="Batch size for AI-guided mode")
@click.option("--timeout", default=2.0, help="Socket timeout in seconds")
@click.option("--output", default="fuzz_output", help="Output directory for crashes")
@click.option("--generate-corpus", is_flag=True, help="Generate seed corpus and exit")
@click.option("--dry-run", is_flag=True, help="Generate and display payloads without sending")
def main(target, protocol, mode, frames, batch_size, timeout, output, generate_corpus, dry_run):
    """SATCOM Fuzzer — AI-guided CCSDS protocol fuzzer."""

    console.print(Panel.fit(
        "[bold red]SATCOM FUZZER[/bold red]\n"
        "[dim]AI-Guided CCSDS Space Protocol Fuzzer[/dim]\n"
        f"[cyan]Target:[/cyan] {target} ({protocol})\n"
        f"[cyan]Mode:[/cyan] {mode}\n"
        f"[cyan]Frames:[/cyan] {frames}",
        border_style="red",
    ))

    mutator = AIGuidedMutator()
    crash_dir = Path(output) / "crashes"
    corpus_dir = Path(output) / "corpus"
    crash_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if generate_corpus:
        console.print("[yellow]Generating seed corpus...[/yellow]")
        for i, seed in enumerate(mutator.seed_corpus):
            path = corpus_dir / f"seed_{i:04d}.bin"
            path.write_bytes(seed)
            console.print(f"  [green]✓[/green] {path} ({len(seed)} bytes)")
        console.print(f"\n[bold green]Generated {len(mutator.seed_corpus)} seeds.[/bold green]")
        return

    start_time = time.time()
    total_sent = 0
    total_crashes = 0
    crash_log = []

    console.print(f"\n[yellow]Starting fuzzing campaign — {frames} frames[/yellow]\n")

    for i in range(0, frames, batch_size):
        current_batch = min(batch_size, frames - i)
        batch = mutator.generate_batch(current_batch)

        for payload, strategy in batch:
            total_sent += 1

            if dry_run:
                console.print(f"[dim]#{total_sent}[/dim] [{strategy.name}] {len(payload)} bytes: {payload[:32].hex()}...")
                continue

            result = send_payload(target, payload, protocol, timeout)

            # Simulate coverage feedback (in real usage, instrument via AFL/libfuzzer)
            fake_edges = set()
            if result["response"]:
                fake_edges = {hash(result["response"][:8]) % 65536}
            new_edges = mutator.coverage.update_coverage(fake_edges)

            is_crash = result["crash"]
            if is_crash:
                total_crashes += 1
                crash_path = crash_dir / f"crash_{total_crashes:06d}_{strategy.name}.bin"
                crash_path.write_bytes(payload)
                crash_log.append({
                    "id": total_crashes,
                    "strategy": strategy.name,
                    "size": len(payload),
                    "error": result["error"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "file": str(crash_path),
                })

            mutator.report_result(strategy, new_edges, is_crash)
            mutator.coverage.total_inputs += 1

        # Progress update
        elapsed = time.time() - start_time
        rate = total_sent / max(elapsed, 0.01)
        console.print(
            f"  [cyan]Progress:[/cyan] {total_sent}/{frames} "
            f"({rate:.0f}/s) | "
            f"[red]Crashes: {total_crashes}[/red] | "
            f"[green]Coverage: {len(mutator.coverage.coverage_bitmap)} edges[/green]"
        )

    # Final report
    elapsed = time.time() - start_time
    console.print("\n")
    console.print(make_stats_table(mutator, elapsed, total_sent))

    console.print(Panel.fit(
        f"[bold]Campaign Complete[/bold]\n"
        f"[cyan]Duration:[/cyan] {elapsed:.1f}s\n"
        f"[cyan]Total sent:[/cyan] {total_sent}\n"
        f"[cyan]Rate:[/cyan] {total_sent / max(elapsed, 0.01):.0f} packets/s\n"
        f"[red]Crashes:[/red] {total_crashes}\n"
        f"[green]Coverage edges:[/green] {len(mutator.coverage.coverage_bitmap)}",
        border_style="green",
    ))

    # Save crash log
    if crash_log:
        log_path = Path(output) / "crash_log.json"
        log_path.write_text(json.dumps(crash_log, indent=2))
        console.print(f"\n[yellow]Crash log saved to {log_path}[/yellow]")


if __name__ == "__main__":
    main()
