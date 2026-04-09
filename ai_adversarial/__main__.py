"""
Adversarial ML Attack Framework — CLI
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .attacks import (
    EvasionAttack, EvasionConfig, EvasionMethod,
    DataPoisonAttack, PoisonConfig, PoisonType,
    ModelExtractionAttack, ExtractionConfig,
    PromptInjectionGenerator,
    MembershipInferenceAttack,
    estimate_transferability,
)

console = Console()


@click.group()
def main():
    """Adversarial ML — Offensive toolkit for AI/ML systems."""
    pass


@main.command()
@click.option("--method", type=click.Choice(["fgsm", "pgd", "patch"]), default="fgsm")
@click.option("--epsilon", default=0.03, help="Perturbation budget")
@click.option("--steps", default=40, help="PGD iterations")
@click.option("--output", default="adversarial_output", help="Output directory")
def evasion(method, epsilon, steps, output):
    """Generate evasion attacks against image classifiers."""
    console.print(Panel.fit(
        f"[bold red]Evasion Attack Generator[/bold red]\n"
        f"[cyan]Method:[/cyan] {method.upper()}\n"
        f"[cyan]Epsilon:[/cyan] {epsilon}\n"
        f"[cyan]Steps:[/cyan] {steps}",
        border_style="red",
    ))

    config = EvasionConfig(
        method=EvasionMethod[method.upper()],
        epsilon=epsilon,
        num_steps=steps,
    )
    attack = EvasionAttack(config)

    # Demo: generate adversarial patch
    if method == "patch":
        patch = attack.generate_adversarial_patch()
        Path(output).mkdir(exist_ok=True)
        np.save(f"{output}/adversarial_patch.npy", patch)
        console.print(f"[green]✓ Adversarial patch saved ({patch.shape})[/green]")
    else:
        # Demo with random image
        dummy_image = np.random.uniform(0, 1, (224, 224, 3)).astype(np.float32)
        dummy_gradient = np.random.randn(224, 224, 3).astype(np.float32)

        if method == "fgsm":
            adv = attack.fgsm(dummy_image, dummy_gradient)
        else:
            adv = attack.pgd(dummy_image, lambda x: np.random.randn(*x.shape))

        perturbation = np.abs(adv - dummy_image)
        console.print(f"[green]✓ Adversarial example generated[/green]")
        console.print(f"  L-inf perturbation: {perturbation.max():.4f}")
        console.print(f"  L2 perturbation: {np.linalg.norm(perturbation):.4f}")
        console.print(f"  Mean perturbation: {perturbation.mean():.6f}")


@main.command()
@click.option("--category", default=None, help="Filter: direct, indirect, stored, context_overflow")
@click.option("--target", default=None, help="Filter: generic, rag, agent, tool_calling")
@click.option("--output-json", default=None, help="Export payloads to JSON")
def prompt_inject(category, target, output_json):
    """Generate prompt injection payloads for LLM-augmented systems."""
    gen = PromptInjectionGenerator()
    payloads = gen.get_payloads(category=category, target_system=target)

    table = Table(title="Prompt Injection Payload Catalog")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Target")
    table.add_column("Evasion")
    table.add_column("Preview", max_width=50)

    for p in payloads:
        table.add_row(
            p.name, p.category, p.target_system,
            p.evasion_technique or "-",
            p.payload[:50] + "...",
        )
    console.print(table)

    if output_json:
        data = [
            {
                "name": p.name,
                "category": p.category,
                "target_system": p.target_system,
                "payload": p.payload,
                "description": p.description,
                "evasion_technique": p.evasion_technique,
            }
            for p in payloads
        ]
        Path(output_json).write_text(json.dumps(data, indent=2))
        console.print(f"[green]Exported {len(data)} payloads to {output_json}[/green]")


@main.command()
@click.option("--dataset-size", default=50000, help="Training dataset size")
@click.option("--poison-rate", default=0.01, help="Fraction to poison")
@click.option("--method", type=click.Choice(["backdoor", "label_flip", "clean_label"]), default="backdoor")
def poison(dataset_size, poison_rate, method):
    """Generate data poisoning attack plans."""
    config = PoisonConfig(
        method=PoisonType[method.upper()],
        poison_rate=poison_rate,
    )
    attack = DataPoisonAttack(config)
    manifest = attack.generate_poison_manifest(dataset_size)

    console.print(Panel.fit(
        f"[bold red]Data Poisoning Plan[/bold red]\n"
        f"[cyan]Method:[/cyan] {manifest['method']}\n"
        f"[cyan]Dataset size:[/cyan] {dataset_size}\n"
        f"[cyan]Poison rate:[/cyan] {poison_rate * 100:.1f}%\n"
        f"[cyan]Samples to poison:[/cyan] {manifest['num_poisoned']}\n"
        f"[cyan]Trigger:[/cyan] {manifest['trigger_pattern']} ({manifest['trigger_size']})",
        border_style="red",
    ))


@main.command()
@click.option("--source", default="resnet50", help="Source model architecture")
@click.option("--target", default="vgg16", help="Target model architecture")
@click.option("--attack", default="PGD", help="Attack method")
def transfer(source, target, attack):
    """Analyze adversarial transferability between architectures."""
    result = estimate_transferability(source, target, attack)
    console.print(Panel.fit(
        f"[bold]Transferability Analysis[/bold]\n"
        f"[cyan]{result['source']}[/cyan] → [cyan]{result['target']}[/cyan]\n"
        f"[cyan]Attack:[/cyan] {result['attack_method']}\n"
        f"[cyan]Transfer rate:[/cyan] {result['estimated_transfer_rate']:.1%}\n"
        f"[yellow]{result['recommendation']}[/yellow]",
        border_style="cyan",
    ))


if __name__ == "__main__":
    main()
