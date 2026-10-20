"""Pipeline Orchestrator — runs the complete bibliometric analysis pipeline.

Usage:
    python src/run_pipeline.py                     # Run ALL steps
    python src/run_pipeline.py --step preprocess    # Run single step
    python src/run_pipeline.py --step bibliometric
    python src/run_pipeline.py --step topics
    python src/run_pipeline.py --step spatial
    python src/run_pipeline.py --step evolution
    python src/run_pipeline.py --step dynamics      # Dynamic topic modeling
    python src/run_pipeline.py --step network       # Topic similarity network
    python src/run_pipeline.py --from bibliometric  # Run from this step onward
    python src/run_pipeline.py --dry-run            # Show what would run
    python src/run_pipeline.py --force              # Force re-run even if outputs exist
    python src/run_pipeline.py --cpu                # Force CPU for topic modeling

Step dependencies:
    preprocess   → [none]
    bibliometric → [preprocess]
    topics       → [preprocess]
    spatial      → [preprocess, topics]
    evolution    → [preprocess, topics]
    dynamics     → [topics]
    network      → [topics]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.config import Config, get_config
from src.utils import setup_logging, Timer

logger = logging.getLogger(__name__)

# =============================================================================
# Step registry
# =============================================================================

STEPS: Dict[str, dict] = {
    "preprocess": {
        "name": "Data Preprocessing",
        "deps": [],
        "input_files": [],
        "output_files": ["data/processed/cleaned_data.csv"],
    },
    "bibliometric": {
        "name": "Bibliometric Analysis",
        "deps": ["preprocess"],
        "input_files": ["data/processed/cleaned_data.csv"],
        "output_files": [
            "outputs/figures/bibliometric_dynamics_annual_vs_cumulative_with_logistic_fit.png",
            "outputs/figures/country_map.png",
        ],
    },
    "collaboration": {
        "name": "Collaboration Network Map",
        "deps": ["preprocess"],
        "input_files": ["data/processed/cleaned_data.csv"],
        "output_files": ["outputs/figures/collaboration_network_map.png"],
    },
    "topics": {
        "name": "Topic Modeling",
        "deps": ["preprocess"],
        "input_files": ["data/processed/cleaned_data.csv"],
        "output_files": [
            "data/processed/topics_info.csv",
            "data/processed/doc_topic_dist.csv",
            "outputs/figures/visualize_topics.html",
        ],
    },
    "spatial": {
        "name": "Spatial Topic Analysis",
        "deps": ["preprocess", "topics"],
        "input_files": [
            "data/processed/cleaned_data.csv",
            "data/processed/doc_topic_dist.csv",
        ],
        "output_files": [
            "data/processed/spatial_topic_si.csv",
            "outputs/figures/spatial_topic_choropleth.html",
            "outputs/figures/top_topics_per_country.png",
        ],
    },
    "evolution": {
        "name": "Topic Evolution",
        "deps": ["preprocess", "topics"],
        "input_files": [
            "data/processed/cleaned_data.csv",
            "data/processed/doc_topic_dist.csv",
        ],
        "output_files": ["outputs/figures/topic_evolution.html"],
    },
    "dynamics": {
        "name": "Topic Dynamics",
        "deps": ["topics"],
        "input_files": [
            "data/processed/cleaned_data.csv",
            "outputs/models/bertopic_model",
        ],
        "output_files": [
            "outputs/figures/topic_dynamics.html",
            "outputs/figures/topic_keywords_evolution.html",
            "data/processed/topic_dynamics.csv",
        ],
    },
    "network": {
        "name": "Topic Network",
        "deps": ["topics"],
        "input_files": ["outputs/models/bertopic_model"],
        "output_files": [
            "outputs/figures/topic_network.html",
            "data/processed/topic_network_edges.csv",
        ],
    },
    "keywords": {
        "name": "Keyword Co-occurrence Network",
        "deps": ["preprocess"],
        "input_files": ["data/processed/cleaned_data.csv"],
        "output_files": ["outputs/figures/keyword_network.html"],
    },
    "sankey": {
        "name": "Topic Sankey Diagram",
        "deps": ["topics", "evolution"],
        "input_files": [
            "data/processed/topic_evolution_proportions.csv",
            "data/processed/topics_info.csv",
            "outputs/models/bertopic_model",
        ],
        "output_files": ["outputs/figures/topic_sankey.html"],
    },
    "combined_viz": {
        "name": "Combined Visualizations (P0-P2)",
        "deps": ["preprocess", "topics", "spatial", "bibliometric"],
        "input_files": [
            "data/processed/cleaned_data.csv",
            "data/processed/doc_topic_dist.csv",
            "data/processed/spatial_topic_si.csv",
            "data/external/HGMF_country_area.csv",
        ],
        "output_files": [
            "outputs/figures/country_area_pubs_bubble.png",
            "outputs/figures/country_area_pubs_bubble_polar.png",
            "outputs/figures/topic_year_heatmap.png",
            "outputs/figures/topic_streamgraph_yearly.html",
            "outputs/figures/topic_lifecycle.png",
        ],
    },
    "emerging": {
        "name": "Emerging Technology Prototype Detection",
        "deps": ["topics"],
        "input_files": [
            "data/processed/doc_topic_dist.csv",
            "data/processed/cleaned_data.csv",
            "outputs/models/bertopic_model",
        ],
        "output_files": [
            "data/processed/emerging_tech_prototypes.csv",
            "data/processed/emerging_window_sensitivity.csv",
            "data/processed/emerging_window_sensitivity_summary.json",
            "outputs/figures/emerging_tech_burst_index.png",
            "outputs/figures/emerging_tech_trajectories.png",
            "outputs/figures/emerging_tech_radar.png",
        ],
    },
}


def _resolve_outputs(output_files: List[str], project_root: Path) -> List[Path]:
    return [project_root / f for f in output_files]


def _resolve_inputs(input_files: List[str], project_root: Path) -> List[Path]:
    return [project_root / f for f in input_files]


# =============================================================================
# Pipeline logic
# =============================================================================


def check_outputs_exist(
    output_files: List[str],
    project_root: Path,
) -> bool:
    """Return True if all output files exist."""
    paths = _resolve_outputs(output_files, project_root)
    return all(p.exists() for p in paths)


def check_outputs_fresh(
    output_files: List[str],
    input_files: List[str],
    project_root: Path,
) -> bool:
    """Return True if all outputs are newer than all inputs."""
    outputs = _resolve_outputs(output_files, project_root)
    inputs = _resolve_inputs(input_files, project_root)

    if not all(p.exists() for p in outputs):
        return False

    max_input_mtime = max((p.stat().st_mtime for p in inputs if p.exists()), default=0)
    min_output_mtime = min(p.stat().st_mtime for p in outputs)
    return min_output_mtime > max_input_mtime


def resolve_steps(
    step: Optional[str] = None,
    from_step: Optional[str] = None,
) -> List[str]:
    """Resolve which steps to run.

    Args:
        step: Single step name to run.
        from_step: Run from this step through the end.

    Returns:
        Ordered list of step keys.
    """
    step_order = ["preprocess", "bibliometric", "collaboration", "topics", "spatial", "evolution", "dynamics", "network", "sankey", "keywords", "combined_viz", "emerging"]

    if step:
        if step not in STEPS:
            raise ValueError(f"Unknown step: {step}. Choose from: {list(STEPS)}")
        return [step]

    if from_step:
        if from_step not in STEPS:
            raise ValueError(f"Unknown step: {from_step}. Choose from: {list(STEPS)}")
        idx = step_order.index(from_step)
        return step_order[idx:]

    # Default: all steps
    return step_order


def validate_dependencies(requested: List[str]) -> None:
    """Check that requested step dependencies are satisfied (warn only)."""
    all_requested = set(requested)
    for step_key in requested:
        for dep in STEPS[step_key]["deps"]:
            if dep not in all_requested:
                logger.warning(
                    "Step '%s' depends on '%s' but '%s' is not in the run list. "
                    "This may fail if '%s' outputs don't exist.",
                    step_key, dep, dep, dep,
                )


# =============================================================================
# Runner functions
# =============================================================================


def run_step_preprocess(config: Config) -> None:
    from src.data_preprocessing import preprocess

    preprocess(config)


def run_step_bibliometric(config: Config) -> None:
    from src.bibliometric_analysis import run_bibliometric

    run_bibliometric(config)


def run_step_collaboration(config: Config) -> None:
    from src.collaboration_network_map import run_collaboration_network

    run_collaboration_network(config)


def run_step_topics(config: Config, force_cpu: bool = False) -> None:
    from src.topic_modeling import run_topic_modeling

    run_topic_modeling(config, force_cpu=force_cpu)


def run_step_spatial(config: Config) -> None:
    from src.spatial_topic_analysis import run_spatial_topic_analysis

    run_spatial_topic_analysis(config)


def run_step_evolution(config: Config) -> None:
    from src.topic_evolution import run_topic_evolution

    run_topic_evolution(config)


def run_step_dynamics(config: Config) -> None:
    from src.topic_dynamics import run_topic_dynamics

    run_topic_dynamics(config)


def run_step_network(config: Config) -> None:
    from src.topic_network import run_topic_network

    run_topic_network(config)


def run_step_keywords(config: Config) -> None:
    from src.keyword_network import run_keyword_network

    run_keyword_network(config)


def run_step_sankey(config: Config) -> None:
    from src.topic_sankey import run_topic_sankey

    run_topic_sankey(config)


def run_step_combined_viz(config: Config) -> None:
    from src.combined_viz import run_combined_viz

    run_combined_viz(config)


def run_step_emerging(config: Config) -> None:
    from src.emerging_tech_detection import run_emerging_tech_detection

    run_emerging_tech_detection(config)


STEP_RUNNERS: Dict[str, Callable] = {
    "preprocess": run_step_preprocess,
    "bibliometric": run_step_bibliometric,
    "collaboration": run_step_collaboration,
    "topics": run_step_topics,
    "spatial": run_step_spatial,
    "evolution": run_step_evolution,
    "dynamics": run_step_dynamics,
    "network": run_step_network,
    "keywords": run_step_keywords,
    "combined_viz": run_step_combined_viz,
    "emerging": run_step_emerging,
    "sankey": run_step_sankey,
}


# =============================================================================
# Main
# =============================================================================


def run_pipeline(
    config: Config,
    steps: Optional[List[str]] = None,
    from_step: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
    force_cpu: bool = False,
) -> None:
    """Execute the pipeline.

    Args:
        config: Validated configuration.
        steps: Specific step(s) to run (overrides from_step).
        from_step: Run from this step onward.
        dry_run: Print what would run without executing.
        force: Re-run even if outputs are fresh.
        force_cpu: Use CPU for topic modeling even if GPU available.
    """
    step_keys = resolve_steps(step=steps[0] if steps else None, from_step=from_step)
    validate_dependencies(step_keys)

    logger.info("=" * 60)
    logger.info("Pipeline: Mangrove RS Bibliometric Analysis")
    logger.info("Project root: %s", config.project_root)
    logger.info("Steps to run: %s", " → ".join(step_keys))
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would execute: %s", step_keys)
        for sk in step_keys:
            info = STEPS[sk]
            logger.info("  %s:", info["name"])
            logger.info("    Dependencies: %s", info["deps"] or "none")
            logger.info("    Inputs:  %s", info["input_files"] or "none")
            logger.info("    Outputs: %s", info["output_files"])
        return

    # Check for fresh outputs
    skipped = []
    executed = []
    for sk in step_keys:
        info = STEPS[sk]
        if not force and check_outputs_fresh(info["output_files"], info["input_files"], config.project_root):
            logger.info("[SKIP] %s — outputs are up-to-date. Use --force to re-run.", info["name"])
            skipped.append(sk)
        else:
            executed.append(sk)

    if skipped:
        logger.info("Skipping %d step(s) with fresh outputs: %s", len(skipped), skipped)
    if not executed:
        logger.info("All steps are up-to-date. Nothing to do.")
        return

    # Execute steps in order
    failed = []
    for sk in executed:
        info = STEPS[sk]
        runner = STEP_RUNNERS.get(sk)

        if runner is None:
            logger.error("No runner registered for step '%s'", sk)
            failed.append(sk)
            continue

        try:
            with Timer(f"Step — {info['name']}"):
                if sk == "topics":
                    runner(config, force_cpu=force_cpu)
                else:
                    runner(config)
        except Exception:
            logger.exception("Step '%s' FAILED", info["name"])
            failed.append(sk)
            # Don't stop — continue with remaining steps if possible
            logger.warning("Continuing with remaining steps despite failure in '%s'", sk)

    # Summary
    logger.info("=" * 60)
    n_ok = len(executed) - len(failed)
    logger.info("Pipeline complete: %d succeeded, %d failed, %d skipped", n_ok, len(failed), len(skipped))
    if failed:
        logger.error("Failed steps: %s", failed)
        sys.exit(1)
    else:
        logger.info("All steps completed successfully.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mangrove RS Bibliometric Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/run_pipeline.py                     # Run all steps
    python src/run_pipeline.py --step preprocess    # Only preprocess
    python src/run_pipeline.py --step spatial       # Only spatial analysis
    python src/run_pipeline.py --step dynamics      # Only dynamic topic modeling
    python src/run_pipeline.py --step network       # Only topic network
    python src/run_pipeline.py --from topics        # From topic modeling onward
    python src/run_pipeline.py --dry-run            # Preview what will run
    python src/run_pipeline.py --force              # Force re-run all steps
    python src/run_pipeline.py --cpu                # Force CPU mode
        """,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--step", type=str, choices=list(STEPS), help="Run a single step")
    parser.add_argument("--from", dest="from_step", type=str, choices=list(STEPS), help="Run from this step onward")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--force", action="store_true", help="Force re-run even if outputs are up-to-date")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode for topic modeling")
    args = parser.parse_args()

    if args.step and args.from_step:
        parser.error("Cannot use both --step and --from together.")

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)

    run_pipeline(
        config=config,
        steps=[args.step] if args.step else None,
        from_step=args.from_step,
        dry_run=args.dry_run,
        force=args.force,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
