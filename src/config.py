"""Configuration loader — reads config.yaml into validated dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration dataclasses (frozen = immutable after load)
# =============================================================================


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    external_dir: Path
    figures_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class PreprocessingConfig:
    raw_file_pattern: str
    output_file: str
    dedup_column: str
    core_fields: List[str]
    text_fields: List[str]
    fillna_values: Dict[str, str]
    encodings: List[str]


@dataclass(frozen=True)
class LogisticBounds:
    L: List[float]
    k: List[float]
    t0: List[float]


@dataclass(frozen=True)
class LogisticFitConfig:
    p0: List[float]
    bounds: LogisticBounds
    maxfev: int
    phase_thresholds: Dict[str, List[float]]
    phase_colors: Dict[str, str]
    phase_labels: Dict[str, str]


@dataclass(frozen=True)
class CountryAnalysisConfig:
    mangrove_area_file: str
    area_column: str
    area_unit: str
    world_map_source: str
    country_name_mapping: Dict[str, str]


@dataclass(frozen=True)
class UMAPConfig:
    n_neighbors: int
    n_components: int
    min_dist: float
    metric: str
    random_state: int


@dataclass(frozen=True)
class HDBSCANConfig:
    min_cluster_size: int
    min_samples: int
    metric: str


@dataclass(frozen=True)
class VectorizerConfig:
    pos_pattern: str
    backend: str = "count"  # "count" = pinned CountVectorizer; "spacy" = KeyphraseCountVectorizer


@dataclass(frozen=True)
class RepresentationConfig:
    top_n_words: int
    nr_repr_docs: int


@dataclass(frozen=True)
class TopicModelingOutputConfig:
    model_name: str
    topics_info: str
    doc_topic_dist: str
    visualize_html: str
    topic_word_scores: str
    topic_word_scores_plot: str


@dataclass(frozen=True)
class LLMLabelingConfig:
    enabled: bool
    provider: str
    model: str
    api_key_env: str
    base_url: Optional[str]
    delay_in_seconds: float
    nr_docs: int


@dataclass(frozen=True)
class CollaborationNetworkConfig:
    mangrove_shapefile: str
    world_map_source: str
    raster_res_deg: float
    top_n_edges: int
    min_pubs_for_label: int
    output_file: str


@dataclass(frozen=True)
class TopicModelingConfig:
    embedding_model: str
    spacy_model: str
    umap: UMAPConfig
    hdbscan: HDBSCANConfig
    vectorizer: VectorizerConfig
    representation: RepresentationConfig
    min_topic_size: int
    compute_coherence: bool
    llm_labeling: LLMLabelingConfig
    output: TopicModelingOutputConfig


@dataclass(frozen=True)
class TopicEvolutionConfig:
    time_periods: Optional[List[List[Optional[int]]]]
    period_labels: Optional[List[str]]
    min_topic_share: float
    output_file: str


@dataclass(frozen=True)
class TopicDynamicsOutputConfig:
    dynamics_html: str
    dynamics_csv: str
    keywords_evolution_html: str


@dataclass(frozen=True)
class TopicDynamicsConfig:
    n_bins: Optional[int]
    evolution_mode: str
    top_n_words: int
    min_docs_per_bin: int
    output: TopicDynamicsOutputConfig


@dataclass(frozen=True)
class TopicNetworkOutputConfig:
    network_html: str
    network_csv: str


@dataclass(frozen=True)
class TopicNetworkConfig:
    similarity_metric: str
    edge_threshold: float
    top_n_topics: int
    output: TopicNetworkOutputConfig


@dataclass(frozen=True)
class TopicSankeyConfig:
    temperature: float
    min_flow: float
    top_n_topics: int
    output_file: str


@dataclass(frozen=True)
class SpatialTopicAnalysisOutputConfig:
    si_matrix: str
    # ARCHIVED — moved to src/archived_viz/spatial_topic_heatmap.py
    # heatmap: str
    choropleth_html: str
    # ARCHIVED — moved to src/archived_viz/spatial_small_multiples.py
    # small_multiples: str
    # ARCHIVED — moved to src/archived_viz/spatial_dominant_topic_map.py
    # dominant_topic_map: str
    # ARCHIVED — moved to src/archived_viz/spatial_country_portfolio.py
    # country_portfolio: str
    top_topics_per_country: str
    # SUPERSEDED 2026-09-26: the 21 x 11 in 12-panel topic-popularity bar chart. It is
    # no longer produced by run_spatial_topic_analysis; the key stays so older configs
    # and notes keep loading.
    popular_topics_heatmap: str
    # Manuscript Fig. 7: the country x topic heatmap from src/country_topic_heatmap.py.
    country_topic_heatmap: str = "country_topic_heatmap.png"


@dataclass(frozen=True)
class SpatialTopicAnalysisConfig:
    min_docs_per_country: int
    top_n_countries: int
    top_n_topics: int
    min_si_threshold: float
    world_map_source: str
    output: SpatialTopicAnalysisOutputConfig


@dataclass(frozen=True)
class VisualizationConfig:
    dpi: int
    figure_format: str
    default_figsize: List[int]
    color_palette: str
    font_family: str


@dataclass(frozen=True)
class SystemConfig:
    log_level: str
    log_file: str
    prefer_gpu: bool
    random_seed: int


@dataclass(frozen=True)
class Config:
    """Root configuration holder. All fields are frozen dataclasses."""

    paths: PathsConfig
    preprocessing: PreprocessingConfig
    logistic_fit: LogisticFitConfig
    country_analysis: CountryAnalysisConfig
    collaboration_network: CollaborationNetworkConfig
    topic_modeling: TopicModelingConfig
    topic_evolution: TopicEvolutionConfig
    topic_dynamics: TopicDynamicsConfig
    topic_network: TopicNetworkConfig
    topic_sankey: TopicSankeyConfig
    spatial_topic_analysis: SpatialTopicAnalysisConfig
    visualization: VisualizationConfig
    system: SystemConfig
    project_root: Path = field(default_factory=Path.cwd)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str | Path = "config.yaml") -> "Config":
        """Load and validate configuration from a YAML file.

        Args:
            yaml_path: Path to config.yaml (absolute or relative to cwd).

        Returns:
            A frozen Config instance with all paths resolved.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If required sections or keys are missing.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        project_root = yaml_path.parent.resolve()

        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        _validate_sections(raw, yaml_path)

        # Resolve all paths relative to project root
        paths_raw = raw["paths"]

        config = cls(
            project_root=project_root,
            paths=PathsConfig(
                raw_dir=_resolve(paths_raw["raw_dir"], project_root),
                processed_dir=_resolve(paths_raw["processed_dir"], project_root),
                external_dir=_resolve(paths_raw["external_dir"], project_root),
                figures_dir=_resolve(paths_raw["figures_dir"], project_root),
                models_dir=_resolve(paths_raw["models_dir"], project_root),
            ),
            preprocessing=PreprocessingConfig(
                raw_file_pattern=raw["preprocessing"]["raw_file_pattern"],
                output_file=raw["preprocessing"]["output_file"],
                dedup_column=raw["preprocessing"]["dedup_column"],
                core_fields=raw["preprocessing"]["core_fields"],
                text_fields=raw["preprocessing"]["text_fields"],
                fillna_values=raw["preprocessing"].get("fillna_values", {}),
                encodings=raw["preprocessing"].get("encodings", ["utf-8-sig", "utf-8", "latin-1"]),
            ),
            logistic_fit=_parse_logistic(raw),
            country_analysis=CountryAnalysisConfig(
                mangrove_area_file=raw["country_analysis"]["mangrove_area_file"],
                area_column=raw["country_analysis"].get("area_column", "sum"),
                area_unit=raw["country_analysis"].get("area_unit", "km2"),
                world_map_source=raw["country_analysis"]["world_map_source"],
                country_name_mapping=raw["country_analysis"].get("country_name_mapping", {}),
            ),
            collaboration_network=_parse_collaboration_network(raw),
            topic_modeling=_parse_topic_modeling(raw),
            topic_evolution=_parse_topic_evolution(raw),
            topic_dynamics=_parse_topic_dynamics(raw),
            topic_network=_parse_topic_network(raw),
            topic_sankey=_parse_topic_sankey(raw),
            spatial_topic_analysis=_parse_spatial_topic_analysis(raw),
            visualization=VisualizationConfig(
                dpi=raw["visualization"]["dpi"],
                figure_format=raw["visualization"]["figure_format"],
                default_figsize=raw["visualization"]["default_figsize"],
                color_palette=raw["visualization"]["color_palette"],
                font_family=raw["visualization"].get("font_family", "DejaVu Sans"),
            ),
            system=SystemConfig(
                log_level=raw["system"].get("log_level", "INFO"),
                log_file=raw["system"].get("log_file", "outputs/pipeline.log"),
                prefer_gpu=raw["system"].get("prefer_gpu", True),
                random_seed=raw["system"].get("random_seed", 42),
            ),
        )

        logger.info("Configuration loaded from %s", yaml_path)
        logger.debug("Project root: %s", config.project_root)
        return config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_path(self, relative_path: str | Path) -> Path:
        """Resolve a relative path against the project root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.project_root / p


# =============================================================================
# Internal helpers
# =============================================================================


def _resolve(rel: str, root: Path) -> Path:
    """Resolve a relative path string to an absolute Path, creating dir if needed."""
    p = (root / rel).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _validate_sections(raw: Dict[str, Any], yaml_path: Path) -> None:
    """Ensure all required top-level sections are present."""
    required = [
        "paths",
        "preprocessing",
        "logistic_fit",
        "country_analysis",
        "topic_modeling",
        "topic_evolution",
        "visualization",
        "system",
    ]
    missing = [s for s in required if s not in raw]
    if missing:
        raise ValueError(
            f"Missing required section(s) in {yaml_path}: {missing}. "
            f"Check that config.yaml has all expected sections."
        )


def _parse_logistic(raw: Dict[str, Any]) -> LogisticFitConfig:
    lf = raw["logistic_fit"]
    b = lf.get("bounds", {})
    return LogisticFitConfig(
        p0=lf.get("p0", [6000, 0.15, 35]),
        bounds=LogisticBounds(
            L=b.get("L", [2004, 8000]),
            k=b.get("k", [0.05, 0.50]),
            t0=b.get("t0", [20, 60]),
        ),
        maxfev=lf.get("maxfev", 10000),
        phase_thresholds=lf.get("phase_thresholds", {}),
        phase_colors=lf.get("phase_colors", {}),
        phase_labels=lf.get("phase_labels", {}),
    )


def _parse_collaboration_network(raw: Dict[str, Any]) -> CollaborationNetworkConfig:
    cn = raw.get("collaboration_network", {})
    return CollaborationNetworkConfig(
        mangrove_shapefile=cn.get("mangrove_shapefile", ""),
        world_map_source=cn.get(
            "world_map_source",
            "data/external/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp",
        ),
        raster_res_deg=cn.get("raster_res_deg", 0.5),
        top_n_edges=cn.get("top_n_edges", 40),
        min_pubs_for_label=cn.get("min_pubs_for_label", 15),
        output_file=cn.get("output_file", "collaboration_network_map.png"),
    )


def _parse_topic_modeling(raw: Dict[str, Any]) -> TopicModelingConfig:
    tm = raw["topic_modeling"]
    llm = tm.get("llm_labeling", {})
    return TopicModelingConfig(
        embedding_model=tm["embedding_model"],
        spacy_model=tm["spacy_model"],
        umap=UMAPConfig(**tm["umap"]),
        hdbscan=HDBSCANConfig(**tm["hdbscan"]),
        vectorizer=VectorizerConfig(**tm["vectorizer"]),
        representation=RepresentationConfig(**tm["representation"]),
        min_topic_size=tm.get("min_topic_size", 10),
        compute_coherence=tm.get("compute_coherence", True),
        llm_labeling=LLMLabelingConfig(
            enabled=llm.get("enabled", False),
            provider=llm.get("provider", "deepseek"),
            model=llm.get("model", "deepseek-v4-pro"),
            api_key_env=llm.get("api_key_env", "DEEPSEEK_API_KEY"),
            base_url=llm.get("base_url"),
            delay_in_seconds=llm.get("delay_in_seconds", 2),
            nr_docs=llm.get("nr_docs", 4),
        ),
        output=TopicModelingOutputConfig(
            model_name=tm["output"]["model_name"],
            topics_info=tm["output"]["topics_info"],
            doc_topic_dist=tm["output"]["doc_topic_dist"],
            visualize_html=tm["output"]["visualize_html"],
            topic_word_scores=tm["output"].get("topic_word_scores", "topic_word_scores.csv"),
            topic_word_scores_plot=tm["output"].get("topic_word_scores_plot", "topic_word_scores.png"),
        ),
    )


def _parse_topic_evolution(raw: Dict[str, Any]) -> TopicEvolutionConfig:
    te = raw["topic_evolution"]
    return TopicEvolutionConfig(
        time_periods=te.get("time_periods"),
        period_labels=te.get("period_labels"),
        min_topic_share=te.get("min_topic_share", 0.01),
        output_file=te.get("output_file", "topic_evolution.html"),
    )


def _parse_spatial_topic_analysis(raw: Dict[str, Any]) -> SpatialTopicAnalysisConfig:
    sa = raw.get("spatial_topic_analysis", {})
    out = sa.get("output", {})
    return SpatialTopicAnalysisConfig(
        min_docs_per_country=sa.get("min_docs_per_country", 5),
        top_n_countries=sa.get("top_n_countries", 25),
        top_n_topics=sa.get("top_n_topics", 12),
        min_si_threshold=sa.get("min_si_threshold", 1.0),
        world_map_source=sa.get(
            "world_map_source",
            "data/external/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp",
        ),
        output=SpatialTopicAnalysisOutputConfig(
            si_matrix=out.get("si_matrix", "spatial_topic_si.csv"),
            # heatmap=out.get("heatmap", "spatial_topic_heatmap.png"),  # ARCHIVED
            choropleth_html=out.get("choropleth_html", "spatial_topic_choropleth.html"),
            # small_multiples=out.get("small_multiples", "spatial_topic_small_multiples.png"),  # ARCHIVED
            # dominant_topic_map=out.get("dominant_topic_map", "spatial_dominant_topic_map.png"),  # ARCHIVED
            # country_portfolio=out.get("country_portfolio", "spatial_country_portfolio.png"),  # ARCHIVED
            top_topics_per_country=out.get("top_topics_per_country", "top_topics_per_country.png"),
            popular_topics_heatmap=out.get("popular_topics_heatmap", "popular_topics_heatmap.png"),
            country_topic_heatmap=out.get(
                "country_topic_heatmap", "country_topic_heatmap.png"
            ),
        ),
    )


def _parse_topic_dynamics(raw: Dict[str, Any]) -> TopicDynamicsConfig:
    td = raw.get("topic_dynamics", {})
    out = td.get("output", {})
    return TopicDynamicsConfig(
        n_bins=td.get("n_bins"),
        evolution_mode=td.get("evolution_mode", "relative"),
        top_n_words=td.get("top_n_words", 10),
        min_docs_per_bin=td.get("min_docs_per_bin", 5),
        output=TopicDynamicsOutputConfig(
            dynamics_html=out.get("dynamics_html", "topic_dynamics.html"),
            dynamics_csv=out.get("dynamics_csv", "topic_dynamics.csv"),
            keywords_evolution_html=out.get("keywords_evolution_html", "topic_keywords_evolution.html"),
        ),
    )


def _parse_topic_network(raw: Dict[str, Any]) -> TopicNetworkConfig:
    tn = raw.get("topic_network", {})
    out = tn.get("output", {})
    return TopicNetworkConfig(
        similarity_metric=tn.get("similarity_metric", "cosine"),
        edge_threshold=tn.get("edge_threshold", 0.15),
        top_n_topics=tn.get("top_n_topics", 20),
        output=TopicNetworkOutputConfig(
            network_html=out.get("network_html", "topic_network.html"),
            network_csv=out.get("network_csv", "topic_network_edges.csv"),
        ),
    )


def _parse_topic_sankey(raw: Dict[str, Any]) -> TopicSankeyConfig:
    """Parse optional ``topic_sankey`` section with sensible defaults."""
    ts = raw.get("topic_sankey", {})
    return TopicSankeyConfig(
        temperature=ts.get("temperature", 0.3),
        min_flow=ts.get("min_flow", 0.002),
        top_n_topics=ts.get("top_n_topics", 12),
        output_file=ts.get("output_file", "topic_sankey.html"),
    )


# =============================================================================
# Module-level singleton — loaded once on first import
# =============================================================================

_CONFIG: Optional[Config] = None


def get_config(config_path: str | Path = "config.yaml") -> Config:
    """Return the cached Config, loading it from disk on the first call."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config.from_yaml(config_path)
    return _CONFIG


def reload_config(config_path: str | Path = "config.yaml") -> Config:
    """Force-reload configuration from disk (useful in notebooks)."""
    global _CONFIG
    _CONFIG = Config.from_yaml(config_path)
    return _CONFIG
