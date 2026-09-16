"""Canonical UniProt feature taxonomy used by the H1/H2 supervision heads."""

RESIDUE_CORE_LABELS = (
    "helix", "strand", "turn", "disulfide bond", "glycosylation site",
    "active site", "binding site", "modified residue", "short sequence motif",
)
DOMAIN_CORE_LABELS = (
    "domain", "region of interest", "repeat", "zinc finger region",
    "dna-binding region", "coiled-coil region", "transmembrane region",
    "topological domain", "signal peptide", "transit peptide",
)

RESIDUE_EXPANDED_LABELS = (
    "site", "initiator methionine", "cross-link", "lipidation",
)
DOMAIN_EXPANDED_LABELS = (
    "compositional bias", "propeptide", "peptide", "intramembrane",
)

# Non-core point/residue annotations are pooled so rare labels still supervise H1.
RESIDUE_OTHER_TYPES = frozenset({
    "sequence conflict", "mutagenesis", "natural variant",
    "non-terminal residue", "sequence uncertainty", "non-adjacent residues",
    "non-standard residue",
})

# Non-core interval annotations are pooled so rare labels still supervise H2.
DOMAIN_OTHER_TYPES = frozenset({
    "alternative sequence",
})

# UniProt's chain annotation is almost universal entry-boundary metadata, not a
# discriminative residue/domain function target.
EXCLUDED_FEATURE_TYPES = frozenset({"chain"})

RESIDUE_LABELS = RESIDUE_CORE_LABELS + RESIDUE_EXPANDED_LABELS + ("other",)
DOMAIN_LABELS = DOMAIN_CORE_LABELS + DOMAIN_EXPANDED_LABELS + ("other",)
RESIDUE_SOURCE_TYPES = frozenset(RESIDUE_CORE_LABELS + RESIDUE_EXPANDED_LABELS) | RESIDUE_OTHER_TYPES
DOMAIN_SOURCE_TYPES = frozenset(DOMAIN_CORE_LABELS + DOMAIN_EXPANDED_LABELS) | DOMAIN_OTHER_TYPES
SUPPORTED_FEATURE_TYPES = RESIDUE_SOURCE_TYPES | DOMAIN_SOURCE_TYPES


def map_feature_type(feature_type: str, level: str) -> str | None:
    """Map a normalized UniProt feature type to an H1/H2 output label."""
    name = feature_type.strip().lower()
    if level == "residue":
        if name in RESIDUE_CORE_LABELS or name in RESIDUE_EXPANDED_LABELS:
            return name
        return "other" if name in RESIDUE_OTHER_TYPES else None
    if level == "domain":
        if name in DOMAIN_CORE_LABELS or name in DOMAIN_EXPANDED_LABELS:
            return name
        return "other" if name in DOMAIN_OTHER_TYPES else None
    raise ValueError(f"unknown feature level: {level}")
