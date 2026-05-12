"""
UniProt accessions that return empty FASTA at rest.uniprot.org (inactive/deleted).

Used to prune ground-truth JSON so accession lists match fetchable sequences.
"""

from __future__ import annotations

# Observed 2025-05 from fetch_sequences on PAZy / merged positive pools.
DEAD_UNIPROT_ACCESSIONS: frozenset[str] = frozenset(
    {
        "A0A0N0MY27",
        "A0A0N0NEY5",
        "A0A1E4LW26",
        "A0A1F4JXW8",
        "A0A1H6AD45",
        "A0A2H5Z9R5",
        "A0A3L8BDT3",
        "A0A3L8BW54",
        "A0A497NK85",
        "A0A7I8E2Z4",
        "A0A7T4K057",
        "A4UZ10",
        "A4UZ11",
        "A4UZ12",
        "A4UZ13",
        "A4UZ14",
        "C3RYL0",
        "E9UPM2",
        "T1W006",
        "T1W153",
    }
)
