"""
Corpus preparation for Vertex AI Search knowledge base.

Prepares and uploads Pt/Pd printing knowledge documents to Google Cloud
Storage for ingestion into a Vertex AI Search data store. Handles:
- Repository documentation (markdown files)
- Codebase documentation (Python modules with docstrings)
- Domain-specific knowledge (chemistry, paper profiles, troubleshooting)

Usage:
    from ptpd_calibration.vertex.corpus import CorpusPreparator

    preparator = CorpusPreparator(repo_path="./", output_dir="./corpus_staging")
    preparator.prepare_all()
    preparator.upload_to_gcs("gs://ptpd-knowledge-corpus")
"""

from __future__ import annotations

import logging
from pathlib import Path

from ptpd_calibration.config import get_settings

logger = logging.getLogger(__name__)


class CorpusPreparator:
    """Prepares Pt/Pd knowledge corpus for Vertex AI Search ingestion.

    Args:
        repo_path: Path to the repository root.
        output_dir: Local staging directory for prepared documents.
    """

    def __init__(
        self,
        repo_path: str | Path = ".",
        output_dir: str | Path | None = None,
    ):
        self.repo_path = Path(repo_path)
        settings = get_settings().vertex
        self.output_dir = Path(output_dir or settings.corpus_local_staging) / "documents"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "CorpusPreparator initialized: repo_path=%s, output_dir=%s",
            self.repo_path,
            self.output_dir,
        )

    def prepare_all(self) -> int:
        """Prepare all corpus documents.

        Returns:
            Total number of documents prepared.
        """
        count = 0
        count += self.prepare_repo_docs()
        count += self.prepare_code_docs()
        count += self.prepare_domain_knowledge()
        logger.info("Prepared %d total corpus documents in %s", count, self.output_dir)
        return count

    def prepare_repo_docs(self) -> int:
        """Convert repository markdown docs into searchable documents.

        Returns:
            Number of documents prepared.
        """
        doc_files = [
            "ARCHITECTURE.md",
            "DEEP_LEARNING_IMPLEMENTATION.md",
            "ENHANCED_CALCULATIONS_SUMMARY.md",
            "PLATINUM_PALLADIUM_AI_USAGE.md",
            "ANALYSIS_GAPS_AND_AI_INTEGRATION.md",
            "QUICK_REFERENCE.md",
            "QUICK_START_QA.md",
            "IMPLEMENTATION_SUMMARY.md",
            "CHANGELOG.md",
            "README.md",
        ]

        count = 0
        for doc_name in doc_files:
            filepath = self.repo_path / doc_name
            if not filepath.exists():
                continue

            try:
                content = filepath.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not read %s: %s", filepath, exc)
                continue
            enriched = (
                f"# Pt/Pd Calibration Studio Documentation\n"
                f"## Source: {doc_name}\n"
                f"## Category: Software Documentation\n"
                f"## Application: Platinum/Palladium photographic printing calibration\n\n"
                f"{content}\n\n"
                f"## Context\n"
                f"This document is part of the Pt/Pd Calibration Studio, an AI-powered\n"
                f"calibration system for platinum/palladium alternative photographic printing.\n"
                f"The system handles step tablet reading, curve generation, chemistry calculations,\n"
                f"exposure calculations, and zone system analysis.\n"
            )

            out_path = self.output_dir / f"repo_doc__{doc_name.replace('.md', '.txt')}"
            out_path.write_text(enriched, encoding="utf-8")
            count += 1

        logger.info("Prepared %d repository docs", count)
        return count

    def prepare_code_docs(self) -> int:
        """Extract and document calibration logic from Python source.

        Returns:
            Number of documents prepared.
        """
        src_path = self.repo_path / "src" / "ptpd_calibration"
        if not src_path.exists():
            src_path = self.repo_path

        key_modules = {
            "detection/detector.py": "Step tablet detection using computer vision",
            "detection/extractor.py": "Optical density measurement and extraction",
            "detection/scanner.py": "Scanner calibration and image scanning",
            "curves/generator.py": "Linearization curve generation algorithms",
            "curves/export.py": "Curve export to QTR and Piezography formats",
            "curves/analysis.py": "Curve analysis and quality metrics",
            "chemistry/calculator.py": "Platinum/palladium chemistry calculations",
            "exposure/calculator.py": "UV exposure time calculations",
            "ml/predictor.py": "Machine learning curve prediction and refinement",
            "llm/assistant.py": "LLM-based AI assistant integration",
            "agents/agent.py": "ReAct-style calibration agent",
            "agents/orchestrator.py": "Multi-agent workflow coordination",
            "agents/tools.py": "Agent tool definitions for calibration tasks",
        }

        count = 0
        for module_path, description in key_modules.items():
            filepath = src_path / module_path
            if not filepath.exists():
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Could not read %s: %s", filepath, exc)
                continue
            enriched = (
                f"# Pt/Pd Calibration Studio - Source Code\n"
                f"## Module: {module_path}\n"
                f"## Purpose: {description}\n"
                f"## Category: Calibration Software Implementation\n\n"
                f"### Code:\n\n"
                f"```python\n{content}\n```\n\n"
                f"### Module Context:\n\n"
                f"{description}. This module is part of the Pt/Pd Calibration Studio's\n"
                f"core calibration pipeline for platinum/palladium alternative photographic printing.\n"
            )

            safe_name = module_path.replace("/", "__").replace(".py", ".txt")
            out_path = self.output_dir / f"code__{safe_name}"
            out_path.write_text(enriched, encoding="utf-8")
            count += 1

        logger.info("Prepared %d code docs", count)
        return count

    def prepare_domain_knowledge(self) -> int:
        """Write domain knowledge documents from the knowledge/ directory or built-in content.

        Returns:
            Number of documents prepared.
        """
        knowledge_dir = Path(__file__).parent / "knowledge"
        count = 0

        if knowledge_dir.exists():
            for filepath in knowledge_dir.iterdir():
                if filepath.is_file() and filepath.suffix == ".txt":
                    try:
                        content = filepath.read_text(encoding="utf-8")
                    except OSError as exc:
                        logger.warning("Could not read %s: %s", filepath, exc)
                        continue
                    out_path = self.output_dir / filepath.name
                    out_path.write_text(content, encoding="utf-8")
                    count += 1

        logger.info("Prepared %d domain knowledge docs", count)
        return count

    def upload_to_gcs(
        self,
        bucket_name: str | None = None,
        prefix: str = "documents",
    ) -> int:
        """Upload all prepared documents to Google Cloud Storage.

        Args:
            bucket_name: GCS bucket name. Falls back to config.
            prefix: Path prefix within the bucket.

        Returns:
            Number of files uploaded.

        Raises:
            ImportError: If google-cloud-storage is not installed.
        """
        try:
            from google.cloud import storage
        except ImportError as err:
            raise ImportError(
                "google-cloud-storage required. Install with: pip install ptpd-calibration[vertex]"
            ) from err

        settings = get_settings().vertex
        bucket_name = bucket_name or settings.corpus_bucket
        if not bucket_name:
            raise ValueError(
                "GCS bucket name required. Set PTPD_VERTEX_CORPUS_BUCKET or pass bucket_name."
            )

        # Strip gs:// prefix if present
        if bucket_name.startswith("gs://"):
            bucket_name = bucket_name[5:]

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        count = 0
        for filepath in self.output_dir.iterdir():
            if filepath.is_file():
                blob_name = f"{prefix}/{filepath.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(filepath))
                count += 1
                logger.debug("Uploaded %s to gs://%s/%s", filepath.name, bucket_name, blob_name)

        logger.info("Uploaded %d documents to gs://%s/%s/", count, bucket_name, prefix)
        return count


def prepare_and_upload_corpus(
    repo_path: str | Path = ".",
    output_dir: str | Path | None = None,
    bucket_name: str | None = None,
) -> dict[str, int]:
    """Convenience function to prepare and upload the full corpus.

    Args:
        repo_path: Path to the repository root.
        output_dir: Local staging directory.
        bucket_name: GCS bucket for upload.

    Returns:
        Dict with counts: {"prepared": N, "uploaded": M}.
    """
    logger.info("Starting corpus preparation and upload (repo=%s)", repo_path)
    preparator = CorpusPreparator(repo_path=repo_path, output_dir=output_dir)
    prepared = preparator.prepare_all()
    uploaded = preparator.upload_to_gcs(bucket_name=bucket_name)
    logger.info("Corpus pipeline complete: %d prepared, %d uploaded", prepared, uploaded)
    return {"prepared": prepared, "uploaded": uploaded}
