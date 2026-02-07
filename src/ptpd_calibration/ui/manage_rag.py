"""
Command-line interface for managing the RAG database.

Provides commands to add documents, search, and check status.

Usage:
  python scripts/manage_rag.py --help
  python scripts/manage_rag.py add --path "path/to/your/docs"
  python scripts/manage_rag.py search "your query"
  python scripts/manage_rag.py status
"""

import logging
import sys
from pathlib import Path

try:
    import typer
    from rich.console import Console
except ImportError:
    print("Please install CLI dependencies: pip install typer rich")
    sys.exit(1)

# Add project root to the Python path to allow importing from 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ptpd_calibration.rag.database import get_rag_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Manage the Platinum/Palladium Printing RAG knowledge base."
)
console = Console()


@app.command()
def add(
    path: Path = typer.Option(
        ...,
        "--path",
        "-p",
        help="Path to a directory of .txt files or a single .txt file to add to the database.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    )
):
    """
    Add documents from a file or directory to the RAG database.
    """
    rag_db = get_rag_db()
    console.print(f"Database contains [bold cyan]{rag_db.get_document_count()}[/] documents.")

    documents = []
    if path.is_dir():
        console.print(f"Scanning directory: [green]{path}[/]")
        files = list(path.glob("*.txt"))
        for file_path in files:
            documents.append(file_path.read_text(encoding="utf-8"))
        console.print(f"Found {len(documents)} .txt files to add.")
    elif path.is_file():
        console.print(f"Reading file: [green]{path}[/]")
        documents.append(path.read_text(encoding="utf-8"))

    if not documents:
        console.print("[yellow]No new documents found to add.[/yellow]")
        raise typer.Exit()

    rag_db.add_documents(documents)
    console.print(f"\n[bold green]Successfully added {len(documents)} document(s).[/bold green]")
    console.print(f"Database now contains [bold cyan]{rag_db.get_document_count()}[/] documents.")


@app.command()
def search(query: str, n_results: int = typer.Option(3, "--n", "-k", help="Number of results to return.")):
    """
    Search the RAG database with a query.
    """
    rag_db = get_rag_db()
    console.print(f"Searching for: '[italic]{query}[/italic]'...")
    results = rag_db.search(query, n_results=n_results)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, doc in enumerate(results):
        console.print(f"\n[bold]Result {i+1}:[/bold]\n---\n{doc}")


@app.command()
def status():
    """Check the status of the RAG database."""
    rag_db = get_rag_db()
    console.print(f"Database contains [bold cyan]{rag_db.get_document_count()}[/] documents.")


if __name__ == "__main__":
    app()
