import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

class TpchDataGenerator:
    """
    TPC-H data generator that clones, builds, and runs the official dbgen tool
    inside the specified data directory directly (e.g., /app/data).
    """

    TABLES = [
        'part', 'supplier', 'partsupp', 'customer',
        'orders', 'lineitem', 'nation', 'region'
    ]

    def __init__(self, scale_factor: float = 1.0, data_dir: str = "/app/data"):
        self.scale_factor = scale_factor
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, force: bool = False) -> Dict[str, str]:
        """
        Generate TPC-H data in the data directory by cloning, building, and
        running dbgen. Cleans up intermediate files after generation.

        Args:
            force: If True, regenerates data even if existing files are present.

        Returns:
            Dictionary mapping table names to their generated file paths.
        """
        table_files = {tbl: self.data_dir / f"{tbl}.tbl" for tbl in self.TABLES}

        # Check if data exists already (skip if not forced)
        if not force and all(f.exists() for f in table_files.values()):
            logger.info("TPC-H data already exists. Use force=True to regenerate.")
            return {table: str(path) for table, path in table_files.items()}

        # Clean out previous contents in data_dir to avoid conflicts
        for item in self.data_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                logger.warning(f"Could not remove {item}: {e}")

        # Define repo clone path inside data_dir
        repo_dir = self.data_dir / "tpch-dbgen"

        # Clone and build dbgen
        self._clone_dbgen(repo_dir)
        self._build_dbgen(repo_dir)

        # Run the dbgen tool to generate .tbl files
        self._run_dbgen(repo_dir)

        # Move the generated .tbl files from repo dir root to data_dir root
        for tbl_file in repo_dir.glob("*.tbl"):
            dest = self.data_dir / tbl_file.name
            shutil.move(str(tbl_file), dest)

        # Clean up cloned repo
        try:
            shutil.rmtree(repo_dir)
        except Exception as e:
            logger.warning(f"Could not remove cloned dbgen directory {repo_dir}: {e}")

        # Return absolute paths of generated files
        generated_files = {table: str(self.data_dir / f"{table}.tbl") for table in self.TABLES}
        logger.info(f"Generated TPC-H data files: {generated_files}")
        return generated_files

    def _clone_dbgen(self, repo_dir: Path) -> None:
        if repo_dir.exists():
            logger.info(f"Removing existing dbgen repo at {repo_dir}")
            shutil.rmtree(repo_dir)

        logger.info(f"Cloning TPC-H dbgen repository into {repo_dir}")
        subprocess.run(
            ["git", "clone", "https://github.com/electrum/tpch-dbgen.git", str(repo_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def _build_dbgen(self, repo_dir: Path) -> None:
        logger.info(f"Building TPC-H dbgen tool in {repo_dir}")
        result = subprocess.run(
            ["make", "-C", str(repo_dir), "-j4"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Make output:\n{result.stdout.decode()}")

    def _run_dbgen(self, repo_dir: Path) -> None:
        logger.info(f"Generating TPC-H data with scale factor {self.scale_factor} inside {repo_dir}")

        env = os.environ.copy()
        env["DSS_PATH"] = str(repo_dir)
        env["DSS_CONFIG"] = str(repo_dir)

        dbgen_cmd = [
            str(repo_dir / "dbgen"),
            "-vf",
            "-s", str(self.scale_factor)
        ]

        result = subprocess.run(
            dbgen_cmd,
            cwd=str(repo_dir),
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"dbgen output:\n{result.stdout.decode()}")


def generate_tpch_data(scale_factor: float = 1.0, data_dir: str = "/app/data", force: bool = False) -> Dict[str, str]:
    """
    Convenience function for generating TPC-H data.

    Args:
        scale_factor: Scale factor for data generation (default 1.0)
        data_dir: Directory where data will be generated (default /app/data)
        force: If True, force regeneration even if data exists

    Returns:
        Dict mapping table names to their absolute file paths.
    """
    generator = TpchDataGenerator(scale_factor=scale_factor, data_dir=data_dir)
    return generator.generate(force=force)
