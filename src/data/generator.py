import os
import logging
from typing import Dict, Any, Optional
import tempfile
import subprocess
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class TpchDataGenerator:
    """
    TPC-H data generator that uses the official TPC-H dbgen tool.
    """
    
    def __init__(self, scale_factor: float = 1.0, data_dir: str = "./data"):
        """
        Initialize the TPC-H data generator.
        
        Args:
            scale_factor: Scale factor for data generation (default: 1.0)
            data_dir: Directory to store generated data
        """
        self.scale_factor = scale_factor
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, force: bool = False) -> Dict[str, str]:
        """
        Generate TPC-H data using dbgen.
        
        Args:
            force: If True, regenerate data even if it already exists
            
        Returns:
            Dictionary mapping table names to their file paths
        """
        table_files = self._get_expected_table_files()
        if not force and all(f.exists() for f in table_files.values()):
            logger.info("TPC-H data already exists. Use force=True to regenerate.")
            return {table: str(path) for table, path in table_files.items()}
            
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                self._build_dbgen(tmp_dir)
                self._run_dbgen(tmp_dir)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error during data generation: {e}")
                logger.error(f"Command: {e.cmd}")
                logger.error(f"Return Code: {e.returncode}")
                logger.error(f"Stdout: {e.stdout.decode() if e.stdout else 'None'}")
                logger.error(f"Stderr: {e.stderr.decode() if e.stderr else 'None'}")
                raise
            
        return self._move_generated_files()
    
    def _get_expected_table_files(self) -> Dict[str, Path]:
        """Get the expected paths for all TPC-H table files."""
        tables = [
            'part', 'supplier', 'partsupp', 'customer', 
            'orders', 'lineitem', 'nation', 'region'
        ]
        return {table: self.data_dir / f"{table}.tbl" for table in tables}
    
    def _build_dbgen(self, tmp_dir: str) -> None:
        """Build the TPC-H dbgen tool."""
        logger.info("Building TPC-H dbgen tool...")
        
        subprocess.run(
            ["git", "clone", "https://github.com/electrum/tpch-dbgen.git", tmp_dir],
            check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        make_cmd = ["make", "-C", tmp_dir, "-j", "4"]
        try:
            result = subprocess.run(make_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"Make command output: {result.stdout.decode()}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during 'make' command: {e}")
            logger.error(f"Stdout: {e.stdout.decode() if e.stdout else 'None'}")
            logger.error(f"Stderr: {e.stderr.decode() if e.stderr else 'None'}")
            raise
    
    def _run_dbgen(self, dbgen_dir: str) -> None:
        """Run the dbgen tool to generate data."""
        logger.info(f"Generating TPC-H data with scale factor {self.scale_factor}...")
        
        env = os.environ.copy()
        env["DSS_PATH"] = dbgen_dir  
        env["DSS_CONFIG"] = dbgen_dir  
        
        dbgen_cmd = [
            os.path.join(dbgen_dir, "dbgen"),
            "-vf", 
            "-s", str(self.scale_factor)
        ]
        
        try:
            subprocess.run(dbgen_cmd, cwd=dbgen_dir, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running dbgen: {e}")
            logger.error(f"Stdout: {e.stdout.decode() if e.stdout else 'None'}")
            logger.error(f"Stderr: {e.stderr.decode() if e.stderr else 'None'}")
            raise
    
    def _move_generated_files(self) -> Dict[str, str]:
        """Move generated .tbl files to the data directory."""
        logger.info("Moving generated files to data directory...")
        
        table_files = {}
        for tbl_file in Path(".").glob("*.tbl"):
            dest = self.data_dir / tbl_file.name
            shutil.move(str(tbl_file), str(dest))
            table_files[tbl_file.stem] = str(dest)
            
        return table_files


def generate_tpch_data(scale_factor: float = 1.0, data_dir: str = "./data", force: bool = False) -> Dict[str, str]:
    """
    Generate TPC-H benchmark data.
    
    Args:
        scale_factor: Scale factor for data generation (default: 1.0)
        data_dir: Directory to store generated data
        force: If True, regenerate data even if it already exists
        
    Returns:
        Dictionary mapping table names to their file paths
    """
    generator = TpchDataGenerator(scale_factor=scale_factor, data_dir=data_dir)
    return generator.generate(force=force)
