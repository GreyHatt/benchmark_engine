"""
Query execution metrics collection and analysis.

This module provides functionality to collect, analyze, and compare performance
metrics from query executions across different engines.
"""
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import statistics
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Container for system resource usage metrics."""
    timestamp: float
    cpu_percent: float
    memory_used: float  # in MB
    memory_percent: float
    disk_read: float    # in MB
    disk_write: float   # in MB
    net_sent: float     # in MB
    net_recv: float     # in MB

@dataclass
class QueryMetrics:
    """Container for query execution metrics."""
    query_id: str
    engine: str
    execution_time: float
    rows_returned: int
    plan_execution_time: Optional[float] = None
    plan_analysis: Optional[Dict[str, Any]] = None
    system_metrics: List[SystemMetrics] = field(default_factory=list)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collects and analyzes query execution metrics."""
    
    def __init__(self, collection_interval: float = 0.1):
        """
        Initialize the metrics collector.
        
        Args:
            collection_interval: Time in seconds between metric samples
        """
        self.collection_interval = collection_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._metrics = []
        self._disk_io_start = None
        self._net_io_start = None
        
    def start(self) -> None:
        """Start collecting metrics in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._metrics = []
        self._disk_io_start = psutil.disk_io_counters()
        self._net_io_start = psutil.net_io_counters()
        
        self._thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self._thread.start()
        
    def stop(self) -> List[SystemMetrics]:
        """
        Stop collecting metrics and return the collected data.
        
        Returns:
            List of collected system metrics
        """
        if self._thread is None or not self._thread.is_alive():
            return self._metrics
            
        self._stop_event.set()
        self._thread.join(timeout=self.collection_interval * 2)
        return self._metrics
    
    def _collect_metrics(self) -> None:
        """Background thread function to collect metrics."""
        disk_io_prev = self._disk_io_start
        net_io_prev = self._net_io_start
        
        while not self._stop_event.is_set():
            try:
                # Get current timestamp
                timestamp = time.time()
                
                # Get CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                # Get disk I/O
                disk_io = psutil.disk_io_counters()
                disk_read = (disk_io.read_bytes - disk_io_prev.read_bytes) / (1024 * 1024)  # MB
                disk_write = (disk_io.write_bytes - disk_io_prev.write_bytes) / (1024 * 1024)  # MB
                disk_io_prev = disk_io
                
                # Get network I/O
                net_io = psutil.net_io_counters()
                net_sent = (net_io.bytes_sent - net_io_prev.bytes_sent) / (1024 * 1024)  # MB
                net_recv = (net_io.bytes_recv - net_io_prev.bytes_recv) / (1024 * 1024)  # MB
                net_io_prev = net_io
                
                # Create metrics object
                metrics = SystemMetrics(
                    timestamp=timestamp,
                    cpu_percent=cpu_percent,
                    memory_used=memory.used / (1024 * 1024),  # Convert to MB
                    memory_percent=memory.percent,
                    disk_read=disk_read,
                    disk_write=disk_write,
                    net_sent=net_sent,
                    net_recv=net_recv
                )
                
                self._metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}", exc_info=True)
                
            # Wait for next collection interval
            self._stop_event.wait(timeout=self.collection_interval)
    
    @staticmethod
    def analyze_metrics(metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """
        Analyze collected metrics and return summary statistics.
        
        Args:
            metrics: List of collected system metrics
            
        Returns:
            Dictionary containing summary statistics
        """
        if not metrics:
            return {}
            
        # Calculate statistics for each metric
        cpu_percent = [m.cpu_percent for m in metrics]
        memory_used = [m.memory_used for m in metrics]
        memory_percent = [m.memory_percent for m in metrics]
        disk_read = [m.disk_read for m in metrics]
        disk_write = [m.disk_write for m in metrics]
        net_sent = [m.net_sent for m in metrics]
        net_recv = [m.net_recv for m in metrics]
        
        def safe_stats(values: List[float]) -> Dict[str, float]:
            """Calculate statistics safely handling empty lists."""
            if not values:
                return {}
            return {
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'total': sum(values)
            }
        
        return {
            'duration_seconds': metrics[-1].timestamp - metrics[0].timestamp,
            'cpu': safe_stats(cpu_percent),
            'memory_used_mb': safe_stats(memory_used),
            'memory_percent': safe_stats(memory_percent),
            'disk_read_mb': safe_stats(disk_read),
            'disk_write_mb': safe_stats(disk_write),
            'network_sent_mb': safe_stats(net_sent),
            'network_recv_mb': safe_stats(net_recv)
        }

class QueryResultValidator:
    """Validates query results between different engines."""
    
    @staticmethod
    def validate_results(
        result1: Any,
        result2: Any,
        tolerance: float = 1e-9,
        ignore_order: bool = True,
        ignore_case: bool = True
    ) -> Dict[str, Any]:
        """
        Validate that two query results are equivalent.
        
        Args:
            result1: First result to compare (can be list, dict, or DataFrame)
            result2: Second result to compare (same type as result1)
            tolerance: Allowed difference for numeric comparisons
            ignore_order: Whether to ignore row order in comparisons
            ignore_case: Whether to ignore case in string comparisons
            
        Returns:
            Dictionary with validation results including:
            - equal: Whether the results are equivalent
            - errors: List of error messages if not equal
            - stats: Statistics about the comparison
        """
        errors = []
        stats = {}
        
        # Handle different result types
        if isinstance(result1, list) and isinstance(result2, list):
            return QueryResultValidator._validate_lists(
                result1, result2, tolerance, ignore_order, ignore_case
            )
        elif hasattr(result1, 'collect') and hasattr(result2, 'collect'):
            # Handle Spark DataFrames
            return QueryResultValidator._validate_dataframes(
                result1, result2, tolerance, ignore_order, ignore_case
            )
        elif isinstance(result1, dict) and isinstance(result2, dict):
            return QueryResultValidator._validate_dicts(
                result1, result2, tolerance, ignore_case
            )
        else:
            # Simple value comparison
            try:
                equal = QueryResultValidator._compare_values(result1, result2, tolerance, ignore_case)
                if not equal:
                    errors.append(f"Values differ: {result1} != {result2}")
                return {
                    'equal': len(errors) == 0,
                    'errors': errors,
                    'stats': stats
                }
            except Exception as e:
                return {
                    'equal': False,
                    'errors': [f"Error comparing values: {str(e)}"],
                    'stats': stats
                }
    
    @staticmethod
    def _validate_lists(
        list1: List[Any],
        list2: List[Any],
        tolerance: float,
        ignore_order: bool,
        ignore_case: bool
    ) -> Dict[str, Any]:
        """Validate two lists of results."""
        errors = []
        stats = {
            'row_count1': len(list1),
            'row_count2': len(list2),
            'matching_rows': 0,
            'mismatched_rows': 0,
            'extra_rows1': 0,
            'extra_rows2': 0
        }
        
        if len(list1) != len(list2) and not ignore_order:
            errors.append(f"Row count mismatch: {len(list1)} != {len(list2)}")
        
        if ignore_order:
            # Convert rows to tuples of comparable values
            def make_comparable(row):
                if isinstance(row, (list, tuple)):
                    return tuple(
                        v.lower() if ignore_case and isinstance(v, str) else v 
                        for v in row
                    )
                elif isinstance(row, dict):
                    return tuple(
                        (k, v.lower() if ignore_case and isinstance(v, str) else v)
                        for k, v in sorted(row.items())
                    )
                return row
                
            set1 = {make_comparable(row) for row in list1}
            set2 = {make_comparable(row) for row in list2}
            
            # Find differences
            extra_in_1 = set1 - set2
            extra_in_2 = set2 - set1
            
            if extra_in_1:
                errors.append(f"Found {len(extra_in_1)} rows in first result not in second")
                stats['extra_rows1'] = len(extra_in_1)
            if extra_in_2:
                errors.append(f"Found {len(extra_in_2)} rows in second result not in first")
                stats['extra_rows2'] = len(extra_in_2)
                
            stats['matching_rows'] = len(set1.intersection(set2))
            stats['mismatched_rows'] = len(extra_in_1) + len(extra_in_2)
            
        else:
            # Compare row by row
            for i, (row1, row2) in enumerate(zip(list1, list2)):
                try:
                    if isinstance(row1, dict) and isinstance(row2, dict):
                        result = QueryResultValidator._validate_dicts(
                            row1, row2, tolerance, ignore_case
                        )
                    elif isinstance(row1, (list, tuple)) and isinstance(row2, (list, tuple)):
                        result = QueryResultValidator._validate_lists(
                            list(row1), list(row2), tolerance, False, ignore_case
                        )
                    else:
                        equal = QueryResultValidator._compare_values(row1, row2, tolerance, ignore_case)
                        result = {'equal': equal, 'errors': []}
                        
                    if not result['equal']:
                        errors.append(f"Row {i} mismatch: {result.get('errors', ['Unknown error'])}")
                        stats['mismatched_rows'] += 1
                    else:
                        stats['matching_rows'] += 1
                        
                except Exception as e:
                    errors.append(f"Error comparing row {i}: {str(e)}")
                    stats['mismatched_rows'] += 1
        
        return {
            'equal': len(errors) == 0,
            'errors': errors,
            'stats': stats
        }
    
    @staticmethod
    def _validate_dicts(
        dict1: Dict[Any, Any],
        dict2: Dict[Any, Any],
        tolerance: float,
        ignore_case: bool
    ) -> Dict[str, Any]:
        """Validate two dictionaries of results."""
        errors = []
        stats = {
            'key_count1': len(dict1),
            'key_count2': len(dict2),
            'matching_keys': 0,
            'mismatched_values': 0,
            'extra_keys1': set(),
            'extra_keys2': set()
        }
        
        # Check for keys only in one dict
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        extra_in_1 = keys1 - keys2
        extra_in_2 = keys2 - keys1
        
        if extra_in_1:
            errors.append(f"Keys in first result not in second: {extra_in_1}")
            stats['extra_keys1'] = extra_in_1
        if extra_in_2:
            errors.append(f"Keys in second result not in first: {extra_in_2}")
            stats['extra_keys2'] = extra_in_2
        
        # Compare common keys
        common_keys = keys1.intersection(keys2)
        stats['matching_keys'] = len(common_keys)
        
        for key in common_keys:
            try:
                val1 = dict1[key]
                val2 = dict2[key]
                
                if not QueryResultValidator._compare_values(val1, val2, tolerance, ignore_case):
                    errors.append(f"Value mismatch for key '{key}': {val1} != {val2}")
                    stats['mismatched_values'] += 1
                    
            except Exception as e:
                errors.append(f"Error comparing values for key '{key}': {str(e)}")
                stats['mismatched_values'] += 1
        
        return {
            'equal': len(errors) == 0,
            'errors': errors,
            'stats': stats
        }
    
    @staticmethod
    def _validate_dataframes(df1, df2, tolerance, ignore_order, ignore_case):
        """Validate two DataFrame-like objects (Spark, Pandas, etc.)."""
        # Convert to list of dicts for comparison
        list1 = [row.asDict() for row in df1.collect()] if hasattr(df1, 'collect') else df1.to_dict('records')
        list2 = [row.asDict() for row in df2.collect()] if hasattr(df2, 'collect') else df2.to_dict('records')
        
        return QueryResultValidator._validate_lists(
            list1, list2, tolerance, ignore_order, ignore_case
        )
    
    @staticmethod
    def _compare_values(val1, val2, tolerance=1e-9, ignore_case=False):
        """Compare two values with optional tolerance for numeric types."""
        # Handle None values
        if val1 is None or val2 is None:
            return val1 is None and val2 is None
        
        # Handle string case insensitivity
        if isinstance(val1, str) and isinstance(val2, str):
            if ignore_case:
                return val1.lower() == val2.lower()
            return val1 == val2
        
        # Handle numeric types with tolerance
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(float(val1) - float(val2)) <= tolerance
        
        # Default comparison
        return val1 == val2
