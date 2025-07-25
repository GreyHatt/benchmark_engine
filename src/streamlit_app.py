import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import json
from query.tpch_queries import get_query

# Set page config
st.set_page_config(
    page_title="Big Data Benchmarking Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000/benchmark"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .success {
        color: #28a745;
    }
    .error {
        color: #dc3545;
    }
    .warning {
        color: #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

def run_benchmark(query_id: str, engine: str, validate_results: bool = True) -> Optional[str]:
    """Run a benchmark and return the benchmark ID."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/run",
            json={
                "query_id": query_id,
                "engine": engine,
                "validate_results": validate_results,
                "execution_mode": "auto"
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error running benchmark: {str(e)}")
        return None

def get_benchmark_status(benchmark_id: str) -> Optional[Dict]:
    """Get the status of a benchmark."""
    try:
        response = requests.get(f"{API_BASE_URL}/benchmark/status/{benchmark_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting benchmark status: {str(e)}")
        return None

# Add this to your existing st.markdown("""<style>...""")
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        height: 100%; /* Ensure cards in a row have same height */
        color: #323232
    }
    .metric-card h3 {
        color: #323232;
        padding-bottom: 0;
    }
    /* Hex colors for categories */
    .card-duration {
        background-color: #E6F3FF; /* Light Blue */
        border-left: 5px solid #007BFF;
    }
    .card-cpu {
        background-color: #E6FFEC; /* Light Green */
        border-left: 5px solid #28A745;
    }
    .card-memory-usage {
        background-color: #F7E6FF; /* Light Purple */
        border-left: 5px solid #6F42C1;
    }
    .card-memory-percent {
        background-color: #FFF3E6; /* Light Orange */
        border-left: 5px solid #FD7E14;
    }
    .card-disk-read {
        background-color: #FFE6E6; /* Light Red */
        border-left: 5px solid #DC3545;
    }
    .card-disk-write {
        background-color: #E6FFFA; /* Light Teal */
        border-left: 5px solid #20C997;
    }
    .card-network-sent {
        background-color: #F0F0F0; /* Light Grey */
        border-left: 5px solid #6C757D;
    }
    .card-network-received {
        background-color: #E0FFFF; /* Aqua */
        border-left: 5px solid #17A2B8;
    }
    .card-other {
        background-color: #F2F2F2; /* Slightly darker grey for 'Other' */
        border-left: 5px solid #A0A0A0;
    }
    .success {
        color: #28a745;
    }
    .error {
        color: #dc3545;
    }
    .warning {
        color: #ffc107;
    }
    /* New styles for metric grid within cards */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); /* Responsive 3-column layout */
        gap: 20px; /* Space between metric items */
        margin-top: 15px; /* Space below category title */
    }
    .metric-item-display {
        /* Styling for the individual metric box within the grid */
        /* background-color: rgba(255,255,255,0.1); /* Optional subtle background */
        padding: 5px 0; /* Adjust padding as needed */
        text-align: left;
    }
    .metric-item-label {
        font-size: 0.8em; /* Smaller label font */
        color: #666; /* Lighter color for label */
        margin-bottom: 2px;
    }
    .metric-item-value {
        font-size: 1.3em; /* Larger value font */
        font-weight: bold;
        color: #333; /* Darker color for value */
    }

    /* Override for dark theme if needed for text colors */
    html[data-theme='dark'] .metric-item-label {
        color: #BBB; /* Lighter color for label in dark theme */
    }
    html[data-theme='dark'] .metric-item-value {
        color: #F8F8F8; /* White/off-white for value in dark theme */
    }
    /* Ensure card height consistency */
    .metric-card {
        height: auto; /* Allow height to adjust to content */
        margin-bottom: 20px; /* Space between cards */
    }

    /* Color classes for cards - ensure these are defined if not already */
    .card-duration { background-color: #E6F3FF; border-left: 5px solid #007BFF; } /* Light Blue */
    .card-cpu { background-color: #E6FFEC; border-left: 5px solid #28A745; } /* Light Green */
    .card-memory-usage { background-color: #F7E6FF; border-left: 5px solid #6F42C1; } /* Light Purple */
    .card-memory-percent { background-color: #FFF3E6; border-left: 5px solid #FD7E14; } /* Light Orange */
    .card-disk-read { background-color: #FFE6E6; border-left: 5px solid #DC3545; } /* Light Red */
    .card-disk-write { background-color: #E6FFFA; border-left: 5px solid #20C997; } /* Light Teal */
    .card-network-sent { background-color: #F0F0F0; border-left: 5px solid #6C757D; } /* Light Grey */
    .card-network-received { background-color: #E0FFFF; border-left: 5px solid #17A2B8; } /* Aqua */
    .card-other { background-color: #F2F2F2; border-left: 5px solid #A0A0A0; } /* Slightly darker grey for 'Other' */

    </style>
""", unsafe_allow_html=True)

def display_metrics(metrics: Dict, title: str = "Metrics") -> None:
    """Display metrics in categorized, color-coded cards, with two cards per row.
    Each card's content, including metrics, is rendered as a single markdown block.
    Category headings are now H3.

    Args:
        metrics: Dictionary of metrics to display
        title: Title for the metrics section
    """
    if not metrics:
        return

    st.header(title)

    # Define categories, their corresponding prefixes, and a color class
    categories = {
        "Duration": {
            "prefixes": ["duration_seconds"],
            "color_class": "card-duration"
        },
        "CPU Metrics (%)": {
            "prefixes": ["cpu.min", "cpu.max", "cpu.mean", "cpu.median", "cpu.total", "cpu.stdev"],
            "color_class": "card-cpu"
        },
        "Memory Usage (MB)": {
            "prefixes": ["memory_used_mb.min", "memory_used_mb.max", "memory_used_mb.mean",
                         "memory_used_mb.median", "memory_used_mb.total", "memory_used_mb.stdev"],
            "color_class": "card-memory-usage"
        },
        "Memory Percentage (%)": {
            "prefixes": ["memory_percent.min", "memory_percent.max", "memory_percent.mean",
                         "memory_percent.median", "memory_percent.total", "memory_percent.stdev"],
            "color_class": "card-memory-percent"
        },
        "Disk Read (MB)": {
            "prefixes": ["disk_read_mb.min", "disk_read_mb.max", "disk_read_mb.mean",
                         "disk_read_mb.median", "disk_read_mb.total", "disk_read_mb.stdev"],
            "color_class": "card-disk-read"
        },
        "Disk Write (MB)": {
            "prefixes": ["disk_write_mb.min", "disk_write_mb.max", "disk_write_mb.mean",
                         "disk_write_mb.median", "disk_write_mb.total", "disk_write_mb.stdev"],
            "color_class": "card-disk-write"
        },
        "Network Sent (MB)": {
            "prefixes": ["network_sent_mb.min", "network_sent_mb.max", "network_sent_mb.mean",
                          "network_sent_mb.median", "network_sent_mb.total", "network_sent_mb.stdev"],
            "color_class": "card-network-sent"
        },
        "Network Received (MB)": {
            "prefixes": ["network_recv_mb.min", "network_recv_mb.max", "network_recv_mb.mean",
                          "network_recv_mb.median", "network_recv_mb.total", "network_recv_mb.stdev"],
            "color_class": "card-network-received"
        }
    }

    # Custom mapping for more relevant metric names
    metric_name_mapping = {
        "duration_seconds": "Execution Duration",
        "min": "Minimum",
        "max": "Maximum",
        "mean": "Average",
        "median": "Median",
        "total": "Total",
        "stdev": "Std. Deviation",
        # Specific overrides for clarity
        "cpu.total": "Total CPU Usage",
        "memory_used_mb.total": "Total Memory Used",
        "memory_percent.total": "Total Memory Percent",
        "disk_read_mb.total": "Total Disk Read",
        "disk_write_mb.total": "Total Disk Write",
        "network_sent_mb.total": "Total Network Sent",
        "network_recv_mb.total": "Total Network Received",
    }

    # Flatten metrics with original keys for easy lookup
    flat_metrics_dict = {}
    def flatten_metrics(metrics_dict: Dict, prefix: str = '') -> None:
        for key, value in metrics_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flatten_metrics(value, full_key)
            else:
                flat_metrics_dict[full_key] = value

    flatten_metrics(metrics)

    # Function to generate a single card's HTML
    def generate_card_html(category_title: str, details: Dict, metrics_dict: Dict) -> Optional[str]:
        prefixes = details["prefixes"]
        color_class = details["color_class"]
        category_metrics = []
        
        for prefix in prefixes:
            if prefix in metrics_dict:
                category_metrics.append((prefix, metrics_dict[prefix]))
        
        if not category_metrics:
            return None

        # Build the HTML content for the entire card
        card_html = f"<div class='metric-card {color_class}'>"
        card_html += f"<h3>{category_title}</h3>" # Changed to H3
        card_html += f"<div class='metrics-grid'>" # Start the grid for internal metrics
        
        for key, value in category_metrics:
            # Determine display name
            key_parts = key.split('.')
            if key in metric_name_mapping:
                display_name = metric_name_mapping[key]
            elif len(key_parts) > 1 and key_parts[-1] in metric_name_mapping:
                display_name = metric_name_mapping[key_parts[-1]]
            else:
                display_name = key.replace('_', ' ').replace('.', ' ').title()
            
            # Format value
            formatted_value = f"{value:.3f}" if isinstance(value, float) else value

            # Add individual metric item to the grid
            card_html += (
                f"<div class='metric-item-display'>"
                f"<div class='metric-item-label'>{display_name}</div>"
                f"<div class='metric-item-value'>{formatted_value}</div>"
                f"</div>"
            )
        if category_title == 'Duration' :
            for _ in range(4):
                card_html += (
                f"<div class='metric-item-display'>"
                f"<div class='metric-item-label' style='color:transparent;'> _ </div>"
                f"<div class='metric-item-value' style='color:transparent;'> _ </div>"
                f"</div>"
            )
                
        card_html += "</div>" # Close metrics-grid
        card_html += "</div>" # Close metric-card
        return card_html

    # Prepare categories for two-column layout
    category_items = list(categories.items())
    num_categories = len(category_items)

    for i in range(0, num_categories, 2):
        # Use st.columns([1, 1]) to explicitly define equal width columns
        col1, col2 = st.columns([1, 1]) 

        # Process first card in the row
        if i < num_categories:
            category_title1, details1 = category_items[i]
            card_html1 = generate_card_html(category_title1, details1, flat_metrics_dict)
            with col1:
                if card_html1:
                    st.markdown(card_html1, unsafe_allow_html=True)

        # Process second card in the row, if it exists
        if i + 1 < num_categories:
            category_title2, details2 = category_items[i+1]
            card_html2 = generate_card_html(category_title2, details2, flat_metrics_dict)
            with col2:
                if card_html2:
                    st.markdown(card_html2, unsafe_allow_html=True)
        
    # Display any remaining metrics not covered by categories (if any unexpected ones appear)
    primary_prefixes = [p.split('.')[0] for details in categories.values() for p in details["prefixes"]]
    primary_prefixes = list(set(primary_prefixes))

    other_metrics = []
    for k, v in flat_metrics_dict.items():
        is_covered = False
        for pp in primary_prefixes:
            if k.startswith(pp) or k == "duration_seconds": # 'duration_seconds' is handled specifically
                is_covered = True
                break
        if not is_covered:
            other_metrics.append((k, v))

    if other_metrics:
        # "Other Metrics" card will always take a full row if it appears
        other_card_html = f"<div class='metric-card card-other'>"
        other_card_html += "<h3>Other Metrics</h3>" # Changed to H3
        other_card_html += f"<div class='metrics-grid'>"
        
        for key, value in other_metrics:
            key_parts = key.split('.')
            if key in metric_name_mapping:
                display_name = metric_name_mapping[key]
            elif len(key_parts) > 1 and key_parts[-1] in metric_name_mapping:
                display_name = metric_name_mapping[key_parts[-1]]
            else:
                display_name = key.replace('_', ' ').replace('.', ' ').title()

            formatted_value = f"{value:.4f}" if isinstance(value, float) else value
            
            other_card_html += (
                f"<div class='metric-item-display'>"
                f"<div class='metric-item-label'>{display_name}</div>"
                f"<div class='metric-item-value'>{formatted_value}</div>"
                f"</div>"
            )
        other_card_html += "</div>" # Close metrics-grid
        other_card_html += "</div>" # Close metric-card

        st.markdown(other_card_html, unsafe_allow_html=True)
        
def display_validation_results(validation: Dict) -> None:
    """Display validation results."""
    if not validation:
        return
        
    with st.expander("Validation Results", expanded=True):
        if validation.get("passed"):
            st.success("✅ Results validated successfully")
        else:
            st.error("❌ Validation failed")
            
        if "errors" in validation and validation["errors"]:
            with st.container():
                st.write("#### Errors")
                for error in validation["errors"][:5]:  # Show first 5 errors
                    st.error(f"- {error}")
                if len(validation["errors"]) > 5:
                    st.warning(f"... and {len(validation['errors']) - 5} more errors")
        
        if "stats" in validation and validation["stats"]:
            st.write("#### Statistics")
            display_metrics(validation["stats"])

def display_system_metrics(metrics: Dict) -> None:
    """Display system metrics in a card."""
    if not metrics:
        return
        
    with st.expander("System Metrics", expanded=False):
        # CPU Usage
        if "cpu" in metrics and "mean" in metrics["cpu"]:
            st.write("#### CPU Usage (%)")
            cpu_data = {
                "min": metrics["cpu"].get("min", 0),
                "mean": metrics["cpu"].get("mean", 0),
                "max": metrics["cpu"].get("max", 0)
            }
            st.metric("Average CPU Usage", f"{cpu_data['mean']:.1f}%")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cpu_data["mean"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgreen"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "red"}],
                       'threshold': {
                           'line': {'color': "black", 'width': 4},
                           'thickness': 0.75,
                           'value': cpu_data["mean"]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Memory Usage
        if "memory_used_mb" in metrics and "max" in metrics["memory_used_mb"]:
            st.write("#### Memory Usage (MB)")
            mem_data = {
                "min": metrics["memory_used_mb"].get("min", 0),
                "mean": metrics["memory_used_mb"].get("mean", 0),
                "max": metrics["memory_used_mb"].get("max", 0)
            }
            st.metric("Peak Memory Usage", f"{mem_data['max']:.1f} MB")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mem_data["max"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Peak Memory Usage (MB)"},
                gauge={'axis': {'range': [0, mem_data["max"] if mem_data["max"] > 0 else 100]},
                       'bar': {'color': "darkblue"},
                       'threshold': {
                           'line': {'color': "black", 'width': 4},
                           'thickness': 0.75,
                           'value': mem_data["max"]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

def display_comparison_metrics(metrics: Dict) -> None:
    """Display comparison metrics between engines."""
    if not metrics:
        return
        
    with st.expander("Engine Comparison", expanded=True):
        if "faster_engine" in metrics and "speedup" in metrics:
            st.metric(
                f"Faster Engine: {metrics['faster_engine'].upper()}",
                f"{metrics['speedup']:.2f}x speedup"
            )
            
            # Create a bar chart comparing execution times
            if "spark_time" in metrics and "duckdb_time" in metrics:
                df = pd.DataFrame({
                    "Engine": ["Spark", "DuckDB"],
                    "Execution Time (s)": [metrics["spark_time"], metrics["duckdb_time"]]
                })
                
                fig = px.bar(
                    df, 
                    x="Engine", 
                    y="Execution Time (s)",
                    color="Engine",
                    title="Execution Time Comparison",
                    text_auto='.3s'
                )
                fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed comparison metrics
        if "time_difference" in metrics:
            st.metric("Time Difference", f"{metrics['time_difference']:.4f} seconds")

def load_queries() -> Optional[Dict]:
    """Load queries from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/queries")
        response.raise_for_status()
        data = response.json()
        if data and "queries" in data:
            # Convert list of queries into dict {id: query_dict}
            st.session_state.queries = {q["id"]: q for q in data["queries"]}
            st.session_state.queries_loaded = True
            return st.session_state.queries
        else:
            st.session_state.queries = {}
            return {}
    except Exception as e:
        st.error(f"Error loading queries: {str(e)}")
        return None

def load_engines() -> Optional[List[str]]:
    """Load engines from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/engines")
        response.raise_for_status()
        data = response.json()
        if data:
            st.session_state.engines = data
            st.session_state.engines_loaded = True
        return data
    except Exception as e:
        st.error(f"Error loading engines: {str(e)}")
        return None

def generate_tpch_data(scale_factor: float = 0.1, force: bool = False) -> Dict:
    """Trigger TPC-H data generation via the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/generate",
            json={
                "scale_factor": scale_factor,
                "force": force
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating TPC-H data: {str(e)}")
        return {"status": "error", "message": str(e)}

def check_data_exists() -> bool:
    """Check if TPC-H data files exist."""
    try:
        response = requests.get(f"{API_BASE_URL}/data/status")
        if response.status_code == 200:
            return response.json().get("data_exists", False)
    except:
        pass
    return False

def main():
    """Main Streamlit application."""
    st.title("📊 Big Data Benchmarking Engine")
    
    # Add data generation section to the sidebar
    with st.sidebar:
        st.header("TPC-H Data Management")
        
        # Check if data exists
        data_exists = check_data_exists()
        
        if data_exists:
            st.success("✅ TPC-H data is ready")
            if st.button("🔁 Regenerate Data"):
                with st.spinner("Regenerating TPC-H data (this may take a while)..."):
                    result = generate_tpch_data(scale_factor=0.1, force=True)
                    if result.get("status") == "success":
                        st.success("✅ " + result["message"])
                        st.experimental_rerun()
                    else:
                        st.error("❌ " + result.get("error", "Failed to regenerate data"))
        else:
            st.warning("⚠️ TPC-H data not found")
            if st.button("🔧 Generate TPC-H Data"):
                with st.spinner("Generating TPC-H data (this may take a while)..."):
                    result = generate_tpch_data(scale_factor=0.1)
                    if result.get("status") == "success":
                        st.success("✅ " + result["message"])
                        st.experimental_rerun()
                    else:
                        st.error("❌ " + result.get("error", "Failed to generate data"))
        
        st.markdown("---")
        
        # Benchmark configuration
        st.header("Benchmark Configuration")
        
        # Load available queries and engines
        if 'queries' not in st.session_state:
            st.session_state.queries = {}
        if 'engines' not in st.session_state:
            st.session_state.engines = []
            
        # Load queries and engines
        if st.button("🔄 Load Queries"):
            st.session_state.queries = load_queries() or {}
            st.session_state.engines = load_engines() or []
        
        # Query selection
        query_id = st.selectbox(
            "Select Query",
            options=[""] + list(st.session_state.queries.keys()),
            format_func=lambda x: st.session_state.queries.get(x, {}).get("name", x) if x else "Select a query"
        )
        
        # Engine selection
        engine = st.selectbox(
            "Select Engine",
            options=[""] + st.session_state.engines,
            format_func=lambda x: x.capitalize() if x else "Select an engine"
        )
        
        # Run benchmark button
        run_benchmark_btn = st.button("▶️ Run Benchmark", 
                                    disabled=not (query_id and engine),
                                    type="primary")

    # Main content area
    if run_benchmark_btn and query_id and engine:
        with st.spinner("Running benchmark..."):
            benchmark_id = run_benchmark(
                query_id=query_id,
                engine=engine.lower(),
                validate_results=True
            )
            
            if benchmark_id:
                st.session_state.current_benchmark = benchmark_id
                st.experimental_rerun()
    
    # Display benchmark results if available
    if "current_benchmark" in st.session_state:
        display_benchmark_results(st.session_state.current_benchmark)

def display_benchmark_results(benchmark_id: str) -> None:
    """Display detailed results for a benchmark."""
    result = st.session_state.current_benchmark
    if not result:
        st.error(f"Could not load results for benchmark {benchmark_id}")
        return
    
    status_emoji = "" if result.get("status") == "completed" else ""
    st.header(f"{status_emoji} Benchmark Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Query ID", result.get("query_id", "N/A"))
    with col2:
        st.metric("Engine", result.get("engine", "N/A").upper())
    
    if result.get("engine") == "hybrid":
        st.write("### Hybrid Execution Results")
        
        if "comparison_metrics" in result:
            display_comparison_metrics(result["comparison_metrics"])
        
        if "spark_result" in result:
            with st.expander("Spark Execution Details", expanded=False):
                display_engine_results(result["spark_result"])
        
        if "duckdb_result" in result:
            with st.expander("DuckDB Execution Details", expanded=False):
                display_engine_results(result["duckdb_result"])
        
        if "validation_result" in result:
            display_validation_results(result["validation_result"])
    else:
        engine_result = result.get("spark_result") or result.get("duckdb_result")
        query = result.get("query_id")
        if engine_result:
            display_engine_results(engine_result, query)
        
        if "validation_result" in result:
            display_validation_results(result["validation_result"])
    
    with st.expander("Raw Results"):
        st.json(result)

def display_engine_results(engine_result: Dict, query: Optional[str] = None) -> None:
    """Display results for a specific engine."""
    if not engine_result:
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Execution Time", f"{engine_result.get('execution_time', 0):.4f} seconds")
    with col2:
        st.metric("Rows Returned", engine_result.get("rows_returned", 0))
    with col3:
        status = "success" if engine_result.get("success", False) else "error"
        st.metric("Status", status.capitalize())
    
    if query:
        with st.expander("View Query"):
            st.code(get_query(query), language="sql")

    if "query_plan" in engine_result and engine_result["query_plan"]:
        with st.expander("Query Plan"):
            st.code(engine_result["query_plan"], language="sql")

    if "error" in engine_result and engine_result["error"]:
        st.error(f"Error: {engine_result['error']}")
    
    if "system_metrics" in engine_result:
        display_system_metrics(engine_result["system_metrics"])
    
    if "metrics" in engine_result:
        display_metrics(engine_result["metrics"])
    
    
if __name__ == "__main__":
    main()
