
from typing import Dict, List, Optional, Any
import os
import re
from pathlib import Path

# Query templates with placeholders for parameters
QUERIES: Dict[str, str] = {
    # Pricing Summary Report
    'q1': """
    -- Pricing Summary Report Query (Q1)
    -- This query reports the amount of business that was billed, shipped, and returned.
    SELECT 
        l_returnflag,
        l_linestatus,
        SUM(l_quantity) AS sum_qty,
        SUM(l_extendedprice) AS sum_base_price,
        SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
        SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
        AVG(l_quantity) AS avg_qty,
        AVG(l_extendedprice) AS avg_price,
        AVG(l_discount) AS avg_disc,
        COUNT(*) AS count_order
    FROM 
        lineitem
    WHERE 
        l_shipdate <= date '1998-12-01' - interval ':1' day
    GROUP BY 
        l_returnflag, l_linestatus
    ORDER BY 
        l_returnflag, l_linestatus;
    """,
    
    # Minimum Cost Supplier Query (Q2)
    'q2': """
    -- Minimum Cost Supplier Query (Q2)
    -- This query finds which supplier should be selected to minimize the cost of a product.
    SELECT
        s_acctbal,
        s_name,
        n_name,
        p_partkey,
        p_mfgr,
        s_address,
        s_phone,
        s_comment
    FROM
        part,
        supplier,
        partsupp,
        nation,
        region
    WHERE
        p_partkey = ps_partkey
        AND s_suppkey = ps_suppkey
        AND p_size = :1
        AND p_type LIKE '%:2'
        AND s_nationkey = n_nationkey
        AND n_regionkey = r_regionkey
        AND r_name = ':3'
        AND ps_supplycost = (
            SELECT
                MIN(ps_supplycost)
            FROM
                partsupp,
                supplier,
                nation,
                region
            WHERE
                p_partkey = ps_partkey
                AND s_suppkey = ps_suppkey
                AND s_nationkey = n_nationkey
                AND n_regionkey = r_regionkey
                AND r_name = ':3'
        )
    ORDER BY
        s_acctbal DESC,
        n_name,
        s_name,
        p_partkey
    LIMIT 100;
    """,
    
    # Shipping Priority Query (Q3)
    'q3': """
    -- Shipping Priority Query (Q3)
    -- This query retrieves the shipping priority and potential revenue of the orders
    -- having the largest revenue.
    SELECT
        l_orderkey,
        SUM(l_extendedprice * (1 - l_discount)) AS revenue,
        o_orderdate,
        o_shippriority
    FROM
        customer,
        orders,
        lineitem
    WHERE
        c_mktsegment = ':1'
        AND c_custkey = o_custkey
        AND l_orderkey = o_orderkey
        AND o_orderdate < date ':2'
        AND l_shipdate > date ':2'
    GROUP BY
        l_orderkey,
        o_orderdate,
        o_shippriority
    ORDER BY
        revenue DESC,
        o_orderdate
    LIMIT 10;
    """,
    
    # Order Priority Checking Query (Q4)
    'q4': """
    -- Order Priority Checking Query (Q4)
    -- This query counts the number of orders ordered in a given quarter of a given year
    -- in which at least one lineitem was received by the customer later than its
    -- committed date.
    SELECT
        o_orderpriority,
        COUNT(*) AS order_count
    FROM
        orders
    WHERE
        o_orderdate >= date ':1'
        AND o_orderdate < date ':1' + interval '3' month
        AND EXISTS (
            SELECT *
            FROM lineitem
            WHERE l_orderkey = o_orderkey
            AND l_commitdate < l_receiptdate
        )
    GROUP BY
        o_orderpriority
    ORDER BY
        o_orderpriority;
    """,
    
    # Local Supplier Volume Query (Q5)
    'q5': """
    -- Local Supplier Volume Query (Q5)
    -- This query lists the revenue volume done through local suppliers.
    SELECT
        n_name,
        SUM(l_extendedprice * (1 - l_discount)) AS revenue
    FROM
        customer,
        orders,
        lineitem,
        supplier,
        nation,
        region
    WHERE
        c_custkey = o_custkey
        AND l_orderkey = o_orderkey
        AND l_suppkey = s_suppkey
        AND c_nationkey = s_nationkey
        AND s_nationkey = n_nationkey
        AND n_regionkey = r_regionkey
        AND r_name = ':1'
        AND o_orderdate >= date ':2'
        AND o_orderdate < date ':2' + interval '1' year
    GROUP BY
        n_name
    ORDER BY
        revenue DESC;
    """,
    
    # Forecasting Revenue Change Query (Q6)
    'q6': """
    -- Forecasting Revenue Change Query (Q6)
    -- This query quantifies the amount of revenue increase that would have resulted
    -- from eliminating certain company-wide discounts in a given percentage range
    -- in a given year.
    SELECT
        SUM(l_extendedprice * l_discount) AS revenue
    FROM
        lineitem
    WHERE
        l_shipdate >= date ':1'
        AND l_shipdate < date ':1' + interval '1' year
        AND l_discount BETWEEN :2 - 0.01 AND :2 + 0.01
        AND l_quantity < :3;
    """,
    
    # Volume Shipping Query (Q7)
    'q7': """
    -- Volume Shipping Query (Q7)
    -- This query determines the value of goods shipped between certain nations to
    -- help in the re-negotiation of shipping contracts.
    SELECT
        supp_nation,
        cust_nation,
        l_year,
        SUM(volume) AS revenue
    FROM (
        SELECT
            n1.n_name AS supp_nation,
            n2.n_name AS cust_nation,
            EXTRACT(YEAR FROM l_shipdate) AS l_year,
            l_extendedprice * (1 - l_discount) AS volume
        FROM
            supplier,
            lineitem,
            orders,
            customer,
            nation n1,
            nation n2
        WHERE
            s_suppkey = l_suppkey
            AND o_orderkey = l_orderkey
            AND c_custkey = o_custkey
            AND s_nationkey = n1.n_nationkey
            AND c_nationkey = n2.n_nationkey
            AND (
                (n1.n_name = ':1' AND n2.n_name = ':2')
                OR (n1.n_name = ':2' AND n2.n_name = ':1')
            )
            AND l_shipdate BETWEEN date '1995-01-01' AND date '1996-12-31'
    ) AS shipping
    GROUP BY
        supp_nation,
        cust_nation,
        l_year
    ORDER BY
        supp_nation,
        cust_nation,
        l_year;
    """,
    
    # National Market Share Query (Q8)
    'q8': """
    -- National Market Share Query (Q8)
    -- This query determines how the market share of a given nation within a given
    -- region has changed over two years for a given part type.
    SELECT
        o_year,
        SUM(CASE
            WHEN nation = ':1' THEN volume
            ELSE 0
        END) / SUM(volume) AS mkt_share
    FROM (
        SELECT
            EXTRACT(YEAR FROM o_orderdate) AS o_year,
            l_extendedprice * (1 - l_discount) AS volume,
            n2.n_name AS nation
        FROM
            part,
            supplier,
            lineitem,
            orders,
            customer,
            nation n1,
            nation n2,
            region
        WHERE
            p_partkey = l_partkey
            AND s_suppkey = l_suppkey
            AND l_orderkey = o_orderkey
            AND o_custkey = c_custkey
            AND c_nationkey = n1.n_nationkey
            AND n1.n_regionkey = r_regionkey
            AND r_name = ':2'
            AND s_nationkey = n2.n_nationkey
            AND o_orderdate BETWEEN date '1995-01-01' AND date '1996-12-31'
            AND p_type = ':3'
    ) AS all_nations
    GROUP BY
        o_year
    ORDER BY
        o_year;
    """,
    
    # Product Type Profit Measure Query (Q9)
    'q9': """
    -- Product Type Profit Measure Query (Q9)
    -- This query determines how much profit is made on a given line of parts,
    -- broken out by supplier nation and year.
    SELECT
        nation,
        o_year,
        SUM(amount) AS sum_profit
    FROM (
        SELECT
            n_name AS nation,
            EXTRACT(YEAR FROM o_orderdate) AS o_year,
            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
        FROM
            part,
            supplier,
            lineitem,
            partsupp,
            orders,
            nation
        WHERE
            s_suppkey = l_suppkey
            AND ps_suppkey = l_suppkey
            AND ps_partkey = l_partkey
            AND p_partkey = l_partkey
            AND o_orderkey = l_orderkey
            AND s_nationkey = n_nationkey
            AND p_name LIKE '%:1%'
    ) AS profit
    GROUP BY
        nation,
        o_year
    ORDER BY
        nation,
        o_year DESC;
    """,
    
    # Returned Item Reporting Query (Q10)
    'q10': """
    -- Returned Item Reporting Query (Q10)
    -- This query identifies customers who might be having problems with the parts
    -- that are shipped to them.
    SELECT
        c_custkey,
        c_name,
        SUM(l_extendedprice * (1 - l_discount)) AS revenue,
        c_acctbal,
        n_name,
        c_address,
        c_phone,
        c_comment
    FROM
        customer,
        orders,
        lineitem,
        nation
    WHERE
        c_custkey = o_custkey
        AND l_orderkey = o_orderkey
        AND o_orderdate >= date ':1'
        AND o_orderdate < date ':1' + interval '3' month
        AND l_returnflag = 'R'
        AND c_nationkey = n_nationkey
    GROUP BY
        c_custkey,
        c_name,
        c_acctbal,
        c_phone,
        n_name,
        c_address,
        c_comment
    ORDER BY
        revenue DESC
    LIMIT 20;
    """
}

# Default parameter values for each query
DEFAULT_PARAMS = {
    'q1': {'1': '90'},  # days
    'q2': {'1': '15', '2': 'BRASS', '3': 'EUROPE'},
    'q3': {'1': 'BUILDING', '2': '1995-03-15'},
    'q4': {'1': '1993-07-01'},
    'q5': {'1': 'ASIA', '2': '1994-01-01'},
    'q6': {'1': '1994-01-01', '2': '0.06', '3': '24'},
    'q7': {'1': 'FRANCE', '2': 'GERMANY'},
    'q8': {'1': 'BRAZIL', '2': 'AMERICA', '3': 'ECONOMY ANODIZED STEEL'},
    'q9': {'1': 'green'},
    'q10': {'1': '1993-10-01'}
}

def get_query(query_id: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Get a TPC-H query by ID with optional parameter substitution.
    
    Args:
        query_id: The ID of the query (e.g., 'q1', 'q2', etc.)
        params: Dictionary of parameter values to substitute
        
    Returns:
        The SQL query with parameters substituted
        
    Raises:
        ValueError: If the query_id is not found
    """
    if query_id not in QUERIES:
        raise ValueError(f"Query {query_id} not found")
    
    query = QUERIES[query_id]
    
    # Use default parameters if none provided
    if params is None:
        params = DEFAULT_PARAMS.get(query_id, {})
    
    # Perform parameter substitution
    for param, value in params.items():
        # Handle different parameter formats: :1, :param, {param}
        query = query.replace(f":{param}", str(value))
        query = query.replace(f"{{{param}}}", str(value))
    
    return query.strip()

def get_all_queries() -> Dict[str, str]:
    """
    Get all available TPC-H queries.
    
    Returns:
        Dictionary mapping query IDs to their SQL templates
    """
    return {k: v.strip() for k, v in QUERIES.items()}

def get_query_description(query_id: str) -> str:
    """
    Get the description of a TPC-H query.
    
    Args:
        query_id: The ID of the query (e.g., 'q1', 'q2', etc.)
        
    Returns:
        The query description from the comment
    """
    if query_id not in QUERIES:
        raise ValueError(f"Query {query_id} not found")
    
    # Extract the first comment line as the description
    query = QUERIES[query_id]
    lines = [line.strip() for line in query.split('\n')]
    for line in lines:
        if line.startswith('--'):
            return line[2:].strip()
    
    return "No description available"

def get_query_parameters(query_id: str) -> Dict[str, Any]:
    """
    Get the default parameters for a TPC-H query.
    
    Args:
        query_id: The ID of the query (e.g., 'q1', 'q2', etc.)
        
    Returns:
        Dictionary of parameter names and their default values
    """
    return DEFAULT_PARAMS.get(query_id, {})
