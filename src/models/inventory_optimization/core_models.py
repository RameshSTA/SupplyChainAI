# File: src/models/inventory_optimization/core_models.py
"""
Core models and functions for inventory optimization calculations.

This module provides functions to calculate:
- Economic Order Quantity (EOQ)
- Z-score for a given service level
- Safety Stock (SS) considering demand and lead time variability
- Reorder Point (ROP)
It also includes a function to run a (Q,R)-like inventory simulation.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm # For Z-score in safety stock calculation

def calculate_eoq(
    annual_demand: float,
    ordering_cost_per_order: float,
    annual_holding_cost_per_unit: float
) -> float | None:
    """
    Calculates the Economic Order Quantity (EOQ).

    The EOQ formula aims to find the optimal order quantity that minimizes
    the total inventory costs, including ordering costs and holding costs.

    Args:
        annual_demand: Total annual demand for the item (units).
        ordering_cost_per_order: Cost incurred each time an order is placed ($).
        annual_holding_cost_per_unit: Cost to hold one unit of inventory for one year ($).

    Returns:
        The calculated EOQ, rounded to the nearest whole number, or None if
        inputs are invalid or an error occurs.
    """
    if not (isinstance(annual_demand, (int, float)) and
            isinstance(ordering_cost_per_order, (int, float)) and
            isinstance(annual_holding_cost_per_unit, (int, float))):
        print("Error: All inputs for EOQ must be numeric.")
        return None
    if annual_demand < 0 or ordering_cost_per_order < 0:
        print("Error: Annual demand and ordering cost must be non-negative for EOQ calculation.")
        return None
    if annual_holding_cost_per_unit <= 0: # Denominator must be positive
        print("Error: Annual holding cost per unit must be positive for EOQ calculation.")
        return None
    try:
        eoq = np.sqrt((2 * annual_demand * ordering_cost_per_order) / annual_holding_cost_per_unit)
        return round(eoq) # EOQ is typically rounded to a whole number for practical purposes
    except Exception as e:
        print(f"Error calculating EOQ: {e}")
        return None

def get_z_score(service_level_percentage: float) -> float | None:
    """
    Calculates the Z-score (number of standard deviations) for a given
    service level percentage using the inverse of the standard normal CDF.

    Args:
        service_level_percentage: The desired probability of not stocking out,
                                  expressed as a percentage (e.g., 95 for 95%).

    Returns:
        The calculated Z-score, or None if the service level is outside a
        valid practical range or an error occurs.
    """
    if not isinstance(service_level_percentage, (int, float)):
        print("Error: Service level percentage must be numeric.")
        return None
    # Practical range for service level: (0, 100).
    # 50% service level (Z=0) is technically valid.
    # Values >= 100% would lead to infinite Z-score.
    if not (0 < service_level_percentage < 100):
        if np.isclose(service_level_percentage, 50.0): # Allow 50%
             pass
        elif service_level_percentage == 100.0: # Handle 100% explicitly for very high Z
            print("Warning: 100% service level results in an infinitely large Z-score. Using a very high Z-score (e.g., for 99.999%).")
            service_level_percentage = 99.999 # Use a practical maximum
        else:
             print("Error: Service level percentage must be between 0 (exclusive) and 100 (inclusive for practical limits).")
             return None

    service_level_decimal = service_level_percentage / 100.0
    try:
        z_score = norm.ppf(service_level_decimal)
        return z_score
    except Exception as e:
        print(f"Error calculating Z-score: {e}")
        return None

def calculate_safety_stock(
    avg_demand_per_period: float,
    std_dev_demand_per_period: float,
    avg_lead_time_in_periods: float,
    std_dev_lead_time_in_periods: float,
    service_level_percentage: float
) -> float | None:
    """
    Calculates Safety Stock considering variability in both demand and lead time.

    The formula used is: Z * sqrt((AvgLT * StdDevD^2) + (AvgD^2 * StdDevLT^2))
    where Z is the Z-score for the desired service level.
    All time units (for demand and lead time) must be consistent (e.g., weekly).

    Args:
        avg_demand_per_period: Average demand in a single period (e.g., average weekly sales).
        std_dev_demand_per_period: Standard deviation of demand during that period.
        avg_lead_time_in_periods: Average lead time, in the same periods as demand.
        std_dev_lead_time_in_periods: Standard deviation of lead time, in the same periods.
        service_level_percentage: Desired service level (e.g., 95 for 95%).

    Returns:
        The calculated safety stock, rounded to the nearest whole number and
        ensured to be non-negative, or None if inputs are invalid or an error occurs.
    """
    if not all(isinstance(val, (int, float)) and val >= 0 for val in [
        avg_demand_per_period, std_dev_demand_per_period,
        avg_lead_time_in_periods, std_dev_lead_time_in_periods
    ]):
        print("Error: Demand and lead time statistics must be non-negative numbers.")
        return None

    z_score = get_z_score(service_level_percentage)
    if z_score is None: # Error message already printed by get_z_score
        return None

    try:
        # Variance of demand during lead time
        var_demand_component = avg_lead_time_in_periods * (std_dev_demand_per_period ** 2)
        # Variance of lead time component (demand during lead time uncertainty)
        var_lead_time_component = (avg_demand_per_period ** 2) * (std_dev_lead_time_in_periods ** 2)

        std_dev_demand_during_lead_time = np.sqrt(var_demand_component + var_lead_time_component)
        safety_stock_val = z_score * std_dev_demand_during_lead_time

        # Safety stock cannot be negative. Round for practical application.
        return round(max(0, safety_stock_val))
    except Exception as e:
        print(f"Error calculating Safety Stock: {e}")
        return None

def calculate_reorder_point(
    avg_demand_per_period: float,
    avg_lead_time_in_periods: float,
    safety_stock: float
) -> float | None:
    """
    Calculates the Reorder Point (ROP).

    ROP is the inventory level at which a new order should be placed.
    Formula: (Average Demand per Period * Average Lead Time in Periods) + Safety Stock.
    All time units must be consistent.

    Args:
        avg_demand_per_period: Average demand in a single period.
        avg_lead_time_in_periods: Average lead time, in the same periods.
        safety_stock: Calculated safety stock.

    Returns:
        The calculated reorder point, rounded to the nearest whole number,
        or None if inputs are invalid or an error occurs.
    """
    if not all(isinstance(val, (int, float)) and val >= 0 for val in [
        avg_demand_per_period, avg_lead_time_in_periods, safety_stock
    ]):
        print("Error: Average demand, average lead time, and safety stock must be non-negative numbers.")
        return None
    try:
        demand_during_lead_time = avg_demand_per_period * avg_lead_time_in_periods
        reorder_point = demand_during_lead_time + safety_stock
        return round(reorder_point) # ROP is typically rounded
    except Exception as e:
        print(f"Error calculating Reorder Point: {e}")
        return None

def run_inventory_simulation(
    demand_series: list | pd.Series | np.ndarray,
    initial_inventory: float,
    order_quantity_eoq: float,
    reorder_point_rop: float,
    lead_time_periods: int,
    simulation_periods: int
) -> pd.DataFrame:
    """
    Simulates inventory levels over a number of periods based on a (Q,R) policy.

    The simulation tracks inventory levels, orders placed, orders received,
    units sold, and stockouts for each period. An order of size EOQ is placed
    when the end-of-period inventory level is at or below the ROP, provided
    no other order is already expected to arrive within the lead time.

    Args:
        demand_series: A list, pandas Series, or NumPy array representing
                       the demand for each period of the simulation. Must have
                       at least `simulation_periods` elements.
        initial_inventory: The starting inventory level.
        order_quantity_eoq: The fixed quantity to order (Q, typically EOQ).
        reorder_point_rop: The inventory level at which to place a new order (R).
        lead_time_periods: The number of periods between placing an order and
                           receiving it. Assumed constant.
        simulation_periods: The total number of periods to simulate.

    Returns:
        pd.DataFrame: A DataFrame detailing the simulation results for each period,
                      with columns including 'Period', 'Demand', 'Inventory_Start_Period',
                      'Order_Arriving_This_Period', 'Inventory_After_Arrival', 'Units_Sold',
                      'Stockout_Units', 'Inventory_End_Period', 'Order_Placed_Qty',
                      and 'Order_Due_Start_of_Period'.

    Raises:
        TypeError: If `demand_series` is not a list, pandas Series, or NumPy array.
        ValueError: If `demand_series` is too short, `order_quantity_eoq` is not positive,
                    or `lead_time_periods` is negative.
    """
    if not isinstance(demand_series, (list, pd.Series, np.ndarray)):
         raise TypeError("`demand_series` must be a list, pandas Series, or NumPy array.")
    if len(demand_series) < simulation_periods:
        raise ValueError(
            f"Demand series length ({len(demand_series)}) is shorter than "
            f"simulation periods ({simulation_periods}). Provide enough demand data."
        )
    if not (isinstance(order_quantity_eoq, (int,float)) and order_quantity_eoq > 0):
        raise ValueError("Order quantity (EOQ) must be a positive number.")
    if not (isinstance(reorder_point_rop, (int,float)) and reorder_point_rop >= 0):
        raise ValueError("Reorder point (ROP) must be a non-negative number.")
    if not (isinstance(lead_time_periods, int) and lead_time_periods >= 0):
        raise ValueError("Lead time must be a non-negative integer.")
    if not (isinstance(initial_inventory, (int,float)) and initial_inventory >= 0):
        raise ValueError("Initial inventory must be a non-negative number.")


    inventory_level = float(initial_inventory)
    # Lists to store simulation results for each period
    inventory_start_list, inventory_end_list, demand_list_sim = [], [], []
    order_placed_list, order_arrival_period_list, order_arriving_now_list = [], [], []
    stockout_units_list, units_sold_list = [], []
    orders_in_pipeline: list[tuple[float, int]] = [] # Stores (quantity, arrival_period_index)

    print(f"\n--- Running Inventory Simulation for {simulation_periods} periods ---")
    print(f"Parameters - Initial Inventory: {initial_inventory:.0f}, EOQ: {order_quantity_eoq:.0f}, "
          f"ROP: {reorder_point_rop:.0f}, Lead Time: {lead_time_periods} periods")

    for period_idx in range(simulation_periods): # 0-indexed period
        current_demand_val = 0.0
        # Access demand for the current period carefully based on type
        if isinstance(demand_series, pd.Series):
            current_demand_val = float(demand_series.iloc[period_idx])
        elif isinstance(demand_series, np.ndarray):
            current_demand_val = float(demand_series[period_idx])
        elif isinstance(demand_series, list):
            current_demand_val = float(demand_series[period_idx])
        # Ensure demand is non-negative
        current_demand_val = max(0, current_demand_val)

        demand_list_sim.append(current_demand_val)
        inventory_start_list.append(inventory_level)

        # Check for and process arriving orders
        arriving_qty_this_period = 0.0
        # Iterate backwards to safely remove items from orders_in_pipeline
        for i in range(len(orders_in_pipeline) - 1, -1, -1):
            order_qty, arrival_period_target_idx = orders_in_pipeline[i]
            if arrival_period_target_idx == period_idx: # Order arrives at the start of this period
                arriving_qty_this_period += order_qty
                orders_in_pipeline.pop(i) # Remove from pipeline

        order_arriving_now_list.append(arriving_qty_this_period)
        inventory_level += arriving_qty_this_period # Add to inventory

        # Satisfy demand
        sold_this_period = min(inventory_level, current_demand_val)
        units_sold_list.append(sold_this_period)
        stockout_this_period = max(0, current_demand_val - inventory_level)
        stockout_units_list.append(stockout_this_period)
        inventory_level -= sold_this_period # Reduce inventory by units sold

        inventory_end_list.append(inventory_level)

        # Check if an order needs to be placed (Q,R policy)
        # An order is placed if inventory level is at or below ROP AND
        # no other order is already scheduled to arrive within the lead time horizon.
        # This 'order_incoming_soon' check prevents placing multiple orders if lead time is long.
        order_is_pending_arrival_soon = any(
            arrival_idx <= period_idx + lead_time_periods for _, arrival_idx in orders_in_pipeline
        )

        placed_order_qty_this_period = 0.0
        order_due_period_logged = np.nan # For logging: period number (1-indexed) when order is due
        if inventory_level <= reorder_point_rop and not order_is_pending_arrival_soon:
            placed_order_qty_this_period = order_quantity_eoq
            # Order arrives at the start of `period_idx + lead_time_periods`
            arrival_target_idx = period_idx + lead_time_periods
            orders_in_pipeline.append((order_quantity_eoq, arrival_target_idx))
            order_due_period_logged = float(arrival_target_idx + 1) # Log 1-indexed period

        order_placed_list.append(placed_order_qty_this_period)
        order_arrival_period_list.append(order_due_period_logged)

    # Compile results into a DataFrame
    results_df = pd.DataFrame({
        'Period': range(1, simulation_periods + 1), # 1-indexed for display
        'Demand': demand_list_sim,
        'Inventory_Start_Period': inventory_start_list,
        'Order_Arriving_This_Period': order_arriving_now_list,
        'Inventory_After_Arrival': np.array(inventory_start_list) + np.array(order_arriving_now_list),
        'Units_Sold': units_sold_list,
        'Stockout_Units': stockout_units_list,
        'Inventory_End_Period': inventory_end_list,
        'Order_Placed_Qty': order_placed_list,
        'Order_Due_Start_of_Period': order_arrival_period_list # 1-indexed period when order is due
    })
    return results_df

if __name__ == '__main__': # pragma: no cover
    print("--- Testing Inventory Core Models ---")

    # Test EOQ
    annual_D = 10000.0
    cost_S_per_order = 50.0
    cost_H_per_unit_annual = 2.5
    eoq = calculate_eoq(annual_D, cost_S_per_order, cost_H_per_unit_annual)
    if eoq is not None:
        print(f"\n--- EOQ Calculation ---")
        print(f"  Annual Demand (D): {annual_D}")
        print(f"  Ordering Cost (S): ${cost_S_per_order:.2f}")
        print(f"  Holding Cost/Unit/Year (H): ${cost_H_per_unit_annual:.2f}")
        print(f"  Calculated EOQ: {eoq:.0f} units")

    # Test Safety Stock and ROP
    avg_demand_weekly = 100.0
    std_dev_demand_weekly = 20.0
    avg_lt_wks = 4.0
    std_dev_lt_wks = 1.0
    service_lvl_pct = 95.0

    safety_stock = calculate_safety_stock(
        avg_demand_weekly, std_dev_demand_weekly,
        avg_lt_wks, std_dev_lt_wks, service_lvl_pct
    )
    if safety_stock is not None:
        print(f"\n--- Safety Stock (SS) Calculation ---")
        print(f"  Avg Weekly Demand: {avg_demand_weekly}, StdDev Weekly Demand: {std_dev_demand_weekly}")
        print(f"  Avg Lead Time (Weeks): {avg_lt_wks}, StdDev Lead Time (Weeks): {std_dev_lt_wks}")
        print(f"  Target Service Level: {service_lvl_pct}%")
        print(f"  Calculated Safety Stock: {safety_stock:.0f} units")

        rop = calculate_reorder_point(avg_demand_weekly, avg_lt_wks, safety_stock)
        if rop is not None:
            print(f"\n--- Reorder Point (ROP) Calculation ---")
            print(f"  Avg Demand During Lead Time: {avg_demand_weekly * avg_lt_wks:.0f} units")
            print(f"  Calculated Reorder Point: {rop:.0f} units")

            # Test Inventory Simulation if all parameters are available
            if eoq is not None:
                print("\n--- Inventory Simulation Test ---")
                num_sim_periods = 52
                np.random.seed(42) # For reproducible random demand
                # Generate sample demand (ensure non-negative)
                sim_demand_np = np.maximum(0, np.random.normal(
                    loc=avg_demand_weekly, scale=std_dev_demand_weekly, size=num_sim_periods
                )).round().astype(float)

                sim_demand_list = list(sim_demand_np)
                sim_demand_series = pd.Series(sim_demand_np)

                initial_inv_for_sim = rop # Start simulation with ROP as initial inventory

                for demand_type_str, demand_data_for_sim in [
                    ("NumPy array", sim_demand_np),
                    ("list", sim_demand_list),
                    ("Pandas Series", sim_demand_series)
                ]:
                    print(f"\nSimulating with demand_series as {demand_type_str}...")
                    try:
                        simulation_df = run_inventory_simulation(
                            demand_series=demand_data_for_sim,
                            initial_inventory=initial_inv_for_sim,
                            order_quantity_eoq=eoq,
                            reorder_point_rop=rop,
                            lead_time_periods=int(round(avg_lt_wks)), # Must be int
                            simulation_periods=num_sim_periods
                        )
                        if not simulation_df.empty:
                            print(f"Simulation Results (demand as {demand_type_str}) - First 5 periods:")
                            print(simulation_df.head(5))
                            print(f"  Total Stockout Units over {num_sim_periods} weeks: "
                                  f"{simulation_df['Stockout_Units'].sum():.0f}")
                            print(f"  Number of Orders Placed: "
                                  f"{len(simulation_df[simulation_df['Order_Placed_Qty'] > 0])}")
                        else:
                            print(f"  Simulation with {demand_type_str} demand did not produce results.")
                    except (TypeError, ValueError) as e_sim:
                        print(f"  Error during simulation with {demand_type_str} demand: {e_sim}")
    