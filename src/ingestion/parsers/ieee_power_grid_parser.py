"""
IEEE Power Grid RRP Parser

Ingests IEEE standard power grid test cases (IEEE-14, IEEE-57, IEEE-118, IEEE-300)
via pandapower into RRP SQLite schema.

Known planar graphs with D_eff=2 ground truth for Fisher Diagnostic Suite validation.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd

import pandapower as pp
import networkx as nx


def load_ieee_case(case_name: str) -> tuple:
    """
    Load IEEE case via pandapower.

    Args:
        case_name: 'case14', 'case57', 'case118', 'case300'

    Returns:
        (buses_df, lines_df, gens_df)
    """
    net = getattr(pp.networks, case_name)()
    return net.bus, net.line, net.gen


def validate_connectivity(buses_df: pd.DataFrame, lines_df: pd.DataFrame) -> bool:
    """
    Check that all buses are reachable via transmission lines.

    Args:
        buses_df: pandapower bus DataFrame
        lines_df: pandapower line DataFrame

    Returns:
        True if bus network is fully connected, False otherwise
    """
    G = nx.Graph()

    # Add all bus nodes
    for bus_idx in buses_df.index:
        G.add_node(bus_idx)

    # Add edges from lines only (ignore generators for connectivity check)
    for _, line in lines_df.iterrows():
        from_bus = line['from_bus']
        to_bus = line['to_bus']
        G.add_edge(from_bus, to_bus)

    # Check if all buses with transmission lines are connected
    # (isolated buses are OK, as long as transmission network is connected)
    components = list(nx.connected_components(G))
    largest = max(components, key=len) if components else set()
    coverage = len(largest) / len(buses_df) if buses_df.shape[0] > 0 else 0

    if coverage >= 0.95:  # Allow up to 5% isolated buses
        return True
    else:
        print(f"WARNING: Only {coverage*100:.1f}% of buses connected via transmission lines")
        return False


def ingest_case_to_db(case_name: str, output_path: str) -> None:
    """
    Parse IEEE case and ingest into RRP SQLite.

    Args:
        case_name: 'case14', 'case57', 'case118', 'case300'
        output_path: Path to output .db file
    """
    # Load data
    buses_df, lines_df, gens_df = load_ieee_case(case_name)

    # Validate
    print(f"[{case_name}] Loaded: {len(buses_df)} buses, {len(lines_df)} lines, {len(gens_df)} gens")
    connectivity_ok = validate_connectivity(buses_df, lines_df)
    if not connectivity_ok:
        print(f"WARNING: {case_name} grid is not fully connected")

    # Create or replace DB
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create RRP schema (standard tables)
    cursor.execute('''
        CREATE TABLE entries (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            entry_type TEXT,
            domain TEXT,
            status TEXT,
            type_group TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE sections (
            entry_id TEXT,
            section_name TEXT,
            content TEXT,
            FOREIGN KEY(entry_id) REFERENCES entries(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE entry_properties (
            entry_id TEXT,
            property_name TEXT,
            property_value TEXT,
            FOREIGN KEY(entry_id) REFERENCES entries(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE links (
            source_id TEXT,
            target_id TEXT,
            link_type TEXT,
            description TEXT,
            confidence_tier TEXT,
            FOREIGN KEY(source_id) REFERENCES entries(id),
            FOREIGN KEY(target_id) REFERENCES entries(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE rrp_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    # Ingest metadata
    cursor.execute('INSERT INTO rrp_meta (key, value) VALUES (?, ?)',
                   ('case_name', case_name))
    cursor.execute('INSERT INTO rrp_meta (key, value) VALUES (?, ?)',
                   ('date_ingested', datetime.now().isoformat()))
    cursor.execute('INSERT INTO rrp_meta (key, value) VALUES (?, ?)',
                   ('num_buses', str(len(buses_df))))
    cursor.execute('INSERT INTO rrp_meta (key, value) VALUES (?, ?)',
                   ('num_lines', str(len(lines_df))))
    cursor.execute('INSERT INTO rrp_meta (key, value) VALUES (?, ?)',
                   ('num_gens', str(len(gens_df))))

    # Ingest bus entries
    for bus_idx, bus_row in buses_df.iterrows():
        bus_id = f'{case_name.upper()}_B{bus_idx}'
        bus_type = _get_bus_type(bus_row)

        # Entry
        cursor.execute('''
            INSERT OR IGNORE INTO entries
            (id, title, entry_type, domain, status, type_group)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            bus_id,
            f'Bus {bus_idx} ({bus_type})',
            'instantiation',
            'engineering',
            'complete',
            'GRID'
        ))

        # Bus Specification section
        spec_content = f"""Bus Number: {bus_idx}
Type: {bus_type}
Nominal Voltage: {bus_row.get('vn_kv', 'N/A')} kV
Base MVA: 100"""

        cursor.execute('''
            INSERT INTO sections (entry_id, section_name, content)
            VALUES (?, ?, ?)
        ''', (bus_id, 'Bus Specification', spec_content))

        # Electrical Parameters section
        param_content = f"""Voltage Magnitude: {bus_row.get('vm_pu', 1.0):.3f} pu
Voltage Angle: {bus_row.get('va_degree', 0.0):.1f} degrees"""

        cursor.execute('''
            INSERT INTO sections (entry_id, section_name, content)
            VALUES (?, ?, ?)
        ''', (bus_id, 'Electrical Parameters', param_content))

        # Properties
        props = [
            ('voltage_magnitude_pu', str(bus_row.get('vm_pu', 1.0))),
            ('voltage_angle_deg', str(bus_row.get('va_degree', 0.0))),
            ('nominal_voltage_kv', str(bus_row.get('vn_kv', 'unknown'))),
            ('bus_type', bus_type),
            ('slack', str(bus_row.get('type', 'load'))),
        ]

        for prop_name, prop_value in props:
            cursor.execute('''
                INSERT INTO entry_properties (entry_id, property_name, property_value)
                VALUES (?, ?, ?)
            ''', (bus_id, prop_name, prop_value))

    # Ingest generator entries and links
    gen_counter = 0
    for gen_idx, gen_row in gens_df.iterrows():
        gen_id = f'{case_name.upper()}_G{gen_idx}'
        gen_bus = int(gen_row['bus'])
        bus_id = f'{case_name.upper()}_B{gen_bus}'

        # Generator entry
        cursor.execute('''
            INSERT OR IGNORE INTO entries
            (id, title, entry_type, domain, status, type_group)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            gen_id,
            f'Generator {gen_idx} (Bus {gen_bus})',
            'instantiation',
            'engineering',
            'complete',
            'GRID_GEN'
        ))

        # Generator section
        gen_spec = f"""Generator ID: {gen_idx}
Connected Bus: {gen_bus}
Real Power: {gen_row.get('p_mw', 0):.2f} MW
Reactive Power: {gen_row.get('q_mvar', 0):.2f} Mvar"""

        cursor.execute('''
            INSERT INTO sections (entry_id, section_name, content)
            VALUES (?, ?, ?)
        ''', (gen_id, 'Generator Specification', gen_spec))

        # Generator to bus link
        cursor.execute('''
            INSERT INTO links
            (source_id, target_id, link_type, description, confidence_tier)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            gen_id,
            bus_id,
            'supplies_power_to',
            f'Generator {gen_idx} supplies power to Bus {gen_bus}',
            '1'
        ))

        gen_counter += 1

    # Ingest transmission lines
    for line_idx, line_row in lines_df.iterrows():
        from_bus = int(line_row['from_bus'])
        to_bus = int(line_row['to_bus'])

        from_id = f'{case_name.upper()}_B{from_bus}'
        to_id = f'{case_name.upper()}_B{to_bus}'

        # Transmission line link (bidirectional)
        r_pu = line_row.get('r_ohm_per_km', 0) if 'r_ohm_per_km' in line_row else line_row.get('r_pu', 0)
        x_pu = line_row.get('x_ohm_per_km', 0) if 'x_ohm_per_km' in line_row else line_row.get('x_pu', 0)

        desc = f'Line {from_bus}-{to_bus}: R={r_pu:.5f}, X={x_pu:.5f} pu, Sn={line_row.get("sn_mva", 0):.0f} MVA'

        # Forward link
        cursor.execute('''
            INSERT INTO links
            (source_id, target_id, link_type, description, confidence_tier)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            from_id,
            to_id,
            'transmits_power_to',
            desc,
            '1'
        ))

        # Backward link (transmission is bidirectional)
        cursor.execute('''
            INSERT INTO links
            (source_id, target_id, link_type, description, confidence_tier)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            to_id,
            from_id,
            'transmits_power_to',
            desc,
            '1'
        ))

    conn.commit()
    conn.close()

    # Summary
    cursor_check = sqlite3.connect(output_path).cursor()
    num_entries = cursor_check.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
    num_links = cursor_check.execute('SELECT COUNT(*) FROM links').fetchone()[0]
    cursor_check.close()

    print(f"[{case_name}] Ingested: {num_entries} entries, {num_links} links -> {output_path}")


def _get_bus_type(bus_row: pd.Series) -> str:
    """Determine bus type from pandapower bus row."""
    bus_type_val = bus_row.get('type', 'load')

    if bus_type_val == 'sl':
        return 'slack'
    elif bus_type_val == 'pv':
        return 'generator'
    elif bus_type_val == 'pq':
        return 'load'
    else:
        return str(bus_type_val)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python ieee_power_grid_parser.py <case_name> <output_db_path>")
        print("Example: python ieee_power_grid_parser.py case14 data/rrp/ieee_power_grid/rrp_ieee_power_grid_case14.db")
        sys.exit(1)

    case = sys.argv[1]
    output = sys.argv[2]
    ingest_case_to_db(case, output)
