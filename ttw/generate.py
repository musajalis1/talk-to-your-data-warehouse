"""Generate a deterministic synthetic e-commerce warehouse in DuckDB.

The data is shaped to have realistic structure worth asking questions about:
growth over time, Q4 seasonality, a 2025 price increase, holiday discounting,
a channel mix that shifts toward mobile, and higher return rates on apparel.
"""

from __future__ import annotations

import bisect
import csv
import random
import tempfile
from datetime import date, timedelta
from pathlib import Path

import duckdb

REGIONS = {"West": 0.30, "Northeast": 0.22, "Midwest": 0.20, "South": 0.28}
SEGMENTS = {"Consumer": 0.70, "SMB": 0.22, "Enterprise": 0.08}
CATEGORY_MIX = {"Electronics": 0.30, "Home": 0.22, "Apparel": 0.20, "Outdoors": 0.15, "Beauty": 0.13}
SEASONALITY = {1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.00,
               7: 0.95, 8: 1.00, 9: 1.00, 10: 1.10, 11: 1.45, 12: 1.60}

# (product name, list price, unit cost)
CATALOG = {
    "Electronics": [("Wireless Earbuds", 79, 32), ("Bluetooth Speaker", 59, 22), ("USB-C Hub", 39, 14),
                    ("Mechanical Keyboard", 129, 55), ("4K Webcam", 99, 41), ("Smart Watch", 249, 120),
                    ("Portable Charger", 35, 12), ("Noise-Cancelling Headphones", 299, 140)],
    "Home": [("Ceramic Mug Set", 32, 9), ("Linen Sheet Set", 119, 38), ("Cast Iron Skillet", 45, 16),
             ("Scented Candle", 24, 6), ("Throw Blanket", 65, 21), ("Air Purifier", 179, 82),
             ("Knife Block Set", 149, 58), ("Coffee Grinder", 89, 34)],
    "Apparel": [("Merino T-Shirt", 48, 15), ("Rain Jacket", 139, 52), ("Running Shoes", 120, 48),
                ("Wool Socks 3-Pack", 22, 6), ("Denim Jeans", 88, 27), ("Fleece Hoodie", 74, 23),
                ("Leather Belt", 45, 12), ("Baseball Cap", 28, 7)],
    "Outdoors": [("Camping Tent", 229, 105), ("Hiking Backpack", 119, 46), ("Insulated Water Bottle", 34, 9),
                 ("Headlamp", 29, 10), ("Sleeping Bag", 149, 62), ("Trekking Poles", 79, 28),
                 ("Camp Stove", 69, 27), ("Hammock", 59, 18)],
    "Beauty": [("Vitamin C Serum", 38, 8), ("Moisturizer", 32, 7), ("Sunscreen SPF 50", 18, 4),
               ("Shampoo Bar", 14, 3), ("Hair Dryer", 99, 40), ("Lip Balm Set", 12, 2),
               ("Face Mask Pack", 25, 5), ("Fragrance", 85, 22)],
}

FIRST_NAMES = ["Ava", "Liam", "Maya", "Noah", "Zoe", "Ethan", "Priya", "Lucas", "Sofia", "Omar",
               "Hana", "Diego", "Chloe", "Kenji", "Amara", "Leo", "Nina", "Mateo", "Isla", "Ravi"]
LAST_NAMES = ["Nguyen", "Garcia", "Smith", "Patel", "Kim", "Johnson", "Lopez", "Chen", "Brown", "Okafor",
              "Rossi", "Silva", "Cohen", "Tanaka", "Miller", "Haddad", "Davis", "Wilson", "Ali", "Moore"]

SCHEMA = """
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    customer_name VARCHAR NOT NULL,
    signup_date   DATE NOT NULL,
    region        VARCHAR NOT NULL,
    segment       VARCHAR NOT NULL
);
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name VARCHAR NOT NULL,
    category     VARCHAR NOT NULL,
    list_price   DECIMAL(10, 2) NOT NULL,
    unit_cost    DECIMAL(10, 2) NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date  DATE NOT NULL,
    status      VARCHAR NOT NULL,   -- completed | returned | cancelled
    channel     VARCHAR NOT NULL    -- web | mobile | marketplace
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    product_id    INTEGER NOT NULL,
    quantity      INTEGER NOT NULL,
    unit_price    DECIMAL(10, 2) NOT NULL,  -- price charged, before discount
    discount      DECIMAL(4, 2) NOT NULL    -- fraction, e.g. 0.10 = 10% off
);
"""


def _pick(rng: random.Random, weights: dict) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def generate(db_path: str | Path, seed: int = 7, n_customers: int = 3000,
             base_orders_per_day: float = 28.0) -> dict:
    """Build the warehouse at db_path (overwriting it) and return row counts."""
    rng = random.Random(seed)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    products = []
    for category, items in CATALOG.items():
        for name, price, cost in items:
            products.append((len(products) + 1, name, category, price, cost))
    by_category = {c: [p for p in products if p[2] == c] for c in CATALOG}

    # Customers sign up between 2023 and 2025, skewed toward later dates.
    signup_start, signup_end = date(2023, 1, 1), date(2025, 12, 31)
    span = (signup_end - signup_start).days
    customers = []
    for cid in range(1, n_customers + 1):
        signup = signup_start + timedelta(days=int(span * rng.random() ** 0.8))
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        customers.append((cid, name, signup, _pick(rng, REGIONS), _pick(rng, SEGMENTS)))
    by_signup = sorted(customers, key=lambda c: c[2])
    signup_dates = [c[2] for c in by_signup]

    orders, items = [], []
    start, end = date(2024, 1, 1), date(2025, 12, 31)
    day = start
    while day <= end:
        progress = (day - start).days / (end - start).days
        expected = (base_orders_per_day * (1 + 0.35 * progress) * SEASONALITY[day.month]
                    * (1.1 if day.weekday() >= 5 else 1.0))
        n_orders = max(0, round(rng.gauss(expected, expected ** 0.5)))
        eligible = bisect.bisect_right(signup_dates, day)
        channel_mix = {"web": 0.55 - 0.12 * progress, "mobile": 0.30 + 0.12 * progress, "marketplace": 0.15}
        holiday = day.month in (11, 12)

        for _ in range(n_orders if eligible else 0):
            customer = by_signup[rng.randrange(eligible)]
            order_id = len(orders) + 1
            has_apparel = False
            for _ in range(rng.choices([1, 2, 3, 4], weights=[0.55, 0.25, 0.13, 0.07])[0]):
                product = rng.choice(by_category[_pick(rng, CATEGORY_MIX)])
                has_apparel |= product[2] == "Apparel"
                if customer[4] == "Enterprise":
                    quantity = rng.randint(1, 5)
                else:
                    quantity = rng.choices([1, 2, 3], weights=[0.8, 0.15, 0.05])[0]
                price = round(product[3] * (1.05 if day.year == 2025 else 1.0), 2)
                discount = rng.choices([0.0, 0.1, 0.2], weights=[0.5, 0.25, 0.25] if holiday else [0.78, 0.14, 0.08])[0]
                items.append((len(items) + 1, order_id, product[0], quantity, price, discount))

            roll = rng.random()
            return_prob = 0.05 + (0.06 if has_apparel else 0.0)
            status = "cancelled" if roll < 0.05 else "returned" if roll < 0.05 + return_prob else "completed"
            orders.append((order_id, customer[0], day, status, _pick(rng, channel_mix)))
        day += timedelta(days=1)

    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    with tempfile.TemporaryDirectory() as tmp:
        for table, rows in [("customers", customers), ("products", products),
                            ("orders", orders), ("order_items", items)]:
            path = Path(tmp) / f"{table}.csv"
            with path.open("w", newline="") as f:
                csv.writer(f).writerows(rows)
            con.execute(f"COPY {table} FROM '{path.as_posix()}' (HEADER false)")
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("customers", "products", "orders", "order_items")}
    con.close()
    return counts
