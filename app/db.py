"""
ShelfMind AI — SQLite Database Layer
Production-ready data storage for products, planograms, compliance logs, and alerts.
"""

import sqlite3
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Database file location
DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "shelfmind.db"

# Ensure directory exists
DB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    """Thread-safe database connection context manager."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")       # Better concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")          # Enforce FK constraints
    conn.row_factory = sqlite3.Row                  # Dict-like row access
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT DEFAULT 'Other',
                price REAL DEFAULT 0,
                image_path TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS planograms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                reference_image_path TEXT,
                n_shelves INTEGER DEFAULT 1,
                total_products INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS planogram_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                planogram_id INTEGER NOT NULL,
                shelf_level INTEGER NOT NULL,
                position INTEGER NOT NULL,
                product_sku TEXT NOT NULL,
                product_name TEXT,
                confidence REAL DEFAULT 0,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                FOREIGN KEY (planogram_id) REFERENCES planograms(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS compliance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                planogram_name TEXT NOT NULL,
                overall_compliance REAL,
                total_detected INTEGER,
                total_expected INTEGER,
                revenue_at_risk REAL DEFAULT 0,
                alert_count INTEGER DEFAULT 0,
                scan_number INTEGER,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                compliance_log_id INTEGER,
                alert_type TEXT NOT NULL,
                shelf_id INTEGER,
                product_name TEXT,
                product_sku TEXT,
                priority TEXT DEFAULT 'MEDIUM',
                expected_count INTEGER,
                found_count INTEGER,
                revenue_at_risk REAL DEFAULT 0,
                position_info TEXT,
                notified INTEGER DEFAULT 0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (compliance_log_id) REFERENCES compliance_logs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_compliance_time ON compliance_logs(recorded_at);
            CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
            CREATE INDEX IF NOT EXISTS idx_planogram_positions ON planogram_positions(planogram_id);
        """)
        # Migration: add barcode column if missing (safe for both old and new DBs)
        try:
            conn.execute("ALTER TABLE products ADD COLUMN barcode TEXT")
        except Exception:
            pass  # Column already exists
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_barcode ON products(barcode)")


# ══════════════════════════════════════════════════════════════════════════
# ── PRODUCTS CRUD ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _serialize_embedding(embedding):
    """Convert embedding list to bytes for storage."""
    if embedding is None:
        return None
    return np.array(embedding, dtype=np.float32).tobytes()


def _deserialize_embedding(blob):
    """Convert bytes back to embedding list."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


def add_product(sku, name, category, price, image_path, embedding, barcode=None):
    """Add a product to the database."""
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO products (sku, name, category, price, barcode, image_path, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sku, name, category, price, barcode, str(image_path), _serialize_embedding(embedding))
        )
    return sku


def get_products():
    """Get all products as a list of dicts (compatible with old catalog format)."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM products ORDER BY id").fetchall()
    products = []
    for row in rows:
        products.append({
            "id": row["id"],
            "sku": row["sku"],
            "name": row["name"],
            "category": row["category"],
            "price": row["price"],
            "barcode": row["barcode"] if "barcode" in row.keys() else None,
            "image_path": row["image_path"],
            "embedding": _deserialize_embedding(row["embedding"]),
            "created_at": row["created_at"],
        })
    return products


def get_product_count():
    """Get total number of registered products."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM products WHERE embedding IS NOT NULL").fetchone()
    return row["cnt"]


def get_next_product_id():
    """Get the next product ID for SKU generation."""
    with get_connection() as conn:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM products").fetchone()
    return row["next_id"]


def delete_product(sku):
    """Delete a product by SKU."""
    with get_connection() as conn:
        conn.execute("DELETE FROM products WHERE sku = ?", (sku,))


def clear_all_products():
    """Delete all products."""
    with get_connection() as conn:
        conn.execute("DELETE FROM products")


def get_catalog_as_dict():
    """Return catalog in the old JSON format for backward compatibility."""
    products = get_products()
    next_id = get_next_product_id()
    return {"products": products, "next_id": next_id}


# ══════════════════════════════════════════════════════════════════════════
# ── PLANOGRAMS CRUD ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def save_planogram_db(name, data):
    """Save a planogram and its positions to the database."""
    with get_connection() as conn:
        # Delete existing planogram with same name
        old = conn.execute("SELECT id FROM planograms WHERE name = ?", (name,)).fetchone()
        if old:
            conn.execute("DELETE FROM planograms WHERE id = ?", (old["id"],))

        # Insert planogram
        cursor = conn.execute(
            """INSERT INTO planograms (name, reference_image_path, n_shelves, total_products, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                name,
                data.get("reference_image_path", ""),
                data.get("n_shelves", 1),
                data.get("total_products", 0),
                data.get("created_at", datetime.now().isoformat()),
            )
        )
        planogram_id = cursor.lastrowid

        # Insert positions
        for shelf in data.get("shelves", []):
            level = shelf.get("level", 1)
            for product in shelf.get("products", []):
                bbox = product.get("bbox", [0, 0, 0, 0])
                conn.execute(
                    """INSERT INTO planogram_positions
                       (planogram_id, shelf_level, position, product_sku, product_name, confidence,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        planogram_id, level,
                        product.get("position", 0),
                        product.get("sku", "UNKNOWN"),
                        product.get("name", "Unknown"),
                        product.get("confidence", 0),
                        bbox[0] if len(bbox) > 0 else 0,
                        bbox[1] if len(bbox) > 1 else 0,
                        bbox[2] if len(bbox) > 2 else 0,
                        bbox[3] if len(bbox) > 3 else 0,
                    )
                )
    return planogram_id


def get_planograms():
    """Get all planograms in the old dict format for backward compatibility."""
    planograms = {}
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM planograms ORDER BY id").fetchall()
        for row in rows:
            planogram_id = row["id"]
            positions = conn.execute(
                "SELECT * FROM planogram_positions WHERE planogram_id = ? ORDER BY shelf_level, position",
                (planogram_id,)
            ).fetchall()

            # Group positions by shelf
            shelves = {}
            for pos in positions:
                level = pos["shelf_level"]
                if level not in shelves:
                    shelves[level] = {
                        "level": level,
                        "product_count": 0,
                        "products": [],
                    }
                shelves[level]["products"].append({
                    "position": pos["position"],
                    "sku": pos["product_sku"],
                    "name": pos["product_name"],
                    "confidence": pos["confidence"],
                    "bbox": [pos["bbox_x1"], pos["bbox_y1"], pos["bbox_x2"], pos["bbox_y2"]],
                })
                shelves[level]["product_count"] += 1

            planograms[row["name"]] = {
                "name": row["name"],
                "created_at": row["created_at"],
                "n_shelves": row["n_shelves"],
                "total_products": row["total_products"],
                "shelves": [shelves[k] for k in sorted(shelves.keys())],
            }
    return planograms


def delete_planogram(name):
    """Delete a planogram by name (CASCADE deletes positions)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM planograms WHERE name = ?", (name,))


# ══════════════════════════════════════════════════════════════════════════
# ── COMPLIANCE LOGS CRUD ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def log_compliance(planogram_name, compliance, detected, expected, revenue_risk, alert_count, scan_number, shelf_data=None):
    """Log a compliance check result."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO compliance_logs
               (planogram_name, overall_compliance, total_detected, total_expected,
                revenue_at_risk, alert_count, scan_number)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (planogram_name, compliance, detected, expected, revenue_risk, alert_count, scan_number)
        )
        return cursor.lastrowid


def log_alert(compliance_log_id, alert_type, shelf_id, product_name, product_sku,
              priority, expected_count=None, found_count=None, revenue=0, position_info=None, notified=False):
    """Log an individual alert."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO alerts
               (compliance_log_id, alert_type, shelf_id, product_name, product_sku,
                priority, expected_count, found_count, revenue_at_risk, position_info, notified)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (compliance_log_id, alert_type, shelf_id, product_name, product_sku,
             priority, expected_count, found_count, revenue, position_info, 1 if notified else 0)
        )


def get_compliance_logs(limit=200):
    """Get recent compliance logs."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM compliance_logs ORDER BY recorded_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_compliance_logs_as_list():
    """Return compliance logs in the old JSON format for backward compatibility."""
    logs = get_compliance_logs()
    result = []
    for log in reversed(logs):  # Oldest first
        result.append({
            "timestamp": log["recorded_at"],
            "planogram": log["planogram_name"],
            "overall_compliance": log["overall_compliance"],
            "total_detected": log["total_detected"],
            "total_expected": log["total_expected"],
            "revenue_at_risk": log["revenue_at_risk"],
            "alerts": log["alert_count"],
        })
    return result


def get_alerts_history(limit=100):
    """Get recent alerts with details."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT a.*, cl.planogram_name, cl.overall_compliance, cl.scan_number
               FROM alerts a
               LEFT JOIN compliance_logs cl ON a.compliance_log_id = cl.id
               ORDER BY a.recorded_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_analytics_summary():
    """Get aggregated analytics for the dashboard."""
    with get_connection() as conn:
        stats = {}

        # Total scans
        row = conn.execute("SELECT COUNT(*) as cnt FROM compliance_logs").fetchone()
        stats["total_scans"] = row["cnt"]

        # Average compliance
        row = conn.execute("SELECT AVG(overall_compliance) as avg_comp FROM compliance_logs").fetchone()
        stats["avg_compliance"] = round(row["avg_comp"] or 0, 1)

        # Total alerts by type
        rows = conn.execute(
            "SELECT alert_type, COUNT(*) as cnt FROM alerts GROUP BY alert_type"
        ).fetchall()
        stats["alerts_by_type"] = {row["alert_type"]: row["cnt"] for row in rows}

        # Total revenue at risk
        row = conn.execute("SELECT SUM(revenue_at_risk) as total FROM compliance_logs").fetchone()
        stats["total_revenue_at_risk"] = round(row["total"] or 0, 2)

        # Recent compliance trend (last 50 entries)
        rows = conn.execute(
            "SELECT overall_compliance, recorded_at FROM compliance_logs ORDER BY recorded_at DESC LIMIT 50"
        ).fetchall()
        stats["compliance_trend"] = [{"compliance": r["overall_compliance"], "time": r["recorded_at"]} for r in reversed(rows)]

        # Alert frequency by hour
        rows = conn.execute(
            """SELECT strftime('%H', recorded_at) as hour, COUNT(*) as cnt
               FROM alerts GROUP BY hour ORDER BY hour"""
        ).fetchall()
        stats["alerts_by_hour"] = {row["hour"]: row["cnt"] for row in rows}

        # Top offending products
        rows = conn.execute(
            """SELECT product_name, alert_type, COUNT(*) as cnt
               FROM alerts GROUP BY product_name, alert_type
               ORDER BY cnt DESC LIMIT 10"""
        ).fetchall()
        stats["top_offenders"] = [dict(row) for row in rows]

    return stats


# ══════════════════════════════════════════════════════════════════════════
# ── MIGRATION: JSON → SQLite ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def migrate_from_json():
    """Auto-migrate existing JSON data to SQLite on first run."""
    root = Path(__file__).resolve().parent.parent
    migrated = False

    # Migrate products.json
    catalog_path = root / "data" / "store_catalog" / "products.json"
    if catalog_path.exists():
        try:
            with open(catalog_path) as f:
                catalog = json.load(f)
            for p in catalog.get("products", []):
                if "embedding" in p:
                    add_product(
                        sku=p.get("sku", f"SKU_{p.get('id', 0):04d}"),
                        name=p.get("name", "Unknown"),
                        category=p.get("category", "Other"),
                        price=p.get("price", 0),
                        image_path=p.get("image_path", p.get("image", "")),
                        embedding=p.get("embedding"),
                    )
            # Rename old file
            catalog_path.rename(catalog_path.with_suffix(".json.bak"))
            migrated = True
        except Exception as e:
            print(f"Warning: Could not migrate products.json: {e}")

    # Migrate planogram JSONs
    planogram_dir = root / "data" / "store_planograms"
    if planogram_dir.exists():
        for f in planogram_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                save_planogram_db(data.get("name", f.stem), data)
                f.rename(f.with_suffix(".json.bak"))
                migrated = True
            except Exception as e:
                print(f"Warning: Could not migrate {f.name}: {e}")

    # Migrate compliance log
    log_path = root / "data" / "compliance_logs" / "compliance_log.json"
    if log_path.exists():
        try:
            with open(log_path) as f:
                logs = json.load(f)
            for entry in logs:
                log_compliance(
                    planogram_name=entry.get("planogram", "Unknown"),
                    compliance=entry.get("overall_compliance", 0),
                    detected=entry.get("total_detected", 0),
                    expected=entry.get("total_expected", 0),
                    revenue_risk=entry.get("revenue_at_risk", 0),
                    alert_count=entry.get("alerts", 0),
                    scan_number=0,
                )
            log_path.rename(log_path.with_suffix(".json.bak"))
            migrated = True
        except Exception as e:
            print(f"Warning: Could not migrate compliance_log.json: {e}")

    return migrated


# ══════════════════════════════════════════════════════════════════════════
# ── INITIALIZATION ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def setup_database():
    """Initialize database and run migrations."""
    init_db()
    # Check if old JSON files exist and migrate them
    root = Path(__file__).resolve().parent.parent
    catalog_path = root / "data" / "store_catalog" / "products.json"
    planogram_dir = root / "data" / "store_planograms"
    log_path = root / "data" / "compliance_logs" / "compliance_log.json"

    has_json = (
        catalog_path.exists() or
        any(planogram_dir.glob("*.json")) if planogram_dir.exists() else False or
        (log_path.exists() and log_path.stat().st_size > 2)
    )
    if has_json:
        migrated = migrate_from_json()
        if migrated:
            print("✅ Migrated JSON data to SQLite database")
    return True
