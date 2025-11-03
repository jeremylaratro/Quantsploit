"""
Database management for Quantsploit
"""

import sqlite3
import json
from typing import Optional, Dict, Any
from datetime import datetime
import os


class Database:
    """
    SQLite database for caching market data and storing analysis results
    """

    def __init__(self, db_path: str = "./quantsploit.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database and create tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Market data cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                period TEXT NOT NULL,
                interval TEXT NOT NULL,
                data TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, period, interval)
            )
        """)

        # Analysis results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                parameters TEXT,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Watchlist table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                notes TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def cache_market_data(self, symbol: str, period: str, interval: str, data: str):
        """Cache market data"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO market_data_cache
            (symbol, period, interval, data, cached_at)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, period, interval, data, datetime.now()))
        self.conn.commit()

    def get_cached_data(self, symbol: str, period: str, interval: str,
                        max_age_seconds: int = 3600) -> Optional[str]:
        """Retrieve cached market data if not expired"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT data, cached_at FROM market_data_cache
            WHERE symbol = ? AND period = ? AND interval = ?
            AND (julianday('now') - julianday(cached_at)) * 86400 < ?
        """, (symbol, period, interval, max_age_seconds))

        row = cursor.fetchone()
        if row:
            return row['data']
        return None

    def save_analysis(self, module_name: str, symbol: str,
                     parameters: Dict[str, Any], results: Dict[str, Any]):
        """Save analysis results"""
        import numpy as np
        import pandas as pd

        def serialize_for_json(obj):
            """Convert numpy/pandas types to JSON-serializable types"""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_list()
            elif isinstance(obj, dict):
                return {k: serialize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_for_json(item) for item in obj]
            return obj

        cursor = self.conn.cursor()
        try:
            clean_params = serialize_for_json(parameters)
            clean_results = serialize_for_json(results)
            cursor.execute("""
                INSERT INTO analysis_results (module_name, symbol, parameters, results)
                VALUES (?, ?, ?, ?)
            """, (module_name, symbol, json.dumps(clean_params), json.dumps(clean_results)))
            self.conn.commit()
        except Exception as e:
            # Silently fail on database errors - don't block module execution
            pass

    def get_analysis_history(self, symbol: Optional[str] = None,
                            module_name: Optional[str] = None) -> list:
        """Retrieve analysis history"""
        cursor = self.conn.cursor()
        query = "SELECT * FROM analysis_results WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if module_name:
            query += " AND module_name = ?"
            params.append(module_name)

        query += " ORDER BY created_at DESC LIMIT 100"

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def add_to_watchlist(self, symbol: str, notes: str = ""):
        """Add symbol to watchlist"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO watchlist (symbol, notes) VALUES (?, ?)
            """, (symbol, notes))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_watchlist(self) -> list:
        """Get all watchlist symbols"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM watchlist ORDER BY added_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
