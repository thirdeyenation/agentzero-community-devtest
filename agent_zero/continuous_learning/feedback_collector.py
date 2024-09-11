# File: agent_zero/continuous_learning/feedback_collector.py

import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

class FeedbackCollector:
    def __init__(self, db_path: str = 'feedback.db'):
        self.db_path = db_path
        self._create_table()

    def _create_table(self) -> None:
        """Create the feedback table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def collect_feedback(self, interaction_id: str, rating: int, comment: Optional[str] = None) -> None:
        """
        Collect feedback for a specific interaction.
        
        :param interaction_id: Unique identifier for the interaction
        :param rating: Numeric rating (e.g., 1-5)
        :param comment: Optional comment provided with the feedback
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (interaction_id, rating, comment, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (interaction_id, rating, comment, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_feedback(self, interaction_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve feedback, optionally filtered by interaction_id.
        
        :param interaction_id: If provided, retrieve feedback for this specific interaction
        :return: List of feedback entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if interaction_id:
            cursor.execute('SELECT * FROM feedback WHERE interaction_id = ?', (interaction_id,))
        else:
            cursor.execute('SELECT * FROM feedback')
        
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

    def get_average_rating(self) -> float:
        """
        Calculate the average rating across all feedback.
        
        :return: Average rating
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT AVG(rating) FROM feedback')
        avg_rating = cursor.fetchone()[0]
        conn.close()
        return float(avg_rating) if avg_rating else 0.0

    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all collected feedback.
        
        :return: Dictionary containing feedback summary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(rating) FROM feedback')
        avg_rating = cursor.fetchone()[0]
        
        cursor.execute('SELECT rating, COUNT(*) FROM feedback GROUP BY rating')
        rating_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_feedback': total_feedback,
            'average_rating': float(avg_rating) if avg_rating else 0.0,
            'rating_distribution': rating_distribution
        }

    def clear_feedback(self) -> None:
        """
        Clear all feedback data. Use with caution!
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM feedback')
        conn.commit()
        conn.close()