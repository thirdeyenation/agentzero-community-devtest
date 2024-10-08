# File: agent_zero/continuous_learning/interaction_logger.py

import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from agent_zero.security import anonymize_data

class InteractionLogger:
    def __init__(self, log_file: str = 'logs/interactions.log'):
        self.logger = logging.getLogger('interaction_logger')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_interaction(self, user_input: str, agent_response: str, metadata: Dict[str, Any] = None) -> None:
        """
        Log an interaction between the user and the agent.
        
        :param user_input: The input provided by the user
        :param agent_response: The response generated by the agent
        :param metadata: Additional metadata about the interaction (optional)
        """
        anonymized_input = anonymize_data(user_input)
        anonymized_response = anonymize_data(agent_response)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': anonymized_input,
            'agent_response': anonymized_response,
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(log_entry))

    def get_logs(self, start_date: datetime = None, end_date: datetime = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve logs within a specified date range.
        
        :param start_date: The start date for log retrieval (optional)
        :param end_date: The end date for log retrieval (optional)
        :param limit: The maximum number of logs to retrieve (optional)
        :return: A list of log entries
        """
        logs = []
        with open(self.logger.handlers[0].baseFilename, 'r') as f:
            for line in f:
                log_entry = json.loads(line.split(' - ', 1)[1])
                log_timestamp = datetime.fromisoformat(log_entry['timestamp'])
                
                if start_date and log_timestamp < start_date:
                    continue
                if end_date and log_timestamp > end_date:
                    break
                
                logs.append(log_entry)
                
                if limit and len(logs) >= limit:
                    break
        
        return logs

    def analyze_interactions(self) -> Dict[str, Any]:
        """
        Perform basic analysis on the logged interactions.
        
        :return: A dictionary containing analysis results
        """
        logs = self.get_logs()
        total_interactions = len(logs)
        avg_input_length = sum(len(log['user_input']) for log in logs) / total_interactions if total_interactions > 0 else 0
        avg_response_length = sum(len(log['agent_response']) for log in logs) / total_interactions if total_interactions > 0 else 0
        
        return {
            'total_interactions': total_interactions,
            'avg_input_length': avg_input_length,
            'avg_response_length': avg_response_length
        }

    def clear_logs(self) -> None:
        """
        Clear all logged interactions.
        """
        with open(self.logger.handlers[0].baseFilename, 'w'):
            pass