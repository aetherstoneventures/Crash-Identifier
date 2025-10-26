"""Base alert class for all alert types."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
import logging


class BaseAlert(ABC):
    """Abstract base class for all alert types."""
    
    def __init__(self, alert_type: str):
        """
        Initialize base alert.
        
        Args:
            alert_type: Type of alert (email, sms, etc.)
        """
        self.alert_type = alert_type
        self.logger = logging.getLogger(__name__)
        self.sent_at = None
    
    @abstractmethod
    def send(self, recipient: str, subject: str, message: str) -> bool:
        """
        Send alert.
        
        Args:
            recipient: Recipient address/number
            subject: Alert subject
            message: Alert message
            
        Returns:
            True if sent successfully, False otherwise
        """
        pass
    
    def format_crash_alert(self, probability: float, confidence: float) -> str:
        """
        Format crash alert message.
        
        Args:
            probability: Crash probability (0-1)
            confidence: Confidence interval (0-1)
            
        Returns:
            Formatted alert message
        """
        return (
            f"🚨 MARKET CRASH ALERT 🚨\n\n"
            f"Crash Probability: {probability:.2%}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Please review the dashboard for more details."
        )
    
    def format_bottom_alert(self, days_to_bottom: int, recovery_date: str) -> str:
        """
        Format bottom alert message.
        
        Args:
            days_to_bottom: Predicted days to bottom
            recovery_date: Predicted recovery date
            
        Returns:
            Formatted alert message
        """
        return (
            f"📊 MARKET BOTTOM PREDICTION 📊\n\n"
            f"Days to Bottom: {days_to_bottom}\n"
            f"Recovery Date: {recovery_date}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Please review the dashboard for more details."
        )
    
    def log_alert(self, recipient: str, alert_type: str, status: str) -> None:
        """
        Log alert sending.
        
        Args:
            recipient: Recipient address/number
            alert_type: Type of alert
            status: Alert status (sent, failed, etc.)
        """
        self.logger.info(
            f"Alert {alert_type} to {recipient}: {status}"
        )

