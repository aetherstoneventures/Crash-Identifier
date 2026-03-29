"""Alert manager for coordinating all alert types."""

import logging
from typing import List, Dict, Any
from datetime import datetime

from .email_alert import EmailAlert
from .sms_alert import SMSAlert


class AlertManager:
    """Manages all alert types and sends coordinated alerts."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.logger = logging.getLogger(__name__)
        self.email_alert = EmailAlert()
        self.sms_alert = SMSAlert()
        self.alert_history = []
    
    def send_crash_alert(self, probability: float, confidence: float,
                        email_recipients: List[str] = None,
                        sms_recipients: List[str] = None) -> Dict[str, bool]:
        """
        Send crash alert via all configured channels.
        
        Args:
            probability: Crash probability (0-1)
            confidence: Confidence interval (0-1)
            email_recipients: List of email addresses
            sms_recipients: List of phone numbers
            
        Returns:
            Dictionary with send status for each channel
        """
        results = {
            'email': False,
            'sms': False,
            'timestamp': datetime.now()
        }
        
        # Send email alerts
        if email_recipients:
            for recipient in email_recipients:
                try:
                    success = self.email_alert.send_crash_alert(
                        recipient, probability, confidence
                    )
                    results['email'] = results['email'] or success
                except Exception as e:
                    self.logger.error(f"Email alert failed: {e}")
        
        # Send SMS alerts
        if sms_recipients:
            for recipient in sms_recipients:
                try:
                    success = self.sms_alert.send_crash_alert(
                        recipient, probability, confidence
                    )
                    results['sms'] = results['sms'] or success
                except Exception as e:
                    self.logger.error(f"SMS alert failed: {e}")
        
        self.alert_history.append({
            'type': 'crash',
            'probability': probability,
            'results': results
        })
        
        return results
    
    def send_bottom_alert(self, days_to_bottom: int, recovery_date: str,
                         email_recipients: List[str] = None,
                         sms_recipients: List[str] = None) -> Dict[str, bool]:
        """
        Send bottom alert via all configured channels.
        
        Args:
            days_to_bottom: Predicted days to bottom
            recovery_date: Predicted recovery date
            email_recipients: List of email addresses
            sms_recipients: List of phone numbers
            
        Returns:
            Dictionary with send status for each channel
        """
        results = {
            'email': False,
            'sms': False,
            'timestamp': datetime.now()
        }
        
        # Send email alerts
        if email_recipients:
            for recipient in email_recipients:
                try:
                    success = self.email_alert.send_bottom_alert(
                        recipient, days_to_bottom, recovery_date
                    )
                    results['email'] = results['email'] or success
                except Exception as e:
                    self.logger.error(f"Email alert failed: {e}")
        
        # Send SMS alerts
        if sms_recipients:
            for recipient in sms_recipients:
                try:
                    success = self.sms_alert.send_bottom_alert(
                        recipient, days_to_bottom, recovery_date
                    )
                    results['sms'] = results['sms'] or success
                except Exception as e:
                    self.logger.error(f"SMS alert failed: {e}")
        
        self.alert_history.append({
            'type': 'bottom',
            'days_to_bottom': days_to_bottom,
            'results': results
        })
        
        return results
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert records
        """
        return self.alert_history[-limit:]

