"""SMS alert implementation."""

import os
from typing import Optional

from .base_alert import BaseAlert


class SMSAlert(BaseAlert):
    """SMS alert implementation using Twilio."""
    
    def __init__(self):
        """Initialize SMS alert."""
        super().__init__("sms")
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        
        # Try to import Twilio if available
        try:
            from twilio.rest import Client
            self.client = Client(self.account_sid, self.auth_token)
            self.twilio_available = True
        except ImportError:
            self.logger.warning("Twilio not installed. SMS alerts disabled.")
            self.twilio_available = False
    
    def send(self, recipient: str, subject: str, message: str) -> bool:
        """
        Send SMS alert.
        
        Args:
            recipient: Recipient phone number
            subject: SMS subject (not used for SMS)
            message: SMS message body
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.twilio_available:
            self.logger.warning("Twilio not available")
            return False
        
        if not self.account_sid or not self.auth_token or not self.from_number:
            self.logger.warning("Twilio credentials not configured")
            return False
        
        try:
            # Limit message to 160 characters for SMS
            sms_message = message[:160]
            
            # Send SMS
            msg = self.client.messages.create(
                body=sms_message,
                from_=self.from_number,
                to=recipient
            )
            
            self.log_alert(recipient, "sms", "sent")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to send SMS: {e}")
            self.log_alert(recipient, "sms", f"failed: {e}")
            return False
    
    def send_crash_alert(self, recipient: str, probability: float,
                        confidence: float) -> bool:
        """
        Send crash alert SMS.
        
        Args:
            recipient: Recipient phone number
            probability: Crash probability
            confidence: Confidence interval
            
        Returns:
            True if sent successfully
        """
        message = (
            f"🚨 CRASH ALERT: {probability:.0%} probability. "
            f"Check dashboard for details."
        )
        return self.send(recipient, "", message)
    
    def send_bottom_alert(self, recipient: str, days_to_bottom: int,
                         recovery_date: str) -> bool:
        """
        Send bottom alert SMS.
        
        Args:
            recipient: Recipient phone number
            days_to_bottom: Predicted days to bottom
            recovery_date: Predicted recovery date
            
        Returns:
            True if sent successfully
        """
        message = (
            f"📊 BOTTOM: {days_to_bottom} days. "
            f"Recovery: {recovery_date}. Check dashboard."
        )
        return self.send(recipient, "", message)

