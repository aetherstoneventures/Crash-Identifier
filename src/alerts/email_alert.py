"""Email alert implementation."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import os

from .base_alert import BaseAlert


class EmailAlert(BaseAlert):
    """Email alert implementation."""
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        """
        Initialize email alert.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
        """
        super().__init__("email")
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = os.getenv("ALERT_EMAIL_SENDER")
        self.sender_password = os.getenv("ALERT_EMAIL_PASSWORD")
    
    def send(self, recipient: str, subject: str, message: str) -> bool:
        """
        Send email alert.
        
        Args:
            recipient: Recipient email address
            subject: Email subject
            message: Email message body
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.sender_email or not self.sender_password:
            self.logger.warning("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            self.log_alert(recipient, "email", "sent")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            self.log_alert(recipient, "email", f"failed: {e}")
            return False
    
    def send_crash_alert(self, recipient: str, probability: float, 
                        confidence: float) -> bool:
        """
        Send crash alert email.
        
        Args:
            recipient: Recipient email address
            probability: Crash probability
            confidence: Confidence interval
            
        Returns:
            True if sent successfully
        """
        subject = "🚨 Market Crash Alert"
        message = self.format_crash_alert(probability, confidence)
        return self.send(recipient, subject, message)
    
    def send_bottom_alert(self, recipient: str, days_to_bottom: int,
                         recovery_date: str) -> bool:
        """
        Send bottom alert email.
        
        Args:
            recipient: Recipient email address
            days_to_bottom: Predicted days to bottom
            recovery_date: Predicted recovery date
            
        Returns:
            True if sent successfully
        """
        subject = "📊 Market Bottom Prediction"
        message = self.format_bottom_alert(days_to_bottom, recovery_date)
        return self.send(recipient, subject, message)

