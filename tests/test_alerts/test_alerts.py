"""Tests for alert system."""

import pytest
from unittest.mock import patch, MagicMock

from src.alerts.base_alert import BaseAlert
from src.alerts.email_alert import EmailAlert
from src.alerts.sms_alert import SMSAlert
from src.alerts.alert_manager import AlertManager


class TestBaseAlert:
    """Test BaseAlert class."""
    
    def test_format_crash_alert(self):
        """Test crash alert formatting."""
        # Create a concrete implementation for testing
        class ConcreteAlert(BaseAlert):
            def send(self, recipient, subject, message):
                return True

        alert = ConcreteAlert("test")
        message = alert.format_crash_alert(0.75, 0.85)

        assert "CRASH ALERT" in message
        assert "75.00%" in message
        assert "85.00%" in message
    
    def test_format_bottom_alert(self):
        """Test bottom alert formatting."""
        class ConcreteAlert(BaseAlert):
            def send(self, recipient, subject, message):
                return True
        
        alert = ConcreteAlert("test")
        message = alert.format_bottom_alert(30, "2024-12-15")
        
        assert "BOTTOM PREDICTION" in message
        assert "30" in message
        assert "2024-12-15" in message


class TestEmailAlert:
    """Test EmailAlert class."""
    
    def test_initialization(self):
        """Test email alert initialization."""
        alert = EmailAlert()
        assert alert.alert_type == "email"
        assert alert.smtp_server == "smtp.gmail.com"
        assert alert.smtp_port == 587
    
    def test_send_without_credentials(self):
        """Test sending email without credentials."""
        alert = EmailAlert()
        alert.sender_email = None
        alert.sender_password = None
        
        result = alert.send("test@example.com", "Test", "Test message")
        assert result is False
    
    @patch('smtplib.SMTP')
    def test_send_with_credentials(self, mock_smtp):
        """Test sending email with credentials."""
        alert = EmailAlert()
        alert.sender_email = "sender@example.com"
        alert.sender_password = "password"
        
        # Mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = alert.send("test@example.com", "Test", "Test message")
        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()


class TestSMSAlert:
    """Test SMSAlert class."""
    
    def test_initialization(self):
        """Test SMS alert initialization."""
        alert = SMSAlert()
        assert alert.alert_type == "sms"
    
    def test_send_without_twilio(self):
        """Test sending SMS without Twilio."""
        alert = SMSAlert()
        alert.twilio_available = False
        
        result = alert.send("+1234567890", "Test", "Test message")
        assert result is False
    
    def test_send_without_credentials(self):
        """Test sending SMS without credentials."""
        alert = SMSAlert()
        alert.twilio_available = True
        alert.account_sid = None
        
        result = alert.send("+1234567890", "Test", "Test message")
        assert result is False


class TestAlertManager:
    """Test AlertManager class."""
    
    def test_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        assert manager.email_alert is not None
        assert manager.sms_alert is not None
        assert len(manager.alert_history) == 0
    
    def test_send_crash_alert_no_recipients(self):
        """Test sending crash alert without recipients."""
        manager = AlertManager()
        results = manager.send_crash_alert(0.75, 0.85)
        
        assert results['email'] is False
        assert results['sms'] is False
        assert len(manager.alert_history) == 1
    
    def test_send_bottom_alert_no_recipients(self):
        """Test sending bottom alert without recipients."""
        manager = AlertManager()
        results = manager.send_bottom_alert(30, "2024-12-15")
        
        assert results['email'] is False
        assert results['sms'] is False
        assert len(manager.alert_history) == 1
    
    def test_get_alert_history(self):
        """Test getting alert history."""
        manager = AlertManager()
        manager.send_crash_alert(0.75, 0.85)
        manager.send_bottom_alert(30, "2024-12-15")
        
        history = manager.get_alert_history()
        assert len(history) == 2
        assert history[0]['type'] == 'crash'
        assert history[1]['type'] == 'bottom'
    
    def test_alert_history_limit(self):
        """Test alert history limit."""
        manager = AlertManager()
        
        # Add 150 alerts
        for i in range(150):
            manager.send_crash_alert(0.5, 0.5)
        
        # Get last 100
        history = manager.get_alert_history(limit=100)
        assert len(history) == 100

