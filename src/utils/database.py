"""Database layer with SQLAlchemy ORM models."""

import logging
from datetime import datetime
from typing import Optional, List

from sqlalchemy import create_engine, Column, Integer, Float, Date, String, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.utils.config import DATABASE_URL

logger = logging.getLogger(__name__)
Base = declarative_base()


class Indicator(Base):
    """ORM model for economic and market indicators."""

    __tablename__ = 'indicators'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True, index=True)

    # Yield curve (3)
    yield_10y_3m = Column(Float)
    yield_10y_2y = Column(Float)
    yield_10y = Column(Float)

    # Credit (1)
    credit_spread_bbb = Column(Float)

    # Economic (5)
    unemployment_rate = Column(Float)
    real_gdp = Column(Float)
    cpi = Column(Float)
    fed_funds_rate = Column(Float)
    industrial_production = Column(Float)

    # Market (3)
    sp500_close = Column(Float)
    sp500_volume = Column(Float)
    vix_close = Column(Float)

    # Sentiment (1)
    consumer_sentiment = Column(Float)

    # Housing (1)
    housing_starts = Column(Float)

    # Monetary (1)
    m2_money_supply = Column(Float)

    # Debt (1)
    debt_to_gdp = Column(Float)

    # Savings (1)
    savings_rate = Column(Float)

    # Composite (1)
    lei = Column(Float)

    # Alternative data sources
    shiller_pe = Column(Float)
    margin_debt = Column(Float)
    put_call_ratio = Column(Float)

    # Calculated Indicators (28 total) - Financial Market (8)
    yield_spread_10y_3m = Column(Float)
    yield_spread_10y_2y = Column(Float)
    vix_level = Column(Float)
    vix_change_rate = Column(Float)
    realized_volatility = Column(Float)
    sp500_momentum_200d = Column(Float)
    sp500_drawdown = Column(Float)

    # Credit Cycle Indicators (6)
    debt_service_ratio = Column(Float)
    credit_gap = Column(Float)
    corporate_debt_growth = Column(Float)
    household_debt_growth = Column(Float)
    m2_growth = Column(Float)

    # Valuation Indicators (4)
    buffett_indicator = Column(Float)
    sp500_pb_ratio = Column(Float)
    earnings_yield_spread = Column(Float)

    # Sentiment Indicators (5)
    put_call_ratio_calc = Column(Float)  # Calculated version
    margin_debt_growth = Column(Float)
    market_breadth = Column(Float)

    # Economic Indicators (5)
    sahm_rule = Column(Float)
    gdp_growth = Column(Float)
    industrial_production_growth = Column(Float)
    housing_starts_growth = Column(Float)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_quality_score = Column(Float, default=1.0)

    __table_args__ = (
        Index('idx_date', 'date'),
    )

    def __repr__(self) -> str:
        return f"<Indicator(date={self.date}, sp500_close={self.sp500_close})>"


class CrashEvent(Base):
    """ORM model for historical crash events."""

    __tablename__ = 'crash_events'

    id = Column(Integer, primary_key=True)
    start_date = Column(Date, nullable=False, index=True)
    end_date = Column(Date, nullable=False)
    trough_date = Column(Date, nullable=False)
    recovery_date = Column(Date)
    max_drawdown = Column(Float, nullable=False)
    recovery_months = Column(Integer)
    crash_type = Column(String(50))
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CrashEvent(start={self.start_date}, max_dd={self.max_drawdown})>"


class Prediction(Base):
    """ORM model for model predictions."""

    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    prediction_date = Column(Date, nullable=False, index=True)
    crash_probability = Column(Float, nullable=False)
    bottom_prediction_date = Column(Date)
    recovery_prediction_date = Column(Date)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    model_version = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Prediction(date={self.prediction_date}, crash_prob={self.crash_probability})>"


class AlertHistory(Base):
    """ORM model for alert history (Phase 6)."""

    __tablename__ = 'alert_history'

    id = Column(Integer, primary_key=True)
    alert_date = Column(DateTime, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    condition_met = Column(String, nullable=False)
    recipients = Column(String, nullable=False)
    delivery_status = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<AlertHistory(date={self.alert_date}, type={self.alert_type}, status={self.delivery_status})>"


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, db_url: str = DATABASE_URL):
        """
        Initialize database manager.

        Args:
            db_url: Database URL (default from config)
        """
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)

    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
        self.logger.info("Database connection closed")

    def add_indicator(self, session: Session, indicator: Indicator) -> None:
        """
        Add or update an indicator record.

        Args:
            session: Database session
            indicator: Indicator object
        """
        try:
            existing = session.query(Indicator).filter_by(date=indicator.date).first()
            if existing:
                for key, value in indicator.__dict__.items():
                    if not key.startswith('_'):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                session.add(indicator)
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to add indicator: {e}")
            raise

    def get_indicators(
        self,
        session: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Indicator]:
        """
        Retrieve indicators within date range.

        Args:
            session: Database session
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of Indicator objects
        """
        query = session.query(Indicator)
        if start_date:
            query = query.filter(Indicator.date >= start_date)
        if end_date:
            query = query.filter(Indicator.date <= end_date)
        return query.order_by(Indicator.date).all()

