"""Tests for database layer."""

import pytest
import tempfile
from datetime import datetime, date
from pathlib import Path

from src.utils.database import DatabaseManager, Indicator, CrashEvent, Prediction


class TestDatabaseManager:
    """Test cases for DatabaseManager."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db_url = f"sqlite:///{db_path}"
            yield db_url

    @pytest.fixture
    def db_manager(self, temp_db):
        """Create DatabaseManager with temporary database."""
        manager = DatabaseManager(db_url=temp_db)
        manager.create_tables()
        return manager

    def test_init_creates_engine(self, temp_db):
        """Test initialization creates SQLAlchemy engine."""
        manager = DatabaseManager(db_url=temp_db)
        assert manager.engine is not None
        assert manager.SessionLocal is not None

    def test_create_tables(self, db_manager):
        """Test table creation."""
        with db_manager.get_session() as session:
            # Should not raise any errors
            assert session is not None

    def test_get_session(self, db_manager):
        """Test getting database session."""
        with db_manager.get_session() as session:
            assert session is not None

    def test_add_indicator_new(self, db_manager):
        """Test adding new indicator record."""
        with db_manager.get_session() as session:

            indicator = Indicator(
                date=date(2023, 1, 1),
                yield_10y_3m=1.5,
                unemployment_rate=5.0,
                sp500_close=4000.0,
                vix_close=20.0
            )

            db_manager.add_indicator(session, indicator)

        # Verify it was added
        with db_manager.get_session() as session:
            result = session.query(Indicator).filter_by(date=date(2023, 1, 1)).first()
            assert result is not None
            assert result.yield_10y_3m == 1.5

    def test_add_indicator_update(self, db_manager):
        """Test updating existing indicator record."""
        with db_manager.get_session() as session:

            # Add initial record
            indicator1 = Indicator(
                date=date(2023, 1, 1),
                yield_10y_3m=1.5,
                unemployment_rate=5.0
            )
            db_manager.add_indicator(session, indicator1)

        # Update record
        with db_manager.get_session() as session:
            indicator2 = Indicator(
                date=date(2023, 1, 1),
                yield_10y_3m=1.6,
                unemployment_rate=5.1
            )
            db_manager.add_indicator(session, indicator2)

        # Verify update
        with db_manager.get_session() as session:
            result = session.query(Indicator).filter_by(date=date(2023, 1, 1)).first()
            assert result.yield_10y_3m == 1.6
            assert result.unemployment_rate == 5.1

    def test_get_indicators_all(self, db_manager):
        """Test retrieving all indicators."""
        with db_manager.get_session() as session:

            # Add multiple records
            for i in range(5):
                indicator = Indicator(
                    date=date(2023, 1, i + 1),
                    sp500_close=4000 + i * 10
                )
                db_manager.add_indicator(session, indicator)

        # Retrieve all
        with db_manager.get_session() as session:
            results = db_manager.get_indicators(session)
            assert len(results) == 5

    def test_get_indicators_date_range(self, db_manager):
        """Test retrieving indicators within date range."""
        with db_manager.get_session() as session:

            # Add records
            for i in range(10):
                indicator = Indicator(
                    date=date(2023, 1, i + 1),
                    sp500_close=4000 + i * 10
                )
                db_manager.add_indicator(session, indicator)

        # Retrieve range
        with db_manager.get_session() as session:
            results = db_manager.get_indicators(
                session,
                start_date=date(2023, 1, 3),
                end_date=date(2023, 1, 7)
            )
            assert len(results) == 5
            assert results[0].date == date(2023, 1, 3)
            assert results[-1].date == date(2023, 1, 7)

    def test_indicator_model_repr(self):
        """Test Indicator model string representation."""
        indicator = Indicator(
            date=date(2023, 1, 1),
            sp500_close=4000.0
        )
        repr_str = repr(indicator)
        assert "Indicator" in repr_str
        assert "2023-01-01" in repr_str

    def test_crash_event_model(self, db_manager):
        """Test CrashEvent model."""
        with db_manager.get_session() as session:

            crash = CrashEvent(
                start_date=date(2008, 9, 1),
                end_date=date(2009, 3, 1),
                trough_date=date(2009, 3, 9),
                max_drawdown=-0.57,
                crash_type="Financial Crisis"
            )

            session.add(crash)
            session.commit()

            result = session.query(CrashEvent).first()
            assert result.max_drawdown == -0.57

    def test_prediction_model(self, db_manager):
        """Test Prediction model."""
        with db_manager.get_session() as session:

            prediction = Prediction(
                prediction_date=date(2023, 1, 1),
                crash_probability=0.25,
                model_version="1.0"
            )

            session.add(prediction)
            session.commit()

            result = session.query(Prediction).first()
            assert result.crash_probability == 0.25

    def test_close_connection(self, db_manager):
        """Test closing database connection."""
        db_manager.close()
        # Should not raise any errors

    def test_multiple_sessions(self, db_manager):
        """Test creating multiple sessions."""
        with db_manager.get_session() as session1:
            with db_manager.get_session() as session2:
                assert session1 is not session2

    def test_indicator_all_fields(self, db_manager):
        """Test Indicator model with all fields."""
        with db_manager.get_session() as session:

            indicator = Indicator(
                date=date(2023, 1, 1),
                yield_10y_3m=1.5,
                yield_10y_2y=1.2,
                yield_10y=3.5,
                credit_spread_bbb=2.0,
                unemployment_rate=5.0,
                real_gdp=27000,
                cpi=300,
                fed_funds_rate=4.5,
                industrial_production=100,
                sp500_close=4000,
                sp500_volume=1000000,
                vix_close=20,
                consumer_sentiment=100,
                housing_starts=1500,
                m2_money_supply=20000,
                debt_to_gdp=120,
                savings_rate=5.0,
                lei=100,
                margin_debt=500,
                put_call_ratio=1.0,
                data_quality_score=0.95
            )

            db_manager.add_indicator(session, indicator)

        with db_manager.get_session() as session:
            result = session.query(Indicator).first()
            assert result.yield_10y_3m == 1.5
            assert result.unemployment_rate == 5.0
            assert result.sp500_close == 4000
            assert result.data_quality_score == 0.95

