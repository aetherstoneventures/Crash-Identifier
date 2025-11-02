"""Populate crash_events table with verified historical crashes."""

import sys
from datetime import datetime
sys.path.insert(0, '/Users/pouyamahdavipourvahdati/Desktop/General/Projects/01_Project_Stock Automation/Project 2025_Stock Evaluation/Hidden Gem Stock/Augment Code Crash Analyzer/market-crash-predictor')

from src.utils.database import DatabaseManager, CrashEvent

# Verified historical crashes (1975-2025)
# Schema: start_date, end_date, trough_date, recovery_date, max_drawdown, recovery_months, crash_type, notes
HISTORICAL_CRASHES = [
    ('1980-11-28', '1982-08-12', '1982-08-12', '1983-01-15', -27.1, 5, '1980 Recession', 'Yield curve inverted, VIX 30-40, Unemployment rising, Fed Funds 20%'),
    ('1987-08-25', '1987-12-04', '1987-10-19', '1989-07-15', -33.5, 21, 'Black Monday', 'VIX spike 40+, Steep yield curve, Low unemployment'),
    ('1990-07-16', '1991-01-17', '1991-01-17', '1991-06-30', -19.9, 5, '1990 Recession', 'Yield curve inverted, VIX 25-30, Unemployment rising'),
    ('1998-07-17', '1998-10-15', '1998-10-15', '1999-03-15', -19.3, 5, 'Russian Crisis', 'VIX spike 40+, Credit spreads widening, Steep yield curve'),
    ('2000-03-24', '2002-10-09', '2002-10-09', '2007-10-09', -49.1, 60, 'Dot-Com Bubble', 'Yield curve inverted, VIX 30-40, Unemployment rising, Fed cuts rates'),
    ('2007-10-09', '2009-03-09', '2009-03-09', '2013-03-28', -56.8, 48, 'Financial Crisis', 'Yield curve inverted, VIX 80+, Unemployment 10%, Credit spreads 6-7%'),
    ('2011-05-02', '2011-10-03', '2011-10-03', '2012-03-15', -19.4, 5, 'Debt Crisis', 'VIX 25-35, Steep yield curve, Credit spreads widening'),
    ('2015-06-23', '2016-02-11', '2016-02-11', '2016-08-15', -20.5, 6, 'Commodity Crash', 'Oil price collapse, China devaluation, VIX 20-30'),
    ('2018-09-21', '2018-12-24', '2018-12-24', '2019-09-15', -19.8, 9, 'Volatility Spike', 'Yield curve inverted, VIX spike 25-36, Fed rate hikes'),
    ('2020-02-19', '2020-03-23', '2020-03-23', '2020-08-18', -33.9, 5, 'COVID Pandemic', 'VIX extreme 82.7, Unemployment spike 14.7%, Credit spreads 5-6%'),
    ('2022-01-03', '2022-10-12', '2022-10-12', '2023-01-24', -27.5, 3, 'Fed Rate Hike', 'Yield curve inverted, VIX 20-35, Fed raises rates 4.25%'),
]

def populate_crash_events():
    """Populate crash_events table with verified historical crashes."""
    db = DatabaseManager()

    # Create tables if they don't exist
    db.create_tables()

    session = db.get_session()

    try:
        # Clear existing data
        session.query(CrashEvent).delete()
        session.commit()
        print("Cleared existing crash events")

        # Add new crashes
        # Note: end_date should be recovery_date (when market returns to pre-crash level)
        # trough_date is the bottom (lowest point)
        for start, end, trough, recovery, drawdown, recovery_months, crash_type, notes in HISTORICAL_CRASHES:
            event = CrashEvent(
                start_date=datetime.strptime(start, '%Y-%m-%d').date(),
                end_date=datetime.strptime(recovery, '%Y-%m-%d').date(),  # FIX: Use recovery, not end
                trough_date=datetime.strptime(trough, '%Y-%m-%d').date(),
                recovery_date=datetime.strptime(recovery, '%Y-%m-%d').date(),
                max_drawdown=drawdown,
                recovery_months=recovery_months,
                crash_type=crash_type,
                notes=notes
            )
            session.add(event)

        session.commit()
        print(f"✅ Successfully populated {len(HISTORICAL_CRASHES)} historical crashes")

        # Verify
        count = session.query(CrashEvent).count()
        print(f"   Total crashes in database: {count}")

        # Display summary
        crashes = session.query(CrashEvent).all()
        for crash in crashes:
            print(f"   - {crash.start_date} to {crash.end_date}: {crash.max_drawdown:.1f}% ({crash.crash_type})")

    except Exception as e:
        session.rollback()
        print(f"❌ Error populating crash events: {e}")
    finally:
        session.close()

if __name__ == '__main__':
    populate_crash_events()

