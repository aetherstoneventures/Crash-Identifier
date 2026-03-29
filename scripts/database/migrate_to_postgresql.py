#!/usr/bin/env python3
"""Migrate database from SQLite to PostgreSQL.

This script:
1. Creates a backup of the SQLite database
2. Creates PostgreSQL tables
3. Migrates all data from SQLite to PostgreSQL
4. Validates the migration
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
import pandas as pd

from src.utils.config import DATABASE_URL, POSTGRESQL_URL
from src.utils.database import Base, Indicator, Prediction, CrashEvent, BottomPrediction
from src.utils.backup import DatabaseBackup
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_postgresql_tables(pg_engine):
    """Create all tables in PostgreSQL database.
    
    Args:
        pg_engine: PostgreSQL SQLAlchemy engine
    """
    logger.info("Creating PostgreSQL tables...")
    
    # Drop all tables (if they exist) and recreate
    Base.metadata.drop_all(pg_engine)
    Base.metadata.create_all(pg_engine)
    
    logger.info("PostgreSQL tables created successfully")


def migrate_table(
    sqlite_session,
    pg_session,
    model_class,
    batch_size: int = 1000
):
    """Migrate a single table from SQLite to PostgreSQL.
    
    Args:
        sqlite_session: SQLite session
        pg_session: PostgreSQL session
        model_class: SQLAlchemy model class
        batch_size: Number of records to migrate per batch
    """
    table_name = model_class.__tablename__
    logger.info(f"Migrating table: {table_name}")
    
    # Count total records
    total_count = sqlite_session.query(model_class).count()
    logger.info(f"  Total records: {total_count}")
    
    if total_count == 0:
        logger.info(f"  No records to migrate for {table_name}")
        return
    
    # Migrate in batches
    migrated = 0
    offset = 0
    
    while offset < total_count:
        # Fetch batch from SQLite
        batch = sqlite_session.query(model_class).limit(batch_size).offset(offset).all()
        
        if not batch:
            break
        
        # Detach from SQLite session
        for obj in batch:
            sqlite_session.expunge(obj)
            # Reset primary key to let PostgreSQL auto-generate
            if hasattr(obj, 'id'):
                obj.id = None
        
        # Add to PostgreSQL session
        pg_session.bulk_save_objects(batch)
        pg_session.commit()
        
        migrated += len(batch)
        offset += batch_size
        
        logger.info(f"  Migrated {migrated}/{total_count} records ({migrated/total_count*100:.1f}%)")
    
    # Verify count
    pg_count = pg_session.query(model_class).count()
    logger.info(f"  PostgreSQL count: {pg_count}")
    
    if pg_count != total_count:
        logger.warning(f"  ⚠️  Count mismatch! SQLite: {total_count}, PostgreSQL: {pg_count}")
    else:
        logger.info(f"  ✓ Migration successful for {table_name}")


def validate_migration(sqlite_session, pg_session):
    """Validate that migration was successful.
    
    Args:
        sqlite_session: SQLite session
        pg_session: PostgreSQL session
        
    Returns:
        True if validation passed
    """
    logger.info("Validating migration...")
    
    models = [Indicator, Prediction, CrashEvent, BottomPrediction]
    all_valid = True
    
    for model in models:
        table_name = model.__tablename__
        
        sqlite_count = sqlite_session.query(model).count()
        pg_count = pg_session.query(model).count()
        
        if sqlite_count == pg_count:
            logger.info(f"  ✓ {table_name}: {pg_count} records")
        else:
            logger.error(f"  ✗ {table_name}: SQLite={sqlite_count}, PostgreSQL={pg_count}")
            all_valid = False
    
    return all_valid


def main():
    """Main migration function."""
    logger.info("=" * 80)
    logger.info("DATABASE MIGRATION: SQLite → PostgreSQL")
    logger.info("=" * 80)
    
    # Check if PostgreSQL URL is configured
    if not POSTGRESQL_URL or POSTGRESQL_URL == "postgresql://user:password@localhost:5432/crash_predictor":
        logger.error("PostgreSQL URL not configured!")
        logger.error("Please set POSTGRESQL_URL in .env file")
        logger.error("Example: postgresql://username:password@localhost:5432/crash_predictor")
        return False
    
    # Check if SQLite database exists
    if "sqlite" not in DATABASE_URL.lower():
        logger.error("Current DATABASE_URL is not SQLite!")
        logger.error("This script migrates FROM SQLite TO PostgreSQL")
        return False
    
    sqlite_db_path = DATABASE_URL.replace("sqlite:///", "")
    if not Path(sqlite_db_path).exists():
        logger.error(f"SQLite database not found: {sqlite_db_path}")
        return False
    
    # Step 1: Create backup of SQLite database
    logger.info("\nStep 1: Creating backup of SQLite database...")
    backup_manager = DatabaseBackup()
    backup_path = backup_manager.create_backup(backup_name="pre_postgresql_migration")
    logger.info(f"Backup created: {backup_path}")
    
    # Step 2: Create PostgreSQL connection
    logger.info("\nStep 2: Connecting to PostgreSQL...")
    try:
        pg_engine = create_engine(POSTGRESQL_URL, echo=False)
        pg_engine.connect()
        logger.info("✓ PostgreSQL connection successful")
    except Exception as e:
        logger.error(f"✗ Failed to connect to PostgreSQL: {e}")
        logger.error("Please ensure PostgreSQL is running and credentials are correct")
        return False
    
    # Step 3: Create tables in PostgreSQL
    logger.info("\nStep 3: Creating PostgreSQL tables...")
    try:
        create_postgresql_tables(pg_engine)
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL tables: {e}")
        return False
    
    # Step 4: Create sessions
    logger.info("\nStep 4: Creating database sessions...")
    sqlite_engine = create_engine(DATABASE_URL, echo=False)
    SQLiteSession = sessionmaker(bind=sqlite_engine)
    PostgreSQLSession = sessionmaker(bind=pg_engine)
    
    sqlite_session = SQLiteSession()
    pg_session = PostgreSQLSession()
    
    # Step 5: Migrate data
    logger.info("\nStep 5: Migrating data...")
    
    try:
        # Migrate in order (respecting foreign keys)
        migrate_table(sqlite_session, pg_session, Indicator, batch_size=1000)
        migrate_table(sqlite_session, pg_session, CrashEvent, batch_size=100)
        migrate_table(sqlite_session, pg_session, Prediction, batch_size=1000)
        migrate_table(sqlite_session, pg_session, BottomPrediction, batch_size=1000)
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        pg_session.rollback()
        return False
    
    # Step 6: Validate migration
    logger.info("\nStep 6: Validating migration...")
    validation_passed = validate_migration(sqlite_session, pg_session)
    
    # Close sessions
    sqlite_session.close()
    pg_session.close()
    
    # Step 7: Summary
    logger.info("\n" + "=" * 80)
    if validation_passed:
        logger.info("✓ MIGRATION SUCCESSFUL!")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("1. Update DATABASE_URL in .env to use PostgreSQL:")
        logger.info(f"   DATABASE_URL={POSTGRESQL_URL}")
        logger.info("2. Test the application with PostgreSQL")
        logger.info("3. Once confirmed working, you can archive the SQLite database")
        logger.info(f"4. SQLite backup saved at: {backup_path}")
    else:
        logger.error("✗ MIGRATION FAILED - VALIDATION ERRORS DETECTED")
        logger.error("=" * 80)
        logger.error("Please review the errors above and try again")
        logger.error("Your SQLite database is unchanged")
    
    logger.info("=" * 80)
    
    return validation_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

