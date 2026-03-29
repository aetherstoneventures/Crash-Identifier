"""Database backup and restore utilities.

Provides automated backup functionality with:
- Scheduled backups
- Compression
- Retention policy
- Restore capability
"""

import logging
import shutil
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import json

from src.utils.config import (
    DATABASE_URL,
    BACKUPS_DIR,
    BACKUP_ENABLED,
    BACKUP_INTERVAL_HOURS,
    BACKUP_RETENTION_DAYS,
    BACKUP_COMPRESS,
    DATA_DIR
)

logger = logging.getLogger(__name__)


class DatabaseBackup:
    """Manages database backups."""
    
    def __init__(
        self,
        backup_dir: Path = BACKUPS_DIR,
        compress: bool = BACKUP_COMPRESS,
        retention_days: int = BACKUP_RETENTION_DAYS
    ):
        """Initialize backup manager.
        
        Args:
            backup_dir: Directory to store backups
            compress: Whether to compress backups
            retention_days: Number of days to retain backups
        """
        self.backup_dir = Path(backup_dir)
        self.compress = compress
        self.retention_days = retention_days
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, backup_name: Optional[str] = None) -> Path:
        """Create a database backup.
        
        Args:
            backup_name: Optional custom backup name
            
        Returns:
            Path to backup file
        """
        if not BACKUP_ENABLED:
            logger.warning("Backups are disabled in configuration")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if backup_name:
            filename = f"{backup_name}_{timestamp}"
        else:
            filename = f"backup_{timestamp}"
        
        # Determine source database file
        if "sqlite" in DATABASE_URL.lower():
            # Extract database path from SQLite URL
            db_path = DATABASE_URL.replace("sqlite:///", "")
            source_path = Path(db_path)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Database file not found: {source_path}")
            
            # Create backup
            if self.compress:
                backup_path = self.backup_dir / f"{filename}.db.gz"
                logger.info(f"Creating compressed backup: {backup_path}")
                
                with open(source_path, 'rb') as f_in:
                    with gzip.open(backup_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                backup_path = self.backup_dir / f"{filename}.db"
                logger.info(f"Creating backup: {backup_path}")
                shutil.copy2(source_path, backup_path)
            
            # Create metadata file
            metadata = {
                'timestamp': timestamp,
                'source': str(source_path),
                'compressed': self.compress,
                'size_bytes': backup_path.stat().st_size
            }
            
            metadata_path = self.backup_dir / f"{filename}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Backup created successfully: {backup_path}")
            logger.info(f"Backup size: {backup_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            return backup_path
        
        else:
            # PostgreSQL backup (requires pg_dump)
            logger.warning("PostgreSQL backup not yet implemented. Use pg_dump manually.")
            return None
    
    def restore_backup(self, backup_path: Path) -> bool:
        """Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Determine target database file
        if "sqlite" in DATABASE_URL.lower():
            db_path = DATABASE_URL.replace("sqlite:///", "")
            target_path = Path(db_path)
            
            # Create backup of current database before restoring
            if target_path.exists():
                current_backup = self.create_backup(backup_name="pre_restore")
                logger.info(f"Created backup of current database: {current_backup}")
            
            # Restore
            if backup_path.suffix == '.gz':
                logger.info(f"Restoring from compressed backup: {backup_path}")
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(target_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                logger.info(f"Restoring from backup: {backup_path}")
                shutil.copy2(backup_path, target_path)
            
            logger.info(f"Database restored successfully from: {backup_path}")
            return True
        
        else:
            logger.warning("PostgreSQL restore not yet implemented. Use pg_restore manually.")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List all available backups.
        
        Returns:
            List of backup metadata dictionaries
        """
        backups = []
        
        for metadata_file in self.backup_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Find corresponding backup file
                backup_file = metadata_file.with_suffix('.db.gz' if metadata.get('compressed') else '.db')
                
                if backup_file.exists():
                    metadata['path'] = str(backup_file)
                    metadata['metadata_path'] = str(metadata_file)
                    backups.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return backups
    
    def cleanup_old_backups(self) -> int:
        """Remove backups older than retention period.
        
        Returns:
            Number of backups deleted
        """
        if self.retention_days <= 0:
            logger.info("Backup retention disabled (retention_days <= 0)")
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        backups = self.list_backups()
        
        for backup in backups:
            backup_date = datetime.strptime(backup['timestamp'], "%Y%m%d_%H%M%S")
            
            if backup_date < cutoff_date:
                try:
                    # Delete backup file
                    backup_path = Path(backup['path'])
                    if backup_path.exists():
                        backup_path.unlink()
                        logger.info(f"Deleted old backup: {backup_path}")
                    
                    # Delete metadata file
                    metadata_path = Path(backup['metadata_path'])
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete backup {backup['path']}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count
    
    def get_latest_backup(self) -> Optional[Path]:
        """Get path to the most recent backup.
        
        Returns:
            Path to latest backup or None
        """
        backups = self.list_backups()
        
        if backups:
            return Path(backups[0]['path'])
        
        return None


def scheduled_backup():
    """Perform scheduled backup (called by scheduler)."""
    logger.info("Running scheduled database backup...")
    
    backup_manager = DatabaseBackup()
    
    try:
        # Create backup
        backup_path = backup_manager.create_backup()
        
        if backup_path:
            # Cleanup old backups
            deleted = backup_manager.cleanup_old_backups()
            logger.info(f"Backup complete. Deleted {deleted} old backups.")
        
    except Exception as e:
        logger.error(f"Scheduled backup failed: {e}")
        raise


if __name__ == "__main__":
    # Test backup functionality
    logging.basicConfig(level=logging.INFO)
    
    backup_manager = DatabaseBackup()
    
    print("Creating backup...")
    backup_path = backup_manager.create_backup()
    
    print("\nListing backups:")
    backups = backup_manager.list_backups()
    for backup in backups:
        print(f"  - {backup['timestamp']}: {backup['path']} ({backup['size_bytes'] / 1024:.2f} KB)")
    
    print("\nCleaning up old backups...")
    deleted = backup_manager.cleanup_old_backups()
    print(f"Deleted {deleted} old backups")

