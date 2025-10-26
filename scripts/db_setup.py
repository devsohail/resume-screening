#!/usr/bin/env python3
"""
Database setup and management script
Usage:
    python scripts/db_setup.py migrate    # Run migrations
    python scripts/db_setup.py seed       # Seed sample data
    python scripts/db_setup.py reset      # Reset database (WARNING: drops all data!)
    python scripts/db_setup.py status     # Show database status
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.storage.db_handler import get_db_handler
from backend.storage.db_models import Job, Resume, ScreeningResult, ModelMetadata
from sqlalchemy import inspect
from datetime import datetime, timedelta
import subprocess


def run_migrations():
    """Run database migrations"""
    print("🔄 Running database migrations...")
    result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Migrations completed successfully!")
        print(result.stdout)
    else:
        print("❌ Migration failed!")
        print(result.stderr)
        sys.exit(1)


def seed_data():
    """Seed sample data for testing"""
    print("🌱 Seeding sample data...")
    db = get_db_handler()
    
    try:
        # Create sample jobs
        jobs = [
            Job(
                title="Senior Python Developer",
                company="TechCorp Inc",
                description="We are looking for an experienced Python developer with ML expertise to join our AI team.",
                required_skills=["python", "fastapi", "aws", "postgresql"],
                preferred_skills=["machine learning", "docker", "kubernetes"],
                min_experience_years=5,
                location="Remote",
                status="active",
                created_at=datetime.utcnow()
            ),
            Job(
                title="ML Engineer",
                company="AI Startup",
                description="Join our team to build cutting-edge machine learning models for production.",
                required_skills=["python", "tensorflow", "pytorch", "ml"],
                preferred_skills=["aws", "sagemaker", "mlflow"],
                min_experience_years=3,
                location="San Francisco, CA",
                status="active",
                created_at=datetime.utcnow()
            ),
            Job(
                title="Full Stack Developer",
                company="WebDev Co",
                description="Looking for a full-stack developer experienced in React and Node.js",
                required_skills=["javascript", "react", "node.js", "typescript"],
                preferred_skills=["aws", "docker", "mongodb"],
                min_experience_years=3,
                location="New York, NY",
                status="active",
                created_at=datetime.utcnow()
            )
        ]
        
        with db.get_session() as session:
            for job in jobs:
                session.add(job)
            session.commit()
            print(f"✅ Created {len(jobs)} sample jobs")
        
        # Create sample model metadata
        model = ModelMetadata(
            model_type="binary_classifier",
            version="v1.0.0",
            s3_path="s3://resume-screening-models/classifier/v1.0.0",
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            is_active=True,
            training_date=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        with db.get_session() as session:
            session.add(model)
            session.commit()
            print("✅ Created sample model metadata")
        
        print("\n✅ Database seeded successfully!")
        
    except Exception as e:
        print(f"❌ Error seeding data: {e}")
        sys.exit(1)


def show_status():
    """Show database status"""
    print("📊 Database Status\n")
    db = get_db_handler()
    
    try:
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        print(f"Connected to: {db.engine.url}")
        print(f"\nTables ({len(tables)}):")
        for table in tables:
            columns = inspector.get_columns(table)
            indexes = inspector.get_indexes(table)
            print(f"  • {table:<20} - {len(columns)} columns, {len(indexes)} indexes")
        
        # Count records
        print("\nRecord counts:")
        with db.get_session() as session:
            job_count = session.query(Job).count()
            resume_count = session.query(Resume).count()
            result_count = session.query(ScreeningResult).count()
            model_count = session.query(ModelMetadata).count()
            
            print(f"  • Jobs: {job_count}")
            print(f"  • Resumes: {resume_count}")
            print(f"  • Screening Results: {result_count}")
            print(f"  • Model Metadata: {model_count}")
        
        print("\n✅ Database is healthy!")
        
    except Exception as e:
        print(f"❌ Error checking database status: {e}")
        sys.exit(1)


def reset_database():
    """Reset database (WARNING: drops all data!)"""
    print("⚠️  WARNING: This will delete ALL data from the database!")
    confirm = input("Type 'YES' to confirm: ")
    
    if confirm != "YES":
        print("❌ Reset cancelled")
        return
    
    print("🔄 Resetting database...")
    
    # Downgrade to base
    result = subprocess.run(["alembic", "downgrade", "base"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Downgrade failed: {result.stderr}")
        sys.exit(1)
    
    # Upgrade to head
    result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Upgrade failed: {result.stderr}")
        sys.exit(1)
    
    print("✅ Database reset complete!")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "migrate":
        run_migrations()
    elif command == "seed":
        seed_data()
    elif command == "status":
        show_status()
    elif command == "reset":
        reset_database()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

