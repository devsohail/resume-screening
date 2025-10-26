"""
Pytest configuration and fixtures
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.storage.db_models import Base


@pytest.fixture(scope="session")
def test_database():
    """Create test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    
    yield SessionLocal
    
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(test_database):
    """Get database session"""
    session = test_database()
    try:
        yield session
    finally:
        session.close()

