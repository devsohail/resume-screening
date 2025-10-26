"""add_human_feedback_columns

Revision ID: 99149cf9cfbb
Revises: 4f1bc033c9cb
Create Date: 2025-10-18 19:17:42.110714

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '99149cf9cfbb'
down_revision: Union[str, None] = '4f1bc033c9cb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add human feedback columns to screening_results table
    op.add_column('screening_results', sa.Column('human_reviewed', sa.Boolean(), nullable=True))
    op.add_column('screening_results', sa.Column('human_decision', sa.String(length=20), nullable=True))
    op.add_column('screening_results', sa.Column('human_notes', sa.Text(), nullable=True))
    op.add_column('screening_results', sa.Column('human_reviewed_at', sa.DateTime(), nullable=True))
    op.add_column('screening_results', sa.Column('needs_retraining', sa.Boolean(), nullable=True))
    
    # Create indexes
    op.create_index(op.f('ix_screening_results_human_reviewed'), 'screening_results', ['human_reviewed'], unique=False)
    op.create_index(op.f('ix_screening_results_needs_retraining'), 'screening_results', ['needs_retraining'], unique=False)
    
    # Set default values
    op.execute("UPDATE screening_results SET human_reviewed = false WHERE human_reviewed IS NULL")
    op.execute("UPDATE screening_results SET needs_retraining = false WHERE needs_retraining IS NULL")
    
    # Make non-nullable after setting defaults
    op.alter_column('screening_results', 'human_reviewed', nullable=False)
    op.alter_column('screening_results', 'needs_retraining', nullable=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index(op.f('ix_screening_results_needs_retraining'), table_name='screening_results')
    op.drop_index(op.f('ix_screening_results_human_reviewed'), table_name='screening_results')
    
    # Drop columns
    op.drop_column('screening_results', 'needs_retraining')
    op.drop_column('screening_results', 'human_reviewed_at')
    op.drop_column('screening_results', 'human_notes')
    op.drop_column('screening_results', 'human_decision')
    op.drop_column('screening_results', 'human_reviewed')

