# Migration script to add the new column
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '<timestamp>'
down_revision = '<previous_revision>'
branch_labels = None
depends_on = None

def upgrade():
op.add_column('developers', sa.Column('email', sa.String(), nullable=True))

def downgrade():
op.drop_column('developers', 'email')