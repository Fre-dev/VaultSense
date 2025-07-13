# File changes (if any)
class Developer(Base):
__tablename__ = 'developers'
id = Column(Integer, primary_key=True)
name = Column(String, nullable=False)
email = Column(String, unique=True, nullable=False)  # New parameter