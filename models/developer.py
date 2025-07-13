# File changes (if any)
class Developer(Base):
__tablename__ = 'developers'
id = Column(Integer, primary_key=True)
name = Column(String)
email = Column(String)  # Newly added parameter