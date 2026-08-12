from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class RequestRecord(Base):
    __tablename__ = "requests"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Integer)
    
    faces = relationship("FaceCropRecord", back_populates="request")

class FaceCropRecord(Base):
    __tablename__ = "face_crops"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String, ForeignKey("requests.id"))
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    confidence = Column(Float)
    file_path = Column(String)
    
    request = relationship("RequestRecord", back_populates="faces")
