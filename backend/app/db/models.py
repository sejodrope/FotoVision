from datetime import datetime
from sqlalchemy import String, Float, DateTime, Integer, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    model_name: Mapped[str] = mapped_column(String(64))
    predicted_class: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    scores_json: Mapped[str] = mapped_column(Text)
    gradcam_image: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(256), nullable=True)
    demo_mode: Mapped[bool] = mapped_column(Boolean, default=True)
