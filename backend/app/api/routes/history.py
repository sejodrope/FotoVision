import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.db.models import Diagnosis
from app.config import CLASS_LABELS
from app.schemas.diagnosis import DiagnosisListItem, DiagnosisResult

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/", response_model=list[DiagnosisListItem])
async def get_history(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Diagnosis).order_by(desc(Diagnosis.created_at)).offset(offset).limit(limit)
    )
    rows = result.scalars().all()
    return [
        DiagnosisListItem(
            id=r.id,
            created_at=r.created_at,
            model_name=r.model_name,
            predicted_class=r.predicted_class,
            predicted_label=CLASS_LABELS.get(r.predicted_class, r.predicted_class),
            confidence=r.confidence,
            demo_mode=r.demo_mode,
            original_filename=r.original_filename,
        )
        for r in rows
    ]


@router.get("/{diagnosis_id}", response_model=DiagnosisResult)
async def get_diagnosis(diagnosis_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Diagnosis).where(Diagnosis.id == diagnosis_id))
    row = result.scalar_one_or_none()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Diagnóstico não encontrado")

    from app.ml.inference import is_model_calibrated
    return DiagnosisResult(
        id=row.id,
        created_at=row.created_at,
        model_name=row.model_name,
        predicted_class=row.predicted_class,
        predicted_label=CLASS_LABELS.get(row.predicted_class, row.predicted_class),
        confidence=row.confidence,
        scores=json.loads(row.scores_json),
        gradcam_image=row.gradcam_image,
        demo_mode=row.demo_mode,
        calibrated=is_model_calibrated(row.model_name),
        original_filename=row.original_filename,
    )


@router.delete("/{diagnosis_id}")
async def delete_diagnosis(diagnosis_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Diagnosis).where(Diagnosis.id == diagnosis_id))
    row = result.scalar_one_or_none()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Diagnóstico não encontrado")
    await db.delete(row)
    await db.commit()
    return {"deleted": diagnosis_id}
