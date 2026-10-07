import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException

from app.schemas.analysis import AnalysisCreate, AnalysisResponse
from app.models.analysis import Analysis
from app.models.audio import AudioFile
from app.models.prediction import ModelPrediction, PredictionScore, AnalysisSegment, Transcript
from app.ml.registry import registry
from app.services import hybrid_service

async def run_analysis(request: AnalysisCreate, db: AsyncSession) -> dict:
    # Validate audio
    audio = await db.get(AudioFile, request.audio_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Validate models
    for model_id in request.model_ids:
        if not registry.get_model(model_id):
            raise HTTPException(status_code=400, detail=f"Model {model_id} not found")

    # Create analysis record
    analysis = Analysis(audio_id=request.audio_id)
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    # Run models concurrently
    tasks = []
    for model_id in request.model_ids:
        model = registry.get_model(model_id)
        tasks.append(_run_model_and_save(analysis.id, model_id, model, audio.storage_path, db))

    await asyncio.gather(*tasks)

    if request.hybrid_mode:
        try:
            transcript = await hybrid_service.transcribe(audio.storage_path)
            text_result = await hybrid_service.analyze_text(transcript["text"])
        except Exception:
            analysis.status = "FAILED"
            analysis.completed_at = datetime.now(timezone.utc)
            await db.commit()
            raise
        transcript_row = Transcript(
            analysis_id=analysis.id,
            language=transcript.get("language_code"),
            full_text=transcript["text"],
        )
        db.add(transcript_row)
        db.add(ModelPrediction(
            analysis_id=analysis.id, model_id="text-sentiment",
            sentiment=text_result["sentiment"], sentiment_confidence=text_result["confidence"],
            status="SUCCESS",
        ))
        audio_predictions = []
        for model_id in request.model_ids:
            model_result = await db.execute(select(ModelPrediction).where(
                ModelPrediction.analysis_id == analysis.id,
                ModelPrediction.model_id == model_id,
            ))
            model_prediction = model_result.scalar_one_or_none()
            if not model_prediction:
                continue
            score_result = await db.execute(select(PredictionScore).where(
                PredictionScore.prediction_id == model_prediction.id
            ))
            audio_predictions.append({
                "sentiment": model_prediction.sentiment,
                "emotion": model_prediction.dominant_emotion,
                "scores": [
                    {"label": score.label, "score": score.score, "score_type": score.score_type}
                    for score in score_result.scalars().all()
                ],
            })
        audio_scores = hybrid_service.audio_distribution(audio_predictions)
        combined = hybrid_service.combine(audio_scores, text_result)
        combined_prediction = ModelPrediction(
            analysis_id=analysis.id,
            model_id="hybrid-combined-confidence_weighted_late_fusion",
            sentiment=combined["sentiment"],
            sentiment_confidence=combined["confidence"],
            status="SUCCESS",
        )
        db.add(combined_prediction)
        await db.flush()
        text_prediction = (await db.execute(select(ModelPrediction).where(
            ModelPrediction.analysis_id == analysis.id,
            ModelPrediction.model_id == "text-sentiment",
        ))).scalar_one()
        for label, score in text_result["scores"].items():
            db.add(PredictionScore(prediction_id=text_prediction.id, label=label,
                                   score=score, score_type="sentiment"))
        for label, score in combined["scores"].items():
            db.add(PredictionScore(prediction_id=combined_prediction.id, label=label,
                                   score=score, score_type="sentiment"))
        await db.commit()

    analysis.status = "COMPLETED"
    analysis.completed_at = datetime.now(timezone.utc)
    await db.commit()

    return await get_analysis(analysis.id, db)

async def _run_model_and_save(analysis_id, model_id, model, audio_path, db: AsyncSession):
    try:
        result = await model.predict(audio_path)
        
        prediction = ModelPrediction(
            analysis_id=analysis_id,
            model_id=model_id,
            sentiment=result.get("sentiment"),
            dominant_emotion=result.get("dominant_emotion"),
            sentiment_confidence=result.get("sentiment_confidence"),
            emotion_confidence=result.get("emotion_confidence"),
            processing_time_ms=result.get("processing_time_ms"),
            status="SUCCESS"
        )
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)

        for score in result.get("scores", []):
            ps = PredictionScore(
                prediction_id=prediction.id,
                label=score["label"],
                score=score["score"],
                score_type=score["score_type"]
            )
            db.add(ps)

        for segment in result.get("segments", []):
            aseg = AnalysisSegment(
                analysis_id=analysis_id,
                model_id=model_id,
                start_time=segment["start_time"],
                end_time=segment["end_time"],
                sentiment=segment.get("sentiment"),
                emotion=segment.get("emotion"),
                confidence=segment.get("confidence")
            )
            db.add(aseg)

        await db.commit()

    except Exception as e:
        prediction = ModelPrediction(
            analysis_id=analysis_id,
            model_id=model_id,
            status="FAILED",
            error_message=str(e)
        )
        db.add(prediction)
        await db.commit()


async def get_analysis(analysis_id, db: AsyncSession) -> dict:
    analysis = await db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    audio = await db.get(AudioFile, analysis.audio_id)

    # Get predictions
    result = await db.execute(select(ModelPrediction).filter(ModelPrediction.analysis_id == analysis_id))
    predictions = result.scalars().all()

    segment_result = await db.execute(
        select(AnalysisSegment)
        .filter(AnalysisSegment.analysis_id == analysis_id)
        .order_by(AnalysisSegment.start_time)
    )
    timeline = [
        {
            "model_id": segment.model_id,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "sentiment": segment.sentiment,
            "emotion": segment.emotion,
            "confidence": segment.confidence,
            "transcript_text": segment.transcript_text,
        }
        for segment in segment_result.scalars().all()
    ]

    prediction_list = []
    models_used = set()
    for p in predictions:
        models_used.add(p.model_id)
        # get scores
        score_res = await db.execute(select(PredictionScore).filter(PredictionScore.prediction_id == p.id))
        scores = [{"label": s.label, "score": s.score, "score_type": s.score_type} for s in score_res.scalars().all()]
        prediction_list.append({
            "model_id": p.model_id,
            "sentiment": p.sentiment,
            "emotion": p.dominant_emotion,
            "confidence": p.sentiment_confidence or p.emotion_confidence,
            "status": p.status,
            "error_message": p.error_message,
            "scores": scores
        })

    hybrid_result = None
    transcript_prediction = next((p for p in prediction_list if p["model_id"] == "text-sentiment"), None)
    combined_prediction = next((p for p in prediction_list if p["model_id"].startswith("hybrid-combined-")), None)
    if transcript_prediction and combined_prediction:
        audio_predictions = [
            p for p in prediction_list
            if p["model_id"] not in {"text-sentiment", combined_prediction["model_id"]}
        ]
        transcript_scores = {
            score["label"]: score["score"]
            for score in transcript_prediction.get("scores", [])
            if score.get("score_type") == "sentiment"
        }
        hybrid_result = hybrid_service.combine(
            hybrid_service.audio_distribution(audio_predictions),
            {**transcript_prediction, "scores": transcript_scores},
        )
        
    models_info = []
    for mid in models_used:
        m = registry.get_model(mid)
        if m:
            models_info.append({"id": mid, "name": m.metadata()["name"]})

    transcript_result = await db.execute(select(Transcript).filter(Transcript.analysis_id == analysis_id).order_by(Transcript.created_at.desc()))
    transcript_row = transcript_result.scalars().first()
    return {
        "id": analysis.id,
        "status": analysis.status,
        "audio": {
            "id": audio.id,
            "filename": audio.original_filename,
            "duration": audio.duration
        },
        "models": models_info,
        "predictions": prediction_list,
        "timeline": timeline,
        "transcript": ({"text": transcript_row.full_text, "language": transcript_row.language} if transcript_row else {}),
        "hybrid": hybrid_result,
    }


async def list_analyses(db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(Analysis, AudioFile)
        .join(AudioFile, AudioFile.id == Analysis.audio_id)
        .order_by(Analysis.created_at.desc())
    )
    return [
        {
            "id": analysis.id,
            "status": analysis.status,
            "started_at": analysis.started_at,
            "completed_at": analysis.completed_at,
            "audio": {
                "id": audio.id,
                "filename": audio.original_filename,
                "duration": audio.duration,
            },
        }
        for analysis, audio in result.all()
    ]

async def get_model_result(analysis_id, model_id, db: AsyncSession):
    # Retrieve the detailed result for a single model in this analysis
    result = await db.execute(select(ModelPrediction).filter(
        ModelPrediction.analysis_id == analysis_id,
        ModelPrediction.model_id == model_id
    ))
    prediction = result.scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=404, detail="Model result not found for this analysis")

    score_res = await db.execute(select(PredictionScore).filter(PredictionScore.prediction_id == prediction.id))
    scores = [{"label": s.label, "score": s.score, "score_type": s.score_type} for s in score_res.scalars().all()]
    
    seg_res = await db.execute(select(AnalysisSegment).filter(
        AnalysisSegment.analysis_id == analysis_id,
        AnalysisSegment.model_id == model_id
    ))
    segments = [{
        "start_time": s.start_time,
        "end_time": s.end_time,
        "sentiment": s.sentiment,
        "emotion": s.emotion,
        "confidence": s.confidence
    } for s in seg_res.scalars().all()]

    return {
        "model_id": prediction.model_id,
        "sentiment": prediction.sentiment,
        "emotion": prediction.dominant_emotion,
        "confidence": prediction.sentiment_confidence or prediction.emotion_confidence,
        "processing_time_ms": prediction.processing_time_ms,
        "status": prediction.status,
        "error_message": prediction.error_message,
        "scores": scores,
        "segments": segments
    }
