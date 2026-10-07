from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException
from app.models.prediction import ModelPrediction, AnalysisSegment
from app.schemas.comparison import ComparisonRequest

async def compare_models(request: ComparisonRequest, db: AsyncSession):
    # This is a mock/basic implementation for the POC
    result = await db.execute(select(ModelPrediction).filter(
        ModelPrediction.analysis_id == request.analysis_id,
        ModelPrediction.model_id.in_(request.model_ids)
    ))
    predictions = result.scalars().all()
    
    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions found for these models in this analysis")

    sentiment_comp = {}
    emotion_comp = {}
    conf_comp = {}
    time_comp = {}
    
    for p in predictions:
        sentiment_comp[p.model_id] = p.sentiment
        emotion_comp[p.model_id] = p.dominant_emotion
        conf_comp[p.model_id] = {
            "sentiment": p.sentiment_confidence,
            "emotion": p.emotion_confidence
        }
        time_comp[p.model_id] = p.processing_time_ms

    # Simple agreement calculation
    sentiments = [p.sentiment for p in predictions if p.sentiment]
    emotions = [p.dominant_emotion for p in predictions if p.dominant_emotion]
    
    sent_agreement = 0.0
    if sentiments and len(sentiments) > 1:
        most_common = max(set(sentiments), key=sentiments.count)
        sent_agreement = (sentiments.count(most_common) / len(sentiments)) * 100
        
    emo_agreement = 0.0
    if emotions and len(emotions) > 1:
        most_common = max(set(emotions), key=emotions.count)
        emo_agreement = (emotions.count(most_common) / len(emotions)) * 100

    return {
        "analysis_id": request.analysis_id,
        "models": request.model_ids,
        "sentiment_comparison": sentiment_comp,
        "emotion_comparison": emotion_comp,
        "confidence_comparison": conf_comp,
        "processing_time_comparison": time_comp,
        "agreement": {
            "sentiment": sent_agreement,
            "emotion": emo_agreement
        },
        "segment_differences": []
    }
