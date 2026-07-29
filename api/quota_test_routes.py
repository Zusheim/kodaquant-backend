# api/quota_test.py
from fastapi import APIRouter, Depends
from core.quota_manager import verify_quota
 
router = APIRouter(prefix="/api", tags=["quota-test"])
 
 
@router.post("/test-chat")
async def test_chat(user=Depends(verify_quota("chat"))):
    return {"msg": "Acceso permitido"}
 
 
@router.post("/test-prediction")
async def test_prediction(user=Depends(verify_quota("prediction"))):
    return {"msg": "Acceso permitido"}