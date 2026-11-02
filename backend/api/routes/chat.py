"""
Chat and conversation endpoints.

    POST   /api/chat                     ask a question (text and/or image)
    GET    /api/conversations            sidebar list
    POST   /api/conversations            create an empty conversation
    GET    /api/conversations/{id}       full transcript
    PATCH  /api/conversations/{id}       rename
    DELETE /api/conversations/{id}       delete
    DELETE /api/conversations            delete all
    GET    /api/conversations/{id}/image the pinned attachment (image/*)
    DELETE /api/conversations/{id}/image detach the attachment
"""
from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, HTTPException, Query, Response, status

from backend.api.deps import SessionDep
from backend.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationList,
    ConversationSummary,
    RenameRequest,
)
from backend.db.repositories.conversations import ConversationRepository
from backend.services.chat_service import ChatService

log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(payload: ChatRequest, session: SessionDep) -> ChatResponse:
    """Run one turn through the reasoning pipeline and persist it."""
    try:
        image = payload.image_bytes()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not payload.message.strip() and image is None:
        raise HTTPException(
            status_code=422, detail="Send a message, an image, or both."
        )

    service = ChatService(session)
    try:
        result = service.handle_turn(
            conversation_id=payload.conversation_id,
            message=payload.message,
            image=image,
            developer_mode=payload.developer_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ChatResponse(**result)


@router.get("/conversations", response_model=ConversationList)
def list_conversations(
    session: SessionDep,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ConversationList:
    repo = ConversationRepository(session)
    items = repo.list(limit=limit, offset=offset)
    return ConversationList(
        total=repo.count(),
        items=[ConversationSummary(**repo.summary(c)) for c in items],
    )


@router.post(
    "/conversations",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(session: SessionDep) -> ConversationSummary:
    repo = ConversationRepository(session)
    conversation = repo.create()
    session.commit()
    return ConversationSummary(**repo.summary(conversation))


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, session: SessionDep) -> ConversationDetail:
    repo = ConversationRepository(session)
    conversation = repo.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    data = repo.to_dict(conversation)
    return ConversationDetail(**data)


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(
    conversation_id: str, payload: RenameRequest, session: SessionDep
) -> ConversationSummary:
    repo = ConversationRepository(session)
    conversation = repo.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    repo.rename(conversation, payload.title)
    session.commit()
    return ConversationSummary(**repo.summary(conversation))


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, session: SessionDep) -> Response:
    service = ChatService(session)
    conversation = service.conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    service.delete(conversation)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/conversations", status_code=status.HTTP_200_OK)
def clear_conversations(session: SessionDep) -> dict:
    removed = ChatService(session).clear()
    return {"removed": removed}


@router.get("/conversations/{conversation_id}/image")
def get_conversation_image(conversation_id: str, session: SessionDep) -> Response:
    repo = ConversationRepository(session)
    conversation = repo.get(conversation_id)
    if conversation is None or not conversation.image_bytes:
        raise HTTPException(status_code=404, detail="No image attached")
    return Response(
        content=conversation.image_bytes,
        media_type=conversation.image_mime or "application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@router.delete(
    "/conversations/{conversation_id}/image", status_code=status.HTTP_200_OK
)
def detach_image(conversation_id: str, session: SessionDep) -> dict:
    from backend.services.engine import engine

    repo = ConversationRepository(session)
    conversation = repo.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    repo.detach_image(conversation)
    session.commit()
    engine.reset_pipeline(conversation_id)
    return {"detached": True}


@router.get("/conversations/{conversation_id}/image/base64")
def get_conversation_image_base64(
    conversation_id: str, session: SessionDep
) -> dict:
    """Convenience for clients that cannot fetch bytes (e.g. a Streamlit rerun)."""
    repo = ConversationRepository(session)
    conversation = repo.get(conversation_id)
    if conversation is None or not conversation.image_bytes:
        raise HTTPException(status_code=404, detail="No image attached")
    return {
        "name": conversation.image_name,
        "mime": conversation.image_mime,
        "image_b64": base64.b64encode(conversation.image_bytes).decode("ascii"),
    }


@router.get("/conversations/{conversation_id}/export")
def export_conversation(
    conversation_id: str, session: SessionDep, include_dev: bool = True
) -> dict:
    """Plain JSON export — handy for reports and for grading a demo."""
    repo = ConversationRepository(session)
    conversation = repo.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    data = repo.to_dict(conversation)
    if not include_dev:
        for message in data["messages"]:
            message.pop("dev", None)
    return data
