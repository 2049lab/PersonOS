"""Model-free browsing routes, attached to the authenticated API router."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query

from personos.session_scope import scoped_session
from personos.storage.memory_query import MemoryQuery


def install_query_routes(router, ctx_dependency) -> None:
    """Reuse the host router's envelope and request-signing dependencies."""

    @router.get("/memories")
    def list_memories(
        ctx=Depends(ctx_dependency), session_id: str | None = None, caller: str = "",
        object_type: str | None = None, holder: str | None = None,
        kind: str | None = None, domain: str | None = None, query: str | None = None,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0, le=100_000),
    ):
        try:
            if caller and session_id is None:
                raise ValueError("caller requires session_id")
            sid = scoped_session(caller, session_id) if session_id is not None else None
            return MemoryQuery(ctx.atoms.db, ctx.user_id).memories(
                session_id=sid, object_type=object_type, holder=holder, kind=kind,
                domain=domain, query=query, limit=limit, offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/chains")
    def list_chains(
        ctx=Depends(ctx_dependency), limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0, le=100_000),
    ):
        return MemoryQuery(ctx.atoms.db, ctx.user_id).chains(limit=limit, offset=offset)

    @router.get("/chains/{chain_id}")
    def get_chain(
        chain_id: str, ctx=Depends(ctx_dependency),
        limit: int = Query(default=20, ge=1, le=100), cursor: str | None = None,
    ):
        try:
            result = MemoryQuery(ctx.atoms.db, ctx.user_id).chain(
                chain_id, limit=limit, cursor=cursor,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="chain not found")
        return result
