"""FastAPI service combining semantic retrieval, LinUCB and async judging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import threading
import time
from typing import Any, Optional
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .judge import DeepSeekJudge
from .retriever import SemanticToolRetriever
from .router import BudgetAwareToolRouter
from .registry import ToolRegistry, definition_assets


class ToolRoutingService:
    def __init__(self, retriever: SemanticToolRetriever, router: BudgetAwareToolRouter,
                 judge: Optional[DeepSeekJudge] = None, *, judge_workers: int = 4,
                 registry: Optional[ToolRegistry] = None) -> None:
        self.retriever = retriever
        self.router = router
        self.judge = judge
        self.registry = registry
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._contexts: dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=judge_workers,
                                            thread_name_prefix="tool-judge")

    def refresh_registry(self) -> int:
        if self.registry is None:
            raise RuntimeError("tool registry is not configured")
        definitions = self.registry.list(limit=100_000, enabled=True)
        assets = [definition_assets(item) for item in definitions]
        documents = [item[0] for item in assets]
        specs = [item[1] for item in assets]
        # Embed first. If the provider fails, the currently live index remains intact.
        with self._lock:
            self.retriever.replace_documents(documents)
            for spec in specs:
                self.router.upsert_tool(spec)
        return len(specs)

    def search(self, request: dict[str, Any]) -> dict[str, Any]:
        query = str(request["query"])
        retrieval_limit = int(request.get("retrieval_limit", 20))
        result_limit = int(request.get("result_limit", 5))
        threshold = float(request.get("retrieval_threshold", 0.0))
        budget = float(request["remaining_budget"])
        sla = request.get("latency_sla")
        sla = None if sla is None else float(sla)
        excluded = request.get("excluded_tools", [])
        with self._lock:
            if not self.retriever.documents:
                return {
                    "request_id": str(uuid.uuid4()),
                    "retrieved_count": 0, "feasible_count": 0, "tools": [],
                }
        context, hits = self.retriever.search(
            query, limit=retrieval_limit, threshold=threshold
        )
        hit_by_name = {hit.tool_name: hit for hit in hits}
        with self._lock:
            decisions = self.router.rank(
                context, remaining_budget=budget, latency_sla=sla,
                candidates=list(hit_by_name), excluded=excluded,
            )
        tools = []
        decision_by_name = {decision.tool_name: decision for decision in decisions}
        # Retrieval owns ordering. LinUCB supplies an additional score only;
        # budget/SLA feasibility may remove a retrieved candidate.
        for hit in hits:
            decision = decision_by_name.get(hit.tool_name)
            if decision is None:
                continue
            row = asdict(decision)
            row.pop("cost_lcb")
            row.pop("cost_ucb")
            row.pop("latency_lcb")
            row.pop("latency_ucb")
            row.pop("remaining_budget")
            row.pop("pass_mean")
            row.pop("pass_uncertainty")
            row.pop("pass_ucb")
            ucb_score = row.pop("score")
            row.update({
                "tool": row.pop("tool_name"),
                "ucb_score": ucb_score,
                "retrieval_score": hit.retrieval_score,
                "retrieval_rank": hit.retrieval_rank,
                "input_schema": hit.metadata.get("input_schema", {}),
                "output_schema": hit.metadata.get("output_schema", {}),
                "metadata": hit.metadata.get("metadata", {}),
            })
            tools.append(row)
        tools = tools[:result_limit]
        request_id = str(uuid.uuid4())
        with self._lock:
            self._contexts[request_id] = context.copy()
        return {
            "request_id": request_id,
            "retrieved_count": len(hits),
            "feasible_count": len(decisions),
            "tools": tools,
        }

    def submit_evaluation(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.judge is None:
            raise RuntimeError("DeepSeek judge is not configured")
        evaluation_id = str(request.get("interaction_id") or uuid.uuid4())
        with self._lock:
            existing = self._jobs.get(evaluation_id)
            if existing is not None:
                return dict(existing)
            self.router.observe_resources(
                str(request["tool_name"]),
                observed_cost=float(request["observed_cost"]),
                observed_latency=float(request["observed_latency"]),
            )
        request_id = request.get("request_id")
        with self._lock:
            saved_context = self._contexts.get(str(request_id)) if request_id else None
        if saved_context is not None:
            context = saved_context.copy()
        else:
            context = self.retriever.encode_query(str(request["context_text"]))
        job = {
            "evaluation_id": evaluation_id, "status": "pending",
            "created_at": time.time(), "tool_name": str(request["tool_name"]),
        }
        with self._lock:
            self._jobs[evaluation_id] = job
        self._executor.submit(self._evaluate, evaluation_id, request, context)
        return dict(job)

    def _evaluate(self, evaluation_id: str, request: dict[str, Any], context) -> None:
        try:
            judgment = self.judge.evaluate(
                query=str(request["tool_intent"]),
                expected_contract=str(request["expected_contract"]),
                tool_name=str(request["tool_name"]),
                tool_output=request.get("tool_output"),
            )
            learned = judgment.verdict != "uncertain"
            if learned:
                with self._lock:
                    self.router.observe_reward(
                        str(request["tool_name"]), context,
                        passed=judgment.passed,
                    )
            result = asdict(judgment)
            result["passed"] = judgment.passed
            with self._lock:
                self._jobs[evaluation_id].update(
                    status="completed", completed_at=time.time(), learned=learned,
                    judgment=result,
                )
        except Exception as error:
            with self._lock:
                self._jobs[evaluation_id].update(
                    status="failed", completed_at=time.time(), error=str(error)
                )

    def get_evaluation(self, evaluation_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(evaluation_id)
            return None if job is None else dict(job)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    remaining_budget: float = Field(ge=0)
    latency_sla: Optional[float] = Field(default=None, gt=0)
    retrieval_limit: int = Field(default=20, ge=1)
    result_limit: int = Field(default=5, ge=1)
    retrieval_threshold: float = 0.0
    excluded_tools: list[str] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    interaction_id: Optional[str] = None
    request_id: Optional[str] = None
    context_text: Optional[str] = None
    tool_name: str = Field(min_length=1)
    tool_intent: str = Field(min_length=1)
    expected_contract: str = Field(min_length=1)
    tool_output: Any = None
    observed_cost: float = Field(ge=0)
    observed_latency: float = Field(ge=0)


class ToolDefinitionRequest(BaseModel):
    tool_name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    cost: float = Field(ge=0)
    latency: Optional[float] = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ToolEnabledRequest(BaseModel):
    enabled: bool


class ToolImportRequest(BaseModel):
    tools: list[ToolDefinitionRequest] = Field(min_length=1)
    replace_existing: bool = False


def _model_dict(model: BaseModel) -> dict[str, Any]:
    # Pydantic v2 uses model_dump; v1 uses dict.
    method = getattr(model, "model_dump", None)
    return method() if method is not None else model.dict()


def create_app(service: ToolRoutingService) -> FastAPI:
    """Build the HTTP application around an already configured router."""
    app = FastAPI(
        title="ToolBandit API", version="1.0.0",
        description="Semantic retrieval, budget-aware LinUCB ranking and delayed LLM judging.",
    )
    app.state.tool_routing_service = service

    @app.get("/health", tags=["system"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok", "judge": service.judge is not None,
            "registry": service.registry is not None,
            "active_tools": len(service.router.tool_names),
            "api_version": "1.1.0",
            "ranking_policy": "semantic_order_with_ucb_annotation",
            "covariance": "diagonal" if service.router.diagonal_covariance else "full",
        }

    @app.post("/v1/tools/search", tags=["routing"])
    def search(request: SearchRequest) -> dict[str, Any]:
        try:
            return service.search(_model_dict(request))
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    def require_registry() -> ToolRegistry:
        if service.registry is None:
            raise HTTPException(status_code=503, detail="tool registry is not configured")
        return service.registry

    @app.get("/v1/tools", tags=["registry"])
    def list_tools(limit: int = 100, offset: int = 0,
                   enabled: Optional[bool] = None) -> dict[str, Any]:
        registry = require_registry()
        limit = min(max(limit, 1), 1000)
        offset = max(offset, 0)
        return {"items": registry.list(limit=limit, offset=offset, enabled=enabled),
                "limit": limit, "offset": offset, "total": registry.count()}

    @app.get("/v1/tools/{tool_name}", tags=["registry"])
    def get_tool(tool_name: str) -> dict[str, Any]:
        item = require_registry().get(tool_name)
        if item is None:
            raise HTTPException(status_code=404, detail="tool not found")
        return item

    @app.post("/v1/tools", status_code=201, tags=["registry"])
    def create_tool(request: ToolDefinitionRequest) -> dict[str, Any]:
        registry = require_registry()
        payload = _model_dict(request)
        try:
            item = registry.put(payload, enabled=request.enabled, create_only=True)
            service.refresh_registry()
            return item
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail="tool already exists") from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail="semantic index update failed: {}".format(error)) from error
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.put("/v1/tools/{tool_name}", tags=["registry"])
    def update_tool(tool_name: str, request: ToolDefinitionRequest) -> dict[str, Any]:
        registry = require_registry()
        if request.tool_name != tool_name:
            raise HTTPException(status_code=409, detail="path and body tool_name differ")
        if registry.get(tool_name) is None:
            raise HTTPException(status_code=404, detail="tool not found")
        item = registry.put(_model_dict(request), enabled=request.enabled)
        service.refresh_registry()
        return item

    @app.patch("/v1/tools/{tool_name}/enabled", tags=["registry"])
    def set_tool_enabled(tool_name: str, request: ToolEnabledRequest) -> dict[str, Any]:
        registry = require_registry()
        item = registry.set_enabled(tool_name, request.enabled)
        if item is None:
            raise HTTPException(status_code=404, detail="tool not found")
        service.refresh_registry()
        return item

    @app.delete("/v1/tools/{tool_name}", status_code=204, tags=["registry"])
    def delete_tool(tool_name: str) -> None:
        registry = require_registry()
        if not registry.delete(tool_name):
            raise HTTPException(status_code=404, detail="tool not found")
        with service._lock:
            service.router.remove_tool(tool_name)
        service.refresh_registry()

    @app.post("/v1/tools/import", tags=["registry"])
    def import_tools(request: ToolImportRequest) -> dict[str, Any]:
        registry = require_registry()
        definitions = [_model_dict(item) for item in request.tools]
        imported = registry.import_many(definitions, replace=request.replace_existing)
        try:
            active = service.refresh_registry()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail={
                "message": "tools were stored but semantic index update failed; fix provider and restart",
                "stored": imported, "cause": str(error),
            }) from error
        return {"imported": imported, "active_tools": active, "total": registry.count()}

    @app.get("/v1/registry/stats", tags=["registry"])
    def registry_stats() -> dict[str, Any]:
        registry = require_registry()
        return {"total": registry.count(), "enabled": registry.count(enabled_only=True),
                "active_in_router": len(service.router.tool_names)}

    @app.post("/v1/evaluations", status_code=202, tags=["feedback"])
    def submit_evaluation(request: EvaluationRequest) -> dict[str, Any]:
        payload = _model_dict(request)
        if not payload.get("request_id") and not payload.get("context_text"):
            raise HTTPException(
                status_code=422,
                detail="request_id or context_text must be provided",
            )
        try:
            return service.submit_evaluation(payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/v1/evaluations/{evaluation_id}", tags=["feedback"])
    def get_evaluation(evaluation_id: str) -> dict[str, Any]:
        job = service.get_evaluation(evaluation_id)
        if job is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return job

    return app


def serve(service: ToolRoutingService, host: str = "127.0.0.1", port: int = 8080) -> None:
    import uvicorn
    uvicorn.run(create_app(service), host=host, port=port)
