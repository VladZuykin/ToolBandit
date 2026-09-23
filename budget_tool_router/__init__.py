"""Budget-aware contextual bandit routing for tools."""

from .router import (
    BudgetAwareToolRouter,
    Decision,
    NoFeasibleToolError,
    ToolSpec,
)
from .judge import DeepSeekJudge, JudgeResult
from .retriever import RetrievalHit, SemanticToolRetriever, ToolDocument, toolbench_documents
from .service import ToolRoutingService, create_app
from .registry import ToolRegistry
from .ada_embeddings import Ada002Encoder
from .adapters import (
    LatencyTracker,
    OpenMeteoWeatherAdapter,
    PublicApiExample,
    PublicJsonApiAdapter,
    RuntimeEstimate,
    ToolAdapter,
    ToolExecution,
    public_api_examples,
)

__all__ = [
    "BudgetAwareToolRouter",
    "Decision",
    "NoFeasibleToolError",
    "ToolSpec",
    "DeepSeekJudge",
    "JudgeResult",
    "RetrievalHit",
    "SemanticToolRetriever",
    "ToolDocument",
    "toolbench_documents",
    "ToolRoutingService",
    "create_app",
    "ToolRegistry",
    "Ada002Encoder",
    "LatencyTracker",
    "OpenMeteoWeatherAdapter",
    "PublicApiExample",
    "PublicJsonApiAdapter",
    "RuntimeEstimate",
    "ToolAdapter",
    "ToolExecution",
    "public_api_examples",
]
