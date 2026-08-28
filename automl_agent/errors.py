class AgentError(Exception):
    """Base error for expected agent failures."""


class ContractError(AgentError):
    """The benchmark contract or assets are invalid."""


class BudgetError(AgentError):
    """The requested action cannot fit inside the remaining budget."""


class ExecutionFailure(AgentError):
    """A child command failed or produced an invalid result."""

    def __init__(self, message: str, *, failure_class: str = "unknown") -> None:
        super().__init__(message)
        self.failure_class = failure_class

