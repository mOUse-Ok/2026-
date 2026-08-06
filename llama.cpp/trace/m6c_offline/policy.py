"""Pure M6C reserved-service credit/debt policy.

The only numeric configurations accepted here are synthetic test fixtures.  A
future Evidence replay must use a separately approved immutable preregistration.
"""

from __future__ import annotations

from dataclasses import dataclass


UINT64_MAX = (1 << 64) - 1


class PolicyConfigurationError(ValueError):
    """A policy fixture violates the frozen M6C semantics."""


@dataclass(frozen=True)
class FixturePolicyConfig:
    reserved_numerator: int
    reserved_denominator: int
    minimum_eligibility_age_ns: int
    hard_urgent_guard_ns: int
    test_fixture_only: bool = True
    experimental_parameter: bool = False

    def __post_init__(self) -> None:
        r = self.reserved_numerator
        d = self.reserved_denominator
        if self.test_fixture_only is not True or self.experimental_parameter is not False:
            raise PolicyConfigurationError(
                "M6C-B accepts only test_fixture_only=true, experimental_parameter=false"
            )
        if not isinstance(r, int) or not isinstance(d, int):
            raise PolicyConfigurationError("R and D must be integers")
        if not (0 < r < d and 2 * r <= d):
            raise PolicyConfigurationError("fixture must satisfy 0 < R < D and 2R <= D")
        if self.minimum_eligibility_age_ns <= 0:
            raise PolicyConfigurationError("AGE_GATED_ALL requires a positive fixture age")
        if not (0 <= self.hard_urgent_guard_ns <= UINT64_MAX):
            raise PolicyConfigurationError("hard-urgent guard is outside uint64")

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "reserved_numerator": self.reserved_numerator,
            "reserved_denominator": self.reserved_denominator,
            "minimum_eligibility_age_ns": self.minimum_eligibility_age_ns,
            "hard_urgent_guard_ns": self.hard_urgent_guard_ns,
            "test_fixture_only": self.test_fixture_only,
            "experimental_parameter": self.experimental_parameter,
        }


@dataclass(frozen=True)
class PolicyState:
    credit: int = 0
    pending_debt: bool = False


@dataclass(frozen=True)
class PolicyTransition:
    selected_source: str
    override_reason: str
    credit_before: int
    credit_accrued: int
    credit_after: int
    debt_before: bool
    debt_after: bool
    reserved_due: bool
    debt_created: bool
    debt_repaid: bool
    credit_reset: bool

    @property
    def state_after(self) -> PolicyState:
        return PolicyState(self.credit_after, self.debt_after)


def saturating_add_u64(left: int, right: int) -> int:
    if left < 0 or right < 0 or left > UINT64_MAX or right > UINT64_MAX:
        raise PolicyConfigurationError("saturating addition inputs must be uint64")
    return min(UINT64_MAX, left + right)


def decide_reserved_service(
    state: PolicyState,
    config: FixturePolicyConfig,
    *,
    waiting_eligible_present: bool,
    hard_urgent_present: bool,
    queue_size: int,
    stopping: bool = False,
) -> PolicyTransition:
    """Return one deterministic policy transition without reading a clock or store."""

    r = config.reserved_numerator
    d = config.reserved_denominator
    if queue_size <= 0:
        raise PolicyConfigurationError("a selection decision requires a nonempty queue")
    if not (0 <= state.credit < d):
        raise PolicyConfigurationError("persistent credit must satisfy 0 <= credit < D")
    if state.pending_debt and state.credit >= r:
        raise PolicyConfigurationError("pending debt requires the frozen remainder to be < R")

    before = state.credit
    debt_before = state.pending_debt
    accrued = before
    after = before
    debt_after = debt_before
    due = debt_before
    debt_created = False
    debt_repaid = False
    credit_reset = False

    if stopping:
        return PolicyTransition(
            "legacy", "SHUTDOWN_DRAIN", before, accrued, after,
            debt_before, debt_after, due, False, False, False,
        )

    if not waiting_eligible_present:
        return PolicyTransition(
            "hard_urgent" if hard_urgent_present else "legacy",
            "NO_WAITING_ELIGIBLE",
            before,
            0,
            0,
            debt_before,
            False,
            False,
            False,
            False,
            before != 0 or debt_before,
        )

    if debt_before:
        due = True
    else:
        accrued = before + r
        due = accrued >= d

    if hard_urgent_present:
        source = "hard_urgent"
        if debt_before:
            reason = "HARD_URGENT_OVERRIDE"
        elif due:
            reason = "HARD_URGENT_OVERRIDE"
            after = accrued - d
            debt_after = True
            debt_created = True
        else:
            reason = "RESERVED_CREDIT_NOT_DUE"
            after = accrued
    elif debt_before:
        source = "reserved"
        reason = "CREDIT_DEBT_REPAYMENT"
        debt_after = False
        debt_repaid = True
    elif due:
        source = "reserved"
        reason = "RESERVED_CREDIT_USED"
        after = accrued - d
    else:
        source = "legacy"
        reason = "QUEUE_ONLY_ONE_CLASS" if queue_size == 1 else "RESERVED_CREDIT_NOT_DUE"
        after = accrued

    if not (0 <= after < d):
        raise PolicyConfigurationError("policy transition produced credit outside [0,D)")
    if debt_after and after >= r:
        raise PolicyConfigurationError("policy transition produced an invalid debt remainder")
    return PolicyTransition(
        source,
        reason,
        before,
        accrued,
        after,
        debt_before,
        debt_after,
        due,
        debt_created,
        debt_repaid,
        credit_reset,
    )
