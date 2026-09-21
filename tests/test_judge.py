"""Contract unit tests for the bench judge: deterministic conversion of relative times
(judge v2) — the LLM only compares, it never does calendar arithmetic in its head.

Why: in real runs, letting the judge model work out weekdays itself made it misjudge
2023-07-18 (the Tuesday before Thursday 7/20). The harness now converts these in Python and
hands the judge a parenthesised annotation instead.
"""

from __future__ import annotations

from scripts.bench.judge import normalize_relative_times as nz


def test_weekday_before_converts_to_single_date():
    assert nz("The Sunday before 25 May 2023") == "The Sunday before 25 May 2023 (= 2023-05-21)"
    assert nz("The Friday before 15 July 2023") == "The Friday before 15 July 2023 (= 2023-07-14)"
    assert nz("The Tuesday before 20 July 2023") == "The Tuesday before 20 July 2023 (= 2023-07-18)"


def test_week_and_weekend_windows():
    assert nz("The week before 9 June 2023") == "The week before 9 June 2023 (= 2023-06-02 to 2023-06-08)"
    assert nz("The weekend before 17 July 2023") == "The weekend before 17 July 2023 (= 2023-07-15 to 2023-07-16)"
    assert nz("two weekends before 17 July 2023") == \
        "two weekends before 17 July 2023 (= 2023-07-08 to 2023-07-09)"
    assert nz("The week of 23 August 2023") == "The week of 23 August 2023 (= 2023-08-21 to 2023-08-27)"


def test_american_month_first_format():
    # The answerer side often emits the "October 20, 2023" form while the gold standard uses
    # day-month-year, so both sides need converting.
    assert nz("the weekend before October 20, 2023") == \
        "the weekend before October 20, 2023 (= 2023-10-14 to 2023-10-15)"


def test_no_match_left_untouched():
    assert nz("around September 2023") == "around September 2023"      # no day, nothing to convert
    assert nz("") == ""
    assert nz("2023-07-18 直接绝对日期") == "2023-07-18 直接绝对日期"


def test_anchor_on_same_weekday_takes_previous_week():
    # When the anchor date falls on the target weekday, take the previous week: "Friday before"
    # a Friday anchor means the Friday before that one.
    assert nz("The Friday before 14 July 2023") == "The Friday before 14 July 2023 (= 2023-07-07)"


def test_no_space_day_month_still_matches():
    # H2: the gold standard contains "7November, 2022" with no space between day and month.
    # The original regex required \s+, so it failed to annotate and left the judge doing the
    # arithmetic in its head.
    assert nz("The Saturday before 7November, 2022") == \
        "The Saturday before 7November, 2022 (= 2022-11-05)"
    assert nz("the weekend before November7 2022") == \
        "the weekend before November7 2022 (= 2022-11-05 to 2022-11-06)"
