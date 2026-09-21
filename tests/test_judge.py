"""bench judge 契约单测:相对时间确定性换算(judge v2)——LLM 只比对,不做日历心算。

实测背景:让 M3 自己心算星期,会把 2023-07-18(7/20 周四前的周二)判错;
harness 里用 Python 换算成括号标注后交 judge。
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
    # answerer 侧常输出 "October 20, 2023" 式;金标是日-月-年式,两侧都要换算
    assert nz("the weekend before October 20, 2023") == \
        "the weekend before October 20, 2023 (= 2023-10-14 to 2023-10-15)"


def test_no_match_left_untouched():
    assert nz("around September 2023") == "around September 2023"      # 无日,不换算
    assert nz("") == ""
    assert nz("2023-07-18 直接绝对日期") == "2023-07-18 直接绝对日期"


def test_anchor_on_same_weekday_takes_previous_week():
    # 锚点当天恰是目标星期几:取前一周("Friday before" 一个周五锚点 → 上周五)
    assert nz("The Friday before 14 July 2023") == "The Friday before 14 July 2023 (= 2023-07-07)"


def test_no_space_day_month_still_matches():
    # H2:金标 "7November, 2022"(日月间无空格)原正则要求 \s+ → 标注不上,只能靠 judge 心算
    assert nz("The Saturday before 7November, 2022") == \
        "The Saturday before 7November, 2022 (= 2022-11-05)"
    assert nz("the weekend before November7 2022") == \
        "the weekend before November7 2022 (= 2022-11-05 to 2022-11-06)"
