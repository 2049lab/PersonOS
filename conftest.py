"""根级 conftest:任何位置的 pytest 运行都带上测试护栏。

tests/conftest.py 只覆盖 tests/ 子树;散落在别处的测试文件(他人开发习惯各异)
在这里兜底——PERSONOS_TEST_GUARD=1 下,未进 rollback_scope 的 Database 一律只读
(见 personos/storage/db.py 的 _guard)。宁可让越界测试报错,不让共享 SIT 库被写脏。
"""

import os

os.environ.setdefault("PERSONOS_TEST_GUARD", "1")
