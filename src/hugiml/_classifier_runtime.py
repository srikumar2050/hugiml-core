# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime access to mutable classifier module dependencies."""

from __future__ import annotations

import sys
from typing import Any


class _RuntimeSymbol:
    """Resolve a symbol from the public classifier module when it is used."""

    def __init__(self, name: str, fallback: Any = None) -> None:
        self._name = name
        self._fallback = fallback

    def _value(self) -> Any:
        module = sys.modules.get("hugiml.classifier")
        if module is None:
            return self._fallback
        return getattr(module, self._name, self._fallback)

    def __bool__(self) -> bool:
        return bool(self._value())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._value()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        value = self._value()
        if value is None:
            raise AttributeError(name)
        return getattr(value, name)

    def __str__(self) -> str:
        return str(self._value())

    def __repr__(self) -> str:
        return repr(self._value())

    def __format__(self, format_spec: str) -> str:
        return format(self._value(), format_spec)


_core = _RuntimeSymbol("_core")
_CORE_AVAILABLE = _RuntimeSymbol("_CORE_AVAILABLE", False)
_CORE_IMPORT_ERROR = _RuntimeSymbol("_CORE_IMPORT_ERROR")
DriftDetector = _RuntimeSymbol("DriftDetector")
PredictionMonitor = _RuntimeSymbol("PredictionMonitor")
