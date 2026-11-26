"""Backward-compatible namespace for checkpoints using the former package name.

New code should import :mod:`ordinal_cqr`. This namespace shares the canonical
package search path so serialized references such as ``ocqr_solar.models``
continue to resolve without duplicating the implementation.
"""

from ordinal_cqr import __path__ as __path__
