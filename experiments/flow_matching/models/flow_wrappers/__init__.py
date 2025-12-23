"""
Flow Matching Wrappers

This package contains model wrappers that adapt existing architectures
for flow matching training and inference.
"""

from .flow_wrapper import FlowMatchingWrapper, TimeConditionedFlowWrapper

__all__ = [
    'FlowMatchingWrapper',
    'TimeConditionedFlowWrapper',
]
