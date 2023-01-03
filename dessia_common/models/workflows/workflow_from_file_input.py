"""
Test Workflow to check good behavior with file inputs.
"""

from dessia_common.workflow.core import Workflow
from dessia_common.workflow.blocks import ClassMethod
from dessia_common.utils.types import ClassMethodType
from dessia_common.forms import StandaloneObject

method_type = ClassMethodType(class_=StandaloneObject, name="generate_from_text")
class_method = ClassMethod(method_type=method_type, name="Class Method")

blocks = [class_method]
pipes = []

workflow_ = Workflow(blocks=blocks, pipes=pipes, output=class_method.outputs[0])
