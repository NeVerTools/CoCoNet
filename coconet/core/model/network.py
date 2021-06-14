from coconet.core.controller.pynevertemp.strategies.verification import NeVerProperty
from coconet.core.controller.pynevertemp.tensor import Tensor


class NetworkNode:
    """
    This class describes a network node block available for the user, with inputs,
    parameters and a description.

    Attributes
    ----------
    name : str
        Name of the block.
    class_name : str
        Name of the class of the block.
    input : dict
        Dictionary of the inputs taken by the node, connecting for each
        input its name to a dictionary of parameters such as type, default
        value, description.
    param : dict
        Dictionary of the parameters of the block, connecting for each
        parameter its name to a dictionary of parameters such as type, default
        value, description.
    output : dict
        Dictionary of the outputs of the block, , connecting for each
        output its name to a dictionary of parameters such as type, default
        value, description.
    descr : str
        Description of the operation done by the node.

    """

    def __init__(self, name: str, class_name: str, input: dict, param: dict, output: dict, descr: str):
        self.name = name
        self.class_name = class_name
        self.input = input
        self.param = param
        self.output = output
        self.descr = descr


class NetworkProperty:
    """
    Abstract class representing a generic property.

    Attributes
    ----------
    type : str
        The property type.
    property_string : str
        A generic SMT-LIB property.
    property_class : NeVerProperty
        A structured property in the pyNever format.

    """

    def __init__(self, p_type: str):
        self.type = p_type
        if p_type == "SMT":
            self.property_string = "-"
            self.property_class = None
        elif p_type == "Polyhedral":
            self.property_string = "Ax - b <= 0"
            self.property_class = NeVerProperty(Tensor([]), Tensor([]), [], [])
