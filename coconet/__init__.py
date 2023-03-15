import os

# ROOT_DIR computes the absolute path of this folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
# APP_NAME is used in the window title
APP_NAME = 'CoCoNet'


def getClassname(object_instance):
    """
    Utility method to return the class of an object instance

    Parameters
    ----------
    object_instance: Any
        An instance of some sort

    Returns
    ----------
        The class name as a string

    """

    return object_instance.__class__.__name__
