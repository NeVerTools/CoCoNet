class Scheduler(abc.ABC):
    """
    Our internal interface for the pytorch learning rate schedulers. Given the absence of a public common interface
    in pytorch we use this so that our schedulers present a standard interface.

    Methods
    -----------
    step(*args)
        Procede with the next step of the scheduling. We do not support the epoch as an argument since it is deprecated
        in pytorch. The *args parameter is needed to support Schedulers like ReduceLROnPlateau which needs parameters
        to compute the next step.

    state_dict()
        Returns the state of the Scheduler as a dictionary

    load_state_dict(dict)
        Loads the Scheduler state.

    """

    @abc.abstractmethod
    def step(self, *args):
        """
        Procede with the next step of the scheduling. We do not support the epoch as an argument since it is deprecated
        in pytorch. The *args parameter is needed to support Schedulers like ReduceLROnPlateau which needs parameters
        to compute the next step.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> Dict:
        """
        Returns the state of the Scheduler as a dictionary.

        Returns
        -------
        Dict
            The state of the Scheduler as a dictionary

        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict):
        """
        Loads the Scheduler state.

        Parameters
        ----------
        state_dict : Dict
            The scheduler state as a dictionary
        """
        raise NotImplementedError


class ReduceLROnPlateau(Scheduler, opt.lr_scheduler.ReduceLROnPlateau):
    """
    Wrapper for the pytorch scheduler ReduceLROnPlateau

    Attributes
    ----------
    optimizer : opt.Optimizer
        Wrapped optimizer.
    mode (str): One of `min`, `max`. In `min` mode, lr will
        be reduced when the quantity monitored has stopped
        decreasing; in `max` mode it will be reduced when the
        quantity monitored has stopped increasing. Default: 'min'.
    factor (float): Factor by which the learning rate will be
        reduced. new_lr = lr * factor. Default: 0.1.
    patience (int): Number of epochs with no improvement after
        which learning rate will be reduced. For example, if
        `patience = 2`, then we will ignore the first 2 epochs
        with no improvement, and will only decrease the LR after the
        3rd epoch if the loss still hasn't improved then.
        Default: 10.
    threshold (float): Threshold for measuring the new optimum,
        to only focus on significant changes. Default: 1e-4.
    threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
        dynamic_threshold = best * ( 1 + threshold ) in 'max'
        mode or best * ( 1 - threshold ) in `min` mode.
        In `abs` mode, dynamic_threshold = best + threshold in
        `max` mode or best - threshold in `min` mode. Default: 'rel'.
    cooldown (int): Number of epochs to wait before resuming
        normal operation after lr has been reduced. Default: 0.
    min_lr (float or list): A scalar or a list of scalars. A
        lower bound on the learning rate of all param groups
        or each group respectively. Default: 0.
    eps (float): Minimal decay applied to lr. If the difference
        between new and old lr is smaller than eps, the update is
        ignored. Default: 1e-8.
    verbose (bool): If ``True``, prints a message to stdout for
        each update. Default: ``False``.

    """

    def __init__(self, optimizer: opt.Optimizer, mode: str = 'min', factor: float = 0.1, patience: int = 10,
                 threshold: float = 1e-4, threshold_mode: str = 'rel', cooldown: int = 0, min_lr: float = 0,
                 eps: float = 1e-8, verbose: bool = False):

        Scheduler.__init__(self)
        opt.lr_scheduler.ReduceLROnPlateau.__init__(self, optimizer, mode, factor, patience, verbose,
                                                    threshold, threshold_mode, cooldown, min_lr, eps)