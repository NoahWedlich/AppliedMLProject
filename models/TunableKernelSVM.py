from AppliedML.courselib.models import svm

from models.TunableModel import TunableModel


class TunableKernelSVM(TunableModel):
    """
    A specialized TunableModel for BinaryKernelSVM that allows tuning of hyperparameters
    """

    def __init__(self, hyperparameters, process_count=None, random_seed=None):
        """
        Initializes the TunableKernelSVM with hyperparameters.
        """
        kernel = hyperparameters.get("kernel", None)
        if kernel is None:
            raise ValueError("Kernel must be specified in hyperparameters.")

        # Ensure the hyperparameters contain necessary parameters for the specified kernel
        if "rbf" in kernel and "sigma" not in hyperparameters:
            raise ValueError("RBF kernel requires 'sigma' in hyperparameters.")
        if "polynomial" in kernel and (
            "degree" not in hyperparameters or "intercept" not in hyperparameters
        ):
            raise ValueError(
                "Polynomial kernel requires 'degree' and 'intercept' in hyperparameters."
            )

        super().__init__(
            model_class=svm.BinaryKernelSVM,
            hyperparameters=hyperparameters,
            validator=self.validator,
            process_count=process_count,
            random_seed=random_seed,
        )

    def validator(self, params):
        """
        Selects valid combinations of hyperparameters for the BinaryKernelSVM.

        Parameters:
        - params: Dictionary of hyperparameters.

        Returns:
        - True if the combination is valid, False otherwise.
        """
        if params["kernel"] == "rbf":
            return (
                params["sigma"] > 0
                and params["degree"] == 0
                and params["intercept"] == 0
            )
        elif params["kernel"] == "polynomial":
            return (
                params["sigma"] == 0
                and params["degree"] > 0
                and params["intercept"] >= 0
            )
        else:
            return False
