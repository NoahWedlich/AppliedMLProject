from models.RandomForest import RandomForestClassifier

from models.TunableModel import TunableModel


class TunableRandomForest(TunableModel):
    """
    A TunableModel for RandomForestClassifier that allows tuning of hyperparameters.
    """

    def __init__(self, hyperparameters, validator=None):
        """
        Initializes the TunableRandomForest with hyperparameters.

        Parameters:
        - hyperparameters: Dictionary of hyperparameters to tune.
        - validator: Optional function to select valid combinations of hyperparameters.
        """
        if "n_estimators" not in hyperparameters:
            raise ValueError(
                "Number of estimators must be specified in hyperparameters."
            )

        if "max_depth" not in hyperparameters:
            raise ValueError("Max depth must be specified in hyperparameters.")

        super().__init__(
            model_class=RandomForestClassifier,
            hyperparameters=hyperparameters,
            validator=validator,
        )
