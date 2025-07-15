import multiprocessing as mp
import threading as th
from dataclasses import dataclass

from AppliedML.courselib.models.base import TrainableModel
from AppliedML.courselib.optimizers import Optimizer

from models.CombinationIterator import CombinationIterator

@dataclass        
class TrainingOrder:
    index: int
    model: any
    params: dict
    X: any
    y: any
    training_params: dict = None
    
@dataclass
class TrainingReport:
    model: str
    progress: int
        
class TrainingDone:
    pass

class ReportingOptimizer(Optimizer):
    
    def __init__(self, optimizer, model, num_epochs, queue):
        self.optimizer = optimizer
        self.model = model
        self.epoch = 0
        self.num_epochs = num_epochs
        self.queue = queue
        
    def update(self, params, grads):
        self.epoch += 1
        if self.epoch % 500 == 0 or self.epoch == self.num_epochs:
            progress = int((self.epoch / self.num_epochs) * 100)
            self.queue.put(TrainingReport(self.model, progress))
            
        self.optimizer.update(params, grads)

class TunableModel:
    """
    Represents a model that can be tuned with various hyperparameters.
    """

    def __init__(
        self, model_class, hyperparameters, validator=None, process_count=None
    ):
        """
        Initializes the TunableModel with a model class and hyperparameters.

        Parameters:
        - model_class: Class of the model to be tuned.
        - hyperparameters: Dictionary of hyperparameters to tune.
        - validator: Optional function to select valid combinations of hyperparameters.
        - process_count: Number of processes to use for parallel fitting. Defaults to half the CPU count.
        """
        self.model_class = model_class
        self.combination_iterator = CombinationIterator(hyperparameters, validator)

        if process_count is None:
            process_count = mp.cpu_count() // 2
        self.process_count = max(1, process_count)

        self.models = []

    def get_model(self, parameters):
        """
        Returns an instance of the model with the given parameters.

        Parameters:
        - parameters: Dictionary of hyperparameters for the model.

        Returns:
        - An instance of the model class initialized with the given parameters.
        """
        return self.model_class(**parameters)
        
    def _training_worker(self, model_queue, result_queue, report_queue):
        while True:
            try:
                training_order = model_queue.get(timeout=0.25)
                if isinstance(training_order, TrainingDone):
                    return
                if not isinstance(training_order, TrainingOrder):
                    continue
                    
                cf = training_order
                
                if callable(cf.params.get('optimizer')):
                    cf.params['optimizer'] = cf.params['optimizer'](cf.params)
                
                injected_progress = cf.params.get('optimizer') is not None
                injected_progress = injected_progress and cf.training_params.get('num_epochs') is not None
                injected_progress = injected_progress and cf.training_params.get('batch_size') is not None
                
                num_batches = int(len(cf.X) / cf.training_params['batch_size'])
                num_epochs = cf.training_params.get('num_epochs') * num_batches
                
                if injected_progress:
                    cf.params['optimizer'] = ReportingOptimizer(
                        cf.params['optimizer'], f"Model {cf.index}", num_epochs, report_queue
                    )
                
                model_instance = cf.model.get_model(cf.params)
                
                metrics = model_instance.fit(cf.X, cf.y, **(cf.training_params or {}))
                
                if injected_progress:
                    model_instance.optimizer = model_instance.optimizer.optimizer
                    cf.params['optimizer'] = model_instance.optimizer
                
                result_queue.put((cf.index, model_instance, cf.params, metrics))
            except mp.queues.Empty:
                return
        
    def _fit_model(self, index, model, X, y, training_params=None):
        """
        Fits the model with the given data and training parameters.

        Parameters:
        - index: Index of the model in the list.
        - model: Instance of the model to fit.
        - X: Training data features.
        - y: Training data labels.
        - training_params: Additional parameters for training the model.

        Returns:
        - Tuple containing the metrics and the fitted model.
        """
        metrics = model.fit(X, y, **(training_params or {}))
        return (metrics, model)
        
    def  _display_progress(self, models, overwrite=True):
        sorted_models = [(model, progress) for model, progress in models]
        sorted_models.sort(key=lambda x: x[0])
        
        model_strings = [
            f"{model}: {progress:3d}%" if progress < 100 else f"{model}: Done"
            for model, progress in sorted_models
        ]
        
        if overwrite:
            print("\r" + " | ".join(model_strings), end="")
        else:
            print(" | ".join(model_strings), end="")
            
    def _progress_monitor(self, models, progress_queue):
        while True:
            try:
                result = progress_queue.get(timeout=1)
                if isinstance(result, TrainingDone):
                    return
                if not isinstance(result, TrainingReport):
                    continue
                
                if result.model in models:
                    if models[result.model] < result.progress:
                        models[result.model] = result.progress
                        self._display_progress(models.items(), overwrite=True)
                        
                if all(progress >= 100 for progress in models.values()):
                    self._display_progress(models.items(), overwrite=True)
                    print("\n")
                    return
                
            except mp.queues.Empty:
                continue
        
    def fit(self, X, y, training_params=None):
        """
        Fits all models with the given training data and parameters.

        Parameters:
        - X: Training data features.
        - y: Training data labels.
        - training_params: Additional parameters for training the models. Can be a callable or a dictionary.

        Returns:
        - List of tuples containing the fitted model, their parameters, and the metrics.
        """
        
        params = list(self.combination_iterator)
        
        # Instantiate training parameters if necessary
        if callable(training_params):
            instantiated_training_params = [training_params(param) for param in params]
        else:
            instantiated_training_params = [training_params] * len(params)
            
        model_queue = mp.Queue()
        result_queue = mp.Queue()
        progress_queue = mp.Queue()
        
        # Use multiprocessing to fit models in parallel
        process_count = max(1, min(self.process_count, len(params)))
        print(f"Fitting {len(params)} models with {process_count} processes.")
        
        model_names = [f"Model {i}" for i in range(len(params))]
        models = {name: 0 for name in model_names}
        self._display_progress(models.items(), overwrite=False)
        
        progress_monitor = th.Thread(
            target=self._progress_monitor, args=(models, progress_queue)
        )
        progress_monitor.start()
        
        for i, param in enumerate(params):
            model_queue.put(TrainingOrder(i, self, param, X, y, instantiated_training_params[i]))
        
        for i in range(process_count):
            model_queue.put(TrainingDone())
            
        pool = mp.Pool(
            processes=process_count,
            initializer=self._training_worker,
            initargs=(model_queue, result_queue, progress_queue)
        )
            
        pool.close()
        
        results = []
        for i in range(len(params)):
            results.append(result_queue.get(True))
        
        progress_monitor.join()
        pool.terminate()
        
        print("All models trained. Collecting results...")
            
        print("Results collected.")
            
        return results
