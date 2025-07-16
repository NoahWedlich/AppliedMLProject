import multiprocessing as mp
import threading as th
from dataclasses import dataclass

import numpy as np

from AppliedML.courselib.optimizers import Optimizer

from models.CombinationIterator import CombinationIterator

@dataclass        
class TrainingOrder:
    """
    Data structure to hold the information needed to train a model.
    """
    
    index: int
    model: any
    params: dict
    X: any
    y: any
    training_params: dict = None
    
@dataclass
class TrainingReport:
    """
    Data structure to hold a training progress report.
    """
    
    model: str
    progress: int
        
class TrainingDone:
    """
    Signal to indicate that training is complete.
    """
    pass

class CommunicationsManager:
    """
    Wrapper for managing communication between processes during model training.
    """
    
    def __init__(self):
        """
        Initializes the communication manager with queues for models, results, and progress.
        """
        self.model_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.progress_queue = mp.Queue()
        
    def get_model_queue(self):
        """
        Returns the queue for model training orders.
        
        Returns:
        - mp.Queue: The queue for sending model training orders.
        """
        return self.model_queue
        
    def get_result_queue(self):
        """
        Returns the queue for receiving training results.
        
        Returns:
        - mp.Queue: The queue for receiving results from training processes.
        """
        return self.result_queue
        
    def get_progress_queue(self):
        """
        Returns the queue for receiving training progress reports.
        
        Returns:
        - mp.Queue: The queue for receiving progress updates from training processes.
        """
        return self.progress_queue

class ReportingOptimizer(Optimizer):
    """
    Wrapper for optimizers that also reports training progress.
    """
    
    def __init__(self, optimizer, model, num_epochs, queue):
        """
        Initializes the reporting optimizer.
        
        Parameters:
        - optimizer: The optimizer to be wrapped.
        - model: The model_identifier being trained.
        - num_epochs: Total number of epochs for training.
        - queue: Queue to send training progress reports.
        """
        
        self.optimizer = optimizer
        self.model = model
        self.epoch = 0
        self.num_epochs = num_epochs
        self.queue = queue
        
        self.freq = self.num_epochs // 100 if self.num_epochs > 100 else 1
        
    def update(self, params, grads):
        """
        Updates the model parameters and sends a progress report.
        
        Parameters:
        - params: Current model parameters.
        - grads: Gradients computed for the parameters.
        """
        self.epoch += 1
        if self.epoch % self.freq == 0 or self.epoch == self.num_epochs:
            progress = int((self.epoch / self.num_epochs) * 100)
            self.queue.put(TrainingReport(self.model, progress))
            
        self.optimizer.update(params, grads)

class TunableModel:
    """
    Represents a model that can be tuned with various hyperparameters.
    """

    def __init__(
        self, model_class, hyperparameters, validator=None, process_count=None, random_seed=None
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
        
        self.random_seed = random_seed

    def get_model(self, parameters):
        """
        Returns an instance of the model with the given parameters.

        Parameters:
        - parameters: Dictionary of hyperparameters for the model.

        Returns:
        - An instance of the model class initialized with the given parameters.
        """
        np.random.seed(self.random_seed)
        return self.model_class(**parameters)
        
    def _training_worker(self, comm_manager):
        """
        A worker function for training models in parallel.
        
        Parameters:
        - comm_manager: CommunicationsManager instance for managing queues.
        """
        while True:
            try:
                # Get the next training order from the queue
                training_order = comm_manager.get_model_queue().get(timeout=0.25)
                if isinstance(training_order, TrainingDone):
                    return
                if not isinstance(training_order, TrainingOrder):
                    continue
                    
                cf = training_order
                
                # Instantiate the optimizer if it's callable
                if callable(cf.params.get('optimizer')):
                    cf.params['optimizer'] = cf.params['optimizer'](cf.params)
                
                # Wrap the optimizer with a ReportingOptimizer if possible
                if cf.training_params is not None:
                    injected_progress = cf.params.get('optimizer') is not None
                    injected_progress = injected_progress and cf.training_params.get('num_epochs') is not None
                    injected_progress = injected_progress and cf.training_params.get('batch_size') is not None
                else:
                    injected_progress = False
                
                if injected_progress:
                    num_batches = int(len(cf.X) / cf.training_params['batch_size'])
                    num_epochs = cf.training_params.get('num_epochs') * num_batches
                    
                    cf.params['optimizer'] = ReportingOptimizer(
                        cf.params['optimizer'], f"Model {cf.index}", num_epochs, comm_manager.get_progress_queue()
                    )
                
                # Train the model with the provided parameters
                model_instance = cf.model.get_model(cf.params)
                metrics = model_instance.fit(cf.X, cf.y, **(cf.training_params or {}))
                
                # Unwrap the optimizer if it was wrapped
                if injected_progress:
                    model_instance.optimizer = model_instance.optimizer.optimizer
                    cf.params['optimizer'] = model_instance.optimizer
                
                # Send the result back to the main process
                comm_manager.get_result_queue().put((cf.index, model_instance, cf.params, metrics))
                
                # Send a finished report
                comm_manager.get_progress_queue().put(TrainingReport(f"Model {cf.index}", 100))
            except mp.queues.Empty:
                return
        
    def  _display_progress(self, models, overwrite=True):
        """
        Displays the progress of model training in a formatted string.
        
        Parameters:
        - models: List of tuples containing model names and their progress.
        - overwrite: If True, overwrites the previous output in the console.
        """
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
            
    def _progress_monitor(self, models, comm_manager):
        """
        Listens for and displays training progress updates from the communication manager.
        
        Parameters:
        - models: Dictionary to track the progress of each model.
        - comm_manager: CommunicationsManager instance for receiving progress updates.
        """
        while True:
            try:
                # Get the next progress report from the queue
                result = comm_manager.get_progress_queue().get(timeout=1)
                if isinstance(result, TrainingDone):
                    return
                if not isinstance(result, TrainingReport):
                    continue
                
                # Update the model's progress
                if result.model in models:
                    if models[result.model] < result.progress:
                        models[result.model] = result.progress
                        self._display_progress(models.items(), overwrite=True)
                        
                # If all models are done, exit the loop
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
            
        comm_manager = CommunicationsManager()
        
        # Determine the number of processes to use
        process_count = max(1, min(self.process_count, len(params)))
        print(f"Fitting {len(params)} models with {process_count} processes.")
        
        # Create the labels for the models
        model_names = [f"Model {i}" for i in range(len(params))]
        models = {name: 0 for name in model_names}
        self._display_progress(models.items(), overwrite=False)
        
        # Start the progress monitor thread
        progress_monitor = th.Thread(
            target=self._progress_monitor, args=(models, comm_manager)
        )
        progress_monitor.start()
        
        # Push the training orders to the queue
        for i, param in enumerate(params):
            comm_manager.get_model_queue().put(TrainingOrder(i, self, param, X, y, instantiated_training_params[i]))
        
        # Add the signals to indicate training completion after all models are trained
        for i in range(process_count):
            comm_manager.get_model_queue().put(TrainingDone())
            
        # Start the training processes
        pool = mp.Pool(
            processes=process_count,
            initializer=self._training_worker,
            initargs=(comm_manager,)
        )
        pool.close()
        
        # Collect the results from the training processes
        results = []
        for i in range(len(params)):
            results.append(comm_manager.get_result_queue().get(True))
        
        progress_monitor.join()
        pool.terminate()
            
        return results
