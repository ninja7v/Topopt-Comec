# app/ui/workers.py
# MIT License - Copyright (c) 2025 Luc Prevost
# QThread worker for running optimizers and displacements in the background.

from PySide6.QtCore import QThread, Signal
from app.core import optimizers, displacements
import numpy as np

class OptimizerWorker(QThread):
    """Runs the topology optimization in a separate thread."""
    # Signal arguments: iteration, objective, change
    progress = Signal(int, float, float)
    # Signal arguments: xPhys_frame
    frameReady = Signal(object)
    # Signal argument: result array
    finished = Signal(np.ndarray)
    # Signal argument: error message string
    error = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.stop_requested = False
        
    def request_stop(self):
        """Public method for the main thread to request a stop."""
        print("Stop request received by worker.")
        self.stop_requested = True

    def run(self):
        """Executes the optimization based on the provided parameters."""
        try:
            optimizer_params = self.params.copy()
            keys_to_remove = ['disp_factor', 'disp_iterations']
            for key in keys_to_remove:
                optimizer_params.pop(key, None)
            
            def progress_callback(iteration, objective, change, xPhys_frame):
                self.progress.emit(iteration, objective, change)
                self.frameReady.emit(xPhys_frame)
                return self.stop_requested

            optimizer_params['progress_callback'] = progress_callback
            
            print("Dispatching to optimizer...")
            result, u = optimizers.optimize(**optimizer_params)
                
            self.finished.emit((result, u)) # Emit the tuple (xPhys, u)
        except Exception as e:
            import traceback
            error_msg = f"An error occurred during optimization:\n{e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class DisplacementWorker(QThread):
    """
    Runs the displacement analysis in a separate thread.
    """
    # Signal arguments: (current_iteration)
    progress = Signal(int)
    # Signal arguments: (xPhys_frame_data)
    frameReady = Signal(np.ndarray)
    # Signal arguments: ()
    linearResultReady = Signal(object) 
    # Signal arguments: (result_message)
    finished = Signal(str, bool)
    # Signal arguments: (error_message)
    error = Signal(str)

    def __init__(self, params: dict, xPhys_initial: np.ndarray, u_vector: np.ndarray):
        super().__init__()
        self.params = params
        self.xPhys = xPhys_initial
        self.u = u_vector
        self._stop_requested = False

    def request_stop(self):
        """Public method for the main thread to request a stop."""
        print("Stop request received by worker.")
        self._stop_requested = True

    def run(self):
        """Executes the analysis based on provided parameters."""
        try:
            def progress_callback(iteration):
                self.progress.emit(iteration)
                return self._stop_requested 

            # The function is a generator, yielding each frame
            for frame_data in displacements.run_iterative_displacement(
                self.params, self.xPhys, progress_callback
            ):
                self.frameReady.emit(frame_data)
                if self._stop_requested:
                    print("Displacement stopped by user.")
                    break
                
            self.finished.emit("Displacement finished or stopped.", True)

        except Exception as e:
            import traceback
            error_msg = f"An error occurred during displacement analysis:\n{e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)