# app/ui/worker.py
# MIT License - Copyright (c) 2025 Luc Prevost
# QThread worker for running optimizers in the background.

from PySide6.QtCore import QThread, Signal
from app.optimizers import optimizer_2d, optimizer_3d
from app.displacements import displacement_2d, displacement_3d
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
                
            is_3d = optimizer_params['nelxyz'][2] > 0
            
            def progress_callback(iteration, objective, change, xPhys_frame):
                self.progress.emit(iteration, objective, change)
                self.frameReady.emit(xPhys_frame)
                return self.stop_requested

            optimizer_params['progress_callback'] = progress_callback
            
            if is_3d:
                print("Dispatching to 3D optimizer...")
                # The optimizer function doesn't know about 'u', so we can't get it directly here
                # We need to call a function that returns it
                result, u = optimizer_3d.optimize(**optimizer_params)
            else:
                print("Dispatching to 2D optimizer...")
                # For 2D, remove 3D-specific keys from the already cleaned dictionary
                params_2d = optimizer_params.copy()
                params_2d.pop('fz', None)
                params_2d.pop('sz', None)
                params_2d['nelxyz'] = params_2d['nelxyz'][:2]
                
                result, u = optimizer_2d.optimize(**params_2d)
                
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

    def __init__(self, params: dict, xPhys_initial: np.ndarray):
        super().__init__()
        self.params = params
        self.xPhys = xPhys_initial

    def run(self):
        """Executes the analysis based on provided parameters."""
        try:
            is_3d = self.params['nelxyz'][2] > 0
            
            def progress_callback(iteration):
                self.progress.emit(iteration)

            if is_3d:
                # The function is a generator, yielding each frame
                for frame_data in displacement_3d.run_iterative_displacement_3d(
                    self.params, self.xPhys, progress_callback
                ):
                    self.frameReady.emit(frame_data)
            else:
                # The function is a generator, yielding each frame
                for frame_data in displacement_2d.run_iterative_displacement_2d(
                    self.params, self.xPhys, progress_callback
                ):
                    self.frameReady.emit(frame_data)
                
            self.finished.emit("Displacement animation complete.", True)

        except Exception as e:
            import traceback
            error_msg = f"An error occurred during displacement analysis:\n{e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)