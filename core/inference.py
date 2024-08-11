import time
from typing import List, Dict, Any

import torch
import tqdm
from torch.utils.data import DataLoader

from torchpack.callbacks.callback import Callback, Callbacks
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from torchpack.utils.typing import Trainer

__all__ = ['InferenceRunner']

# class CustomCallbacks(Callbacks):
#     def __init__(self, callbacks):
#         super().__init__(callbacks)
#         self.step_miou = None

#     def get_step_miou(self):
#         """
#         Placeholder method for getting step miou.
#         """
#         raise NotImplementedError("Subclasses must implement get_step_miou method.")



class InferenceRunner(Callback):
    """
    A callback that runs inference with a list of :class:`Callback`.
    """
    def __init__(self, dataflow: DataLoader, *,
                 callbacks: List[Callback]) -> None:
        self.dataflow = dataflow
        self.callbacks = Callbacks(callbacks)
        # self.miou = None

    def _set_trainer(self, trainer: Trainer) -> None:
        self.callbacks.set_trainer(trainer)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        self.callbacks.before_epoch()
        self.callbacks.after_step(output_dict)

    def trigger_step(self) -> float:
        

        self.callbacks.trigger_step()
        # miou = self.callbacks.get_step_miou()
        # print(miou)
        # # logger.info('Inference finished in {}.'.format(
        # #     humanize.naturaldelta(time.perf_counter() - start_time)))
        # return miou

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        start_time = time.perf_counter()
        self.callbacks.before_epoch()

        with torch.no_grad():
            for feed_dict in tqdm.tqdm(self.dataflow, ncols=0):
                self.callbacks.before_step(feed_dict)
                output_dict = self.trainer.run_step_inference(feed_dict)
                self.callbacks.after_step(output_dict)

        self.callbacks.after_epoch()
        # self.miou = self.callbacks.get_miou()
        logger.info('Inference finished in {}.'.format(
            humanize.naturaldelta(time.perf_counter() - start_time)))

    # def get_step_miou(self):
    #     res = self.callbacks.step_miou
    #     print(res)
    #     return res