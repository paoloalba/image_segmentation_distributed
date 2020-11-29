import copy
import time
import logging

import numpy as np
import tensorflow as tf

class OverrideProgbarLogger(tf.keras.callbacks.ProgbarLogger):

  def on_epoch_begin(self, epoch, logs=None):
    self._reset_progbar()
    if self.verbose and self.epochs > 1:
      logging.info('Epoch %d/%d' % (epoch + 1, self.epochs))

  def _maybe_init_progbar(self):
    if self.stateful_metrics is None:
      if self.model:
        self.stateful_metrics = (set(m.name for m in self.model.metrics))
      else:
        self.stateful_metrics = set()

    if self.progbar is None:
      self.progbar = OverrideProgbar(
          target=self.target,
          verbose=self.verbose,
          stateful_metrics=self.stateful_metrics,
          unit_name='step' if self.use_steps else 'sample')

  def _batch_update_progbar(self, batch, logs=None):
    """Updates the progbar."""
    logs = logs or {}
    self._maybe_init_progbar()
    if self.use_steps:
      self.seen = batch + 1  # One-indexed.
    else:
      # v1 path only.
      logs = copy.copy(logs)
      batch_size = logs.pop('size', 0)
      num_steps = logs.pop('num_steps', 1)
      logs.pop('batch', None)
      add_seen = num_steps * batch_size
      self.seen += add_seen

class OverrideProgbar(tf.keras.utils.Progbar):

  def update(self, current, values=None, finalize=None):
    if finalize is None:
      if self.target is None:
        finalize = False
      else:
        finalize = current >= self.target

    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        value_base = max(current - self._seen_so_far, 1)
        if k not in self._values:
          self._values[k] = [v * value_base, value_base]
        else:
          self._values[k][0] += v * value_base
          self._values[k][1] += value_base
      else:
        self._values[k] = [v, 1]
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if now - self._last_update < self.interval and not finalize:
        return

      prev_total_width = self._total_width

      if self.target is not None:
        numdigits = int(np.log10(self.target)) + 1
        bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width - 1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
      else:
        bar = '%7d/Unknown' % current

      self._total_width = len(bar)
      info = bar + info

      if current:
        time_per_unit = (now - self._start) / current
      else:
        time_per_unit = 0

      if self.target is None or finalize:
        if time_per_unit >= 1 or time_per_unit == 0:
          info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
        else:
          info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)
      else:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600,
                                         (eta % 3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta

        info = ' - ETA: %s' % eta_format

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))

      logging.info(info)

    elif self.verbose == 2:
      if finalize:
        numdigits = int(np.log10(self.target)) + 1
        count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
        info = count + info
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg

        logging.info(info)

    self._last_update = now

