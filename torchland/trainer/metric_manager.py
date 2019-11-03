from collections import defaultdict
import math


class MetricManager:
    """Class for managing multiple metrics."""
    def __init__(self):
        self.metric_counter = defaultdict(int)  # init to 0
        self.metric_avgs = defaultdict(float)  # init to 0.0
        self.metric_mins = defaultdict(lambda: math.inf)
        self.metric_maxes = defaultdict(lambda: -math.inf)

    def append_metric(self, metric: dict):
        """
        Introduce a new metric values and update the statistics.
        It mainly updates the count, average value, minimum value, and the maximum value.

        Args:
            metric (dict): various metric values
        """
        for key, val in metric.items():
            prev_count = self.metric_counter[key]
            prev_avg = self.metric_avgs[key]
            total_val = prev_count * prev_avg + val

            # calculate the new average
            self.metric_avgs[key] = total_val / (prev_count + 1)
            self.metric_counter[key] = prev_count + 1
            if val < self.metric_mins[key]:
                self.metric_mins[key] = val
            if val > self.metric_maxes[key]:
                self.metric_maxes[key] = val

    def mean(self, key: str) -> float:
        """
        Retrieve the mean value of the given key.

        Args:
            key (str): the key value

        Returns:
            the mean value of the key
        """
        return self.metric_avgs[key]

    def set_mean(self, key: str, val):
        self.metric_avgs[key] = val
