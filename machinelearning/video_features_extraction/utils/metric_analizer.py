

class MetricAnalizer:
    def __init__(self, name: str):
        self.name = name
        self.sum_of_values = 0
        self.max_singe_change = 0
        self.sum_of_changes = 0
        self.prev_value = None

        self.all_counter = 0
        self.change_counter = 0

        # variance from wikipedia
        self.K = 0.0
        self.n = 0.0
        self.Ex = 0.0
        self.Ex2 = 0.0

        self.last_three_values = []

        self.interpolation_sum = 0.0
        self.interpolation_max = 0.0
        self.interpolation_counter = 0


    def update(self, value):
        # avg-value, avg-change, max-change
        if self.prev_value is not None:
            change = abs(self.prev_value - value)
            self.sum_of_changes += change
            self.change_counter += 1

            if change > self.max_singe_change:
                self.max_singe_change = change

        self.sum_of_values += value
        self.prev_value = value
        self.all_counter += 1

        # variance
        if self.n == 0:
            self.K = value
        self.n += 1
        self.Ex += value - self.K
        self.Ex2 += (value - self.K) * (value - self.K)

        # avg-interp-diff, max-interp-diff
        self.last_three_values.append(value)
        if len(self.last_three_values) > 3:
            self.last_three_values.pop(0)

        if len(self.last_three_values) == 3:
            interpolation = (self.last_three_values[2] + self.last_three_values[0]) / 2
            diff = abs(interpolation - self.last_three_values[1])

            self.interpolation_sum += diff
            self.interpolation_counter += 1
            if diff > self.interpolation_max:
                self.interpolation_max = diff


    def get_metrics(self) -> dict:
        return {
            "{}-avg-value".format(self.name): self.sum_of_values / self.all_counter if self.all_counter else 0,
            "{}-variance-value".format(self.name): (self.Ex2 - (self.Ex * self.Ex) / self.n) / (self.n - 1) if self.n > 1 else 0,
            "{}-avg-change".format(self.name): self.sum_of_changes / self.change_counter if self.change_counter else 0,
            "{}-max-change".format(self.name): self.max_singe_change,
            "{}-avg-interp-diff".format(self.name): self.interpolation_sum / self.interpolation_counter if self.interpolation_counter else 0,
            "{}-max-interp-diff".format(self.name): self.interpolation_max
            }
