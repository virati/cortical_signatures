import json


class segment:
    def __init__(self, **kwargs):
        if "config_file" in kwargs:
            self.config = json.load(open(kwargs["config_file"]))
        else:
            self.num_features = kwargs["num_features"]
            self.num_segments = kwargs["num_segments"]
            self.num_conditions = kwargs["num_conditions"]
            self.num_patients = kwargs["num_patients"]
            self.num_times = kwargs["num_times"]

        self.dim_order = ["features", "segments", "times", "condition", "patients"]

    def check_consistency(self):
        if self._data.shape != [getattr(self, "num_" + key) for key in self.dim_order]:
            raise ValueError("Data shape does not match the dimensions specified")
