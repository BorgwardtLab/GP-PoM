"""Implementation of dataset base class."""
import abc


class Dataset(abc.ABC):
    """Abstract base class of a dataset."""

    @abc.abstractmethod
    def __len__(self):
        """Get the size (number of instances) of the dataset."""

    @abc.abstractmethod
    def __getitem__(self, index):
        """Get a single instance."""

    @property
    @abc.abstractmethod
    def measurement_dims(self):
        """Get dimensionality of a measurement."""

    @property
    @abc.abstractmethod
    def n_classes(self):
        """Get number of classes."""

    # @property
    # @abc.abstractmethod
    # def task(self):
    #     """Return the task specific for this dataset."""

    def __repr__(self):
        """Get representation of Dataset."""
        classname = self.__class__.__name__
        output = \
            f'<{classname} '\
            f'measurement_dims={self.measurement_dims}, '\
            f'n_classes={self.n_classes}>'
        for split in self.split_names:
            output += repr(self.get_data(split))
        output += f'</ {classname}Dataset>'
        return output
