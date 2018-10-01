from inspect import isabstract
from CHECLabPy.core.base_reducer import WaveformReducer
from CHECLabPy.core.spectrum_fitter import SpectrumFitter


class Factory:
    """
    Factory to provide a list of subclasses for a base class, and allow its
    selection at runtime.
    """
    subclasses = None  # Factory.child_subclasses(ParentClass)
    subclass_names = None  # [c.__name__ for c in subclasses]

    @staticmethod
    def child_subclasses(base):
        """
        Return all non-abstract subclasses of a base class.

        Parameters
        ----------
        base : class
            high level class object that is inherited by the
            desired subclasses

        Returns
        -------
        children : list
            list of non-abstract subclasses

        """
        family = base.__subclasses__() + [
            g for s in base.__subclasses__()
            for g in Factory.child_subclasses(s)
        ]
        children = [g for g in family if not isabstract(g)]

        unique = []
        unique_names = []
        for c in children:
            if c.__name__ not in unique_names:
                unique.append(c)
                unique_names.append(c.__name__)

        return unique

    @classmethod
    def produce(cls, product_name, *args, **kwargs):
        print("Obtaining {} from {}".format(product_name, cls.__name__))
        factory = cls()
        subclass_dict = dict(zip(factory.subclass_names, factory.subclasses))

        try:
            product = subclass_dict[product_name]
        except KeyError:
            msg = ('No product found with name "{}" '
                   'for factory.'.format(product_name))
            raise KeyError(msg)

        return product(*args, **kwargs)


class WaveformReducerFactory(Factory):
    import CHECLabPy.waveform_reducers
    subclasses = Factory.child_subclasses(WaveformReducer)
    subclass_names = [c.__name__ for c in subclasses]


class SpectrumFitterFactory(Factory):
    import CHECLabPy.spectrum_fitters
    subclasses = Factory.child_subclasses(SpectrumFitter)
    subclass_names = [c.__name__ for c in subclasses]
