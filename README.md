## Neural Framework

Simple, but yet powerful neural computation framework for fast prototyping.

### Install with pip
```
pip install git+https://github.com/vegardbotnen/neural_framework.git
```

### Upgrade with pip
```
pip install --upgrade git+https://github.com/vegardbotnen/neural_framework.git
```

### Build new package
```
python setup.py sdist bdist_wheel
```


### How To
```python
from neural_framework.brain import Brain, fully_connected

fc = fully_connected([2,4,4,1])

fc.excite_neuron(0, 1.0)
fc.step()
fc.step()
fc.step()

outputs = fc.get_activation([10])
print(outputs)


fc.show()
```
