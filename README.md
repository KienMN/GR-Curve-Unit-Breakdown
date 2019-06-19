# Gamma Ray curve units breakdown

## Introduction
This package provides several functions to deal with gamma ray curve in geological field.

## Installation
Installing the package as a library in Python by the command.
```bash
pip insatll git+https://github.com/KienMN/GR-Curve-Unit-Breakdown.git
```

## Dependencies
numpy  
pandas  
matplotlib  
seaborn  
pywt  
statsmodels

## Usage
List of major methods. Details parameters and returns are described in each method definition.
```python
UnitBreaker().fill_null_values()
UnitBreaker().detect_changing_direction_point()
UnitBreaker().refine_peak()
UnitBreaker().select_boundary()
UnitBreaker().stack_unit()
UnitBreaker().detect_sharp_boundary()
UnitBreaker().detect_lithofacies()
UnitBreaker().label_shape_code()
UnitBreaker().assign_unit_index()
UnitBreaker().find_similar_units()
UnitBreaker().visualize_units_boundary()
UnitBreaker().visualize_units()
```
Addition methods are included in smoothing_functions.py, dataset.py and utils.py.

## Testing
Test cases are written following format of unittest module. Run test files by command.
```bash
python3 tests/{test_file_name}.py
```

## Visualization
Sample visualization of units on the GR curve.
![Alt text](result_pic/demo.png?raw=true "Sample visualization")
