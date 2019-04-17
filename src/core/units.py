class UnitBreaker():
  def __init__(self, gr = None, md = None, tvd = None, mud = None, *args, **kwargs):
    self._gr = gr
    self._md = md
    self._tvd = tvd
    self._mud = mud
    self._unit_boundary_flag = None
    self._stacking_patterns = None
    self._lithofacies = None
    self._sharp_boundary_flag = None
    self._gr_shape_code = None

  def break_unit(self, *args, **kwargs):
    pass

  def stack_unit(self, *args, **kwargs):
    pass

  def detect_sharp_boundary(self, *args, **kwargs):
    pass

  def compute_lithofacies(self, *args, **kwargs):
    pass

  def label_shape_code(self, *args, **kwagrs):
    pass
