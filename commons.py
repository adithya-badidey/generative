from turtle import color
from PIL import Image, ImageDraw
import math
import numpy as np
import matplotlib.pyplot as plt

strict = False
random = np.random.default_rng()

def lineplot(p1, p2, s):
  """A generator which generates s points between the start and the end points
  """
  if s == 0:
    return
  if s == 1:
    yield p1, (0,0)
    return
  x1, y1 = p1
  x2, y2 = p2
  s -= 1 
  dx = (x2-x1)/s
  dy = (y2-y1)/s
  for i in range(s):
    t = (s-i)/s
    x = x1 * t + x2 * (1-t)
    y = y1 * t + y2 * (1-t)
    yield (x, y), (dx, dy)
      
def rotate(vec, angle):
  # Rotates a vector by (angle) degrees
  theta = np.deg2rad(angle)
  rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
  return np.dot(rot, vec)

def clamp(n, smallest, largest): return max(smallest, min(n, largest))
  
class Drawing:
  def __init__(self, image, scaling=1, span=None):
    self.img = image
    if span is None:
      self.img_height = image.height//scaling
      self.img_width = image.width//scaling
      self.base_x = 0
      self.base_y = 0
    else:
      self.base_x, x = span[0]
      self.img_width = (x - self.base_x)

      self.base_y, y = span[0]
      self.img_height = (y - self.base_y)
      
    self.scaling = scaling

  def _transform(self, arr):
    return (int(arr[0]*(self.img_width-1)), int(arr[1]*(self.img_height-1)))
    
  def point(self, p, color, abs=True):
    # print(p)
    if abs and 0 <= p[0] < self.img_width and 0 <= p[1] < self.img_height:
      pass #all is good
    elif (not abs) and 0 <= p[0] <= 1 and 0 <= p[1] <= 1:
      p = self._transform(p)
    else:
      if strict:
        print(f"Warning: drawing.point{p} out of range.")
      return

    p = [(self.base_x +int(p[0])) * self.scaling, (self.base_y + int(p[1])) * self.scaling]
    
    if self.scaling == 1:
      self.img.putpixel(p, color)
    else:
      for i in range(self.scaling):
        for j in range(self.scaling):
          self.img.putpixel(( p[0] + i, p[1] + j), color)

  def line(self, p1, p2, brush, steps = 1):
    # for p, dp in lineplot(p1, p2, steps):
    brush.stroke(p1, p2, self)

class Brush:
  def stroke(self, start, end, drawing: Drawing):
    raise NotImplementedError("stroke not implemented")

class NoisyBrush(Brush):
  def __init__(self, spread, mean_length, sd_length, spread_factor=2, num=10, color=(0,0,0)):
    self.spread_factor = spread_factor
    self.mean_length = mean_length
    self.sd_length = sd_length
    self.spread = spread
    self.color = color
    self.num = num

  def stroke(self, start, end, drawing: Drawing):
    start = drawing._transform(start)
    end = drawing._transform(end)
    # First we calculate the number of steps 
    # (adding 2 here to compress the stroke and avoid the errors due to math)
    steps = max(abs(start[0] - end[0]), abs(start[1] - end[1])) + 2 

    # This keeps track of the bristles in action. 
    # bristles[normal_offset] = (start_offset, end_offset)
    # where normal_offset is {-inf, +inf} and adds an offset perpendicular to
    # the direction of the stroke.
    bristles = {}

    for step, pos in enumerate(lineplot(start, end, steps)):
      p, dp = pos
      # dp is a tangent to the stroke and pdp is normal to the stroke
      pdp = rotate(dp, 90) 

      for i in list(bristles.keys()):
        start_offset, end_offset = bristles[i]
        if end_offset < step:
          del bristles[i]

      # This ensures there are a full number of bristles
      if len(bristles) < self.num:     
        for i in range(5): # Try 10 times to get valid bristles
          for i in random.normal(0, self.spread, self.num - len(bristles)):
            i = int(i)
            if i not in bristles:
              # start = step + abs(random.normal(0,self.sd_length,1))[0]
              start = step
              mean_length = self.mean_length/((abs(i)+1) ** self.spread_factor)
              length =  random.normal(mean_length, self.sd_length, 1)[0]
              bristles[i] = (start, start + length)
          if len(bristles) >= self.num:
            break
          # print(bristles[0])
      
      for i in list(bristles.keys()):
        start_offset, end_offset = bristles[i]
        if start_offset <= step:
          drawing.point(p + pdp*(i/1.2), self.color) #Some compression here
      drawing.point(p, self.color)

class Gradient:
  def __init__(self, *args) -> None:
    if len(args) == 2 and isinstance(args[0], np.ndarray):
      self.gradient = [(0, args[0]), (1, args[1])]
    else:
      self.gradient = args

  def getColor(self, x):
    for i in range(len(self.gradient)-1):
      limit1, color1 = self.gradient[i]
      limit2, color2 = self.gradient[i+1]
      if limit1 <= x <= limit2:
        x = (x - limit1)/(limit2 - limit1)
        return color2 * x + color1 * (1-x)
  
  def plot(self):
    fig, ax = plt.subplots(2, 1)

    r = np.array([self.getColor(i/99) for i in range(100)])

    ax[0].plot(r[:,0], color="red")
    ax[0].plot(r[:,1], color="green")
    ax[0].plot(r[:,2], color="blue")

    img = np.array(r, dtype=int).reshape((1, len(r), 3))
    ax[1].imshow(img, extent=[0, 16000, 0, 1], aspect='auto')
    return fig, ax


def hexcode(hexcode):
    hexcode=hexcode.strip(" #")
    if len(hexcode) == 6:
      return np.array([int(hexcode[i:i+2], 16) for i in (0, 2, 4)])
    elif len(hexcode) == 3:
      return np.array([int(hexcode[i], 16)*16 for i in (0, 1, 2)])
    else:
      raise Exception(f"Invalid hexcode({hexcode})")
