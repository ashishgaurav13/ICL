from tools.base import Environment
from tools.data import Features
from tools.graphics import Group, Text2D
from tools.math import Direction2D, Box2D
import pyglet
from pyglet import gl
import numpy as np
from inspect import isfunction
import os
from tools.graphics import Image2D
from tools.utils.misc import get_package_root_path
import numba

class Car2D(Group, Environment):
    """
    A Car2D is a special drawable that behaves like a 2D car.
    Taken from github.com/ashishgaurav13/wm2
    """

    MAX_ACCELERATION = 2.0
    MAX_STEERING_ANGLE_RATE = 1.0
    VEHICLE_WHEEL_BASE = 1.7
    DT = 0.1
    DT_over_2 = DT / 2
    DT_2_over_3 = 2 * DT / 3
    DT_over_3 = DT / 3
    DT_over_6 = DT / 6
    VEHICLE_X = 2.5
    VEHICLE_Y = 1.7
    THETA_DEVIATION_ALLOWED = np.pi/2
    MAX_STEERING_ANGLE = np.pi/3
    SAFETY_GAP = 6.0
    SPEED_MAX = 11.176 # 11.176 = 40kmph

    def __init__(self, x, y, v, ego = False, direction = None, canvas = None,
        name = None, centerpts = None, angles = None, h=None, w=None, frames=None,
        frame = None, speedxs = None, speedys = None, accxs = None, accys = None, a=None):
        """
        Initialize Car2D.
        theta, psi cannot be specified, it is created based on direction
        """
        assert(type(direction) == Direction2D)
        assert(canvas != None)
        assert(name != None)
        self.name = name
        self.method = 'kinematic_bicycle_RK4'
        self.ego = ego
        self.direction = direction
        dir_angle = direction.angle()
        self.original_features = {
            'x': self.parse_init_function(x),
            'y': self.parse_init_function(y),
            'v': self.parse_init_function(v),
            'theta': self.parse_init_function(dir_angle)
        }
        self.of = self.original_features
        self.features = Features({
            'x': self.of['x'](), 'y': self.of['y'](), 'v': self.of['v'](),
            'acc': 0.0, 'psi_dot': 0.0, 'psi': 0.0,
            'theta': self.of['theta'](),
        })
        self.a = a
        self.f = self.features # shorthand
        if self.a != None: self.f['acc'] = a
        self.canvas = canvas
        ox, oy, scale = canvas.ox, canvas.oy, canvas.scale
        x = ox + self.f['x'] * scale
        y = oy + self.f['y'] * scale
        if w == None:
            w = self.VEHICLE_X
        if h == None:
            h = self.VEHICLE_Y
        w = w*scale*canvas.agentscale
        h = h*scale*canvas.agentscale
        self.centerpts = centerpts
        self.angles = angles
        self.frames = frames
        self.speedxs, self.speedys = speedxs, speedys
        self.accxs, self.accys = accxs, accys
        self.frame = frame
        self.t = 0
        self.fr = 0
        if self.ego:
            vehicle_url = "assets/driving/ego.png"
        else:
            vehicle_url = "assets/driving/vehicle.png"
        car = Image2D(vehicle_url, x, y, w, h, \
            np.rad2deg(self.f['theta']), anchor_centered=True)
        super().__init__(items = [car])
        self.ego_appeared = False
        if self.frame is None:
            self.ego_appeared = True
        if self.frame is not None and not self.ego_appeared:
            assert(self.ego)
            if self.fr == self.frame:
                self.f = Features({
                    'x': self.of['x'](), 'y': self.of['y'](), 'v': self.of['v'](),
                    'acc': 0.0, 'psi_dot': 0.0, 'psi': 0.0,
                    'theta': self.of['theta'](),
                })
                if self.a != None: self.f['acc'] = self.a
                ox, oy, scale = self.canvas.ox, self.canvas.oy, self.canvas.scale
                x = ox + float(self.f['x']) * scale
                y = oy + float(self.f['y']) * scale
                # print(self.f['x'], self.f['y'], x, y, 'init')
                self.items[0].items[0].update(x = x, y = y)
                self.ego_appeared = True
                # print('ego appeared at', self.fr)
            else:
                self.f['x'] = np.inf
                self.f['y'] = np.inf
                self.items[0].items[0].update(x = 3*self.canvas.ox, y = 3*self.canvas.oy)
        if self.frames is not None:
            assert(not self.ego)
            if self.t < len(self.frames) and self.fr == self.frames[self.t]:
                ox, oy, scale = self.canvas.ox, self.canvas.oy, self.canvas.scale
                x = ox + float(self.centerpts[self.t][0]) * scale
                self.f['x'] = self.centerpts[self.t][0]
                y = oy + float(self.centerpts[self.t][1]) * scale
                self.f['y'] = self.centerpts[self.t][1]
                self.f['v'] = np.sqrt(self.speedxs[self.t]**2+self.speedys[self.t]**2)
                self.f['acc'] = np.sqrt(self.accxs[self.t]**2+self.accys[self.t]**2)
                a = self.angles[self.t]
                self.f['theta'] = a
                self.items[0].items[0].update(x = x, y = y, rotation = \
                    np.rad2deg(a))
                self.t += 1
            elif self.fr < self.frames[0] or self.fr > self.frames[-1]:
                self.f['x'] = np.inf
                self.f['y'] = np.inf
                self.f['theta'] = np.inf
                self.f['v'] = np.inf
                self.f['acc'] = np.inf
                self.items[0].items[0].update(x = 3*self.canvas.ox, y = 3*self.canvas.oy)
    
    def parse_init_function(self, x):
        """
        Produces an intialization function; if passed a constant
        lambda-wrap it, else return it as is
        """
        if not isfunction(x):
            return lambda: x
        else:
            return x

    def seed(self, s=None):
        """
        Seed the car. Since the environment does this, this function is blank.
        """
        return

    def render(self, **kwargs):
        """
        Render this Car2D (dummy).
        """
        pass

    @property
    def state(self):
        """
        Current state.
        """
        return self.f.numpy()

    def reset(self):
        """
        Reset this car to original features.
        """
        self.t = 0
        self.fr = 0
        for attr in ['x', 'y', 'v', 'theta']:
            self.f[attr] = self.of[attr]()
        if self.a != None: self.f['acc'] = self.a
        self.ego_appeared = False
        if self.frame is None:
            self.ego_appeared = True
        if self.frame is not None and not self.ego_appeared:
            assert(self.ego)
            if self.fr == self.frame:
                self.f = Features({
                    'x': self.of['x'](), 'y': self.of['y'](), 'v': self.of['v'](),
                    'acc': 0.0, 'psi_dot': 0.0, 'psi': 0.0,
                    'theta': self.of['theta'](),
                })
                if self.a != None: self.f['acc'] = self.a
                ox, oy, scale = self.canvas.ox, self.canvas.oy, self.canvas.scale
                x = ox + float(self.f['x']) * scale
                y = oy + float(self.f['y']) * scale
                # print(self.f['x'], self.f['y'], x, y, 'reset')
                self.items[0].items[0].update(x = x, y = y)
                self.ego_appeared = True
                # print('ego appeared at', self.fr)
            else:
                self.f['x'] = np.inf
                self.f['y'] = np.inf
                self.items[0].items[0].update(x = 3*self.canvas.ox, y = 3*self.canvas.oy)
        if self.frames is not None:
            assert(not self.ego)
            if self.t < len(self.frames) and self.fr == self.frames[self.t]:
                ox, oy, scale = self.canvas.ox, self.canvas.oy, self.canvas.scale
                x = ox + float(self.centerpts[self.t][0]) * scale
                self.f['x'] = self.centerpts[self.t][0]
                y = oy + float(self.centerpts[self.t][1]) * scale
                self.f['y'] = self.centerpts[self.t][1]
                a = self.angles[self.t]
                self.f['theta'] = a
                self.f['v'] = np.sqrt(self.speedxs[self.t]**2+self.speedys[self.t]**2)
                self.f['acc'] = np.sqrt(self.accxs[self.t]**2+self.accys[self.t]**2)
                self.items[0].items[0].update(x = x, y = y, rotation = \
                    np.rad2deg(a))
                self.t += 1
            elif self.fr < self.frames[0] or self.fr > self.frames[-1]:
                self.f['x'] = np.inf
                self.f['y'] = np.inf
                self.f['theta'] = np.inf
                self.f['v'] = np.inf
                self.f['acc'] = np.inf
                self.items[0].items[0].update(x = 3*self.canvas.ox, y = 3*self.canvas.oy)

    def which_allowed_regions(self, filter_fn = None):
        """
        Which regions is the car in? (Apply filter if provided)
        """
        in_regions = []
        for region in self.canvas.allowed_regions:
            if type(region) == Box2D and \
                region.inside(self.f['x'], self.f['y']):
                if filter_fn and filter_fn(region):
                    in_regions += [region]
        return in_regions
    
    def collided(self, gap=5.0):
        """
        Has the car collided with any other agent?
        """
        for agent in self.canvas.agents:
            distances = []
            if agent is not self:
                distances += [self.Lp(agent.f['x'], agent.f['y'])]
                if self.Lp(agent.f['x'], agent.f['y']) <= gap:
                    return True
        return False
    
    def in_allowed_regions(self, search_word = ''):
        """
        Which regions is the car in, where the region name has `search_word`?
        """
        return self.which_allowed_regions(filter_fn = \
            lambda x: search_word in x.name) != []
    
    def Lp(self, x, y, p = 2):
        """
        Lp norm of displacement from (x, y).
        """
        return float(np.linalg.norm([self.f['x']-x, self.f['y']-y], p))

    def agents_in_front_behind(self):
        """
        Return relevant agents in front or behind.
        """
        all_cars = [agent for agent in self.canvas.agents if agent is not self]
        rx1, rx2, ry1, ry2 = self.lane_boundaries() # cars we really want to consider
        within_range_cars = [agent for agent in all_cars \
            if (rx1 <= agent.f['x'] <= rx2 and ry1 <= agent.f['y'] <= ry2)]
        return within_range_cars

    def lane_boundaries(self):
        """
        Return lane bounds (x1, x2, y1, y2).
        Either x1/x2 or y1/y2 is supposed to be -np.inf/np.inf.
        Will not work if direction is not exactly vertical or horizontal.
        """
        assert(self.direction.mode in ['+x', '-x', '+y', '-y'])
        assert(self.canvas.lane_width > 0)
        half_lane_width = self.canvas.lane_width / (2.0*self.canvas.lane_factor)
        if 'x' in self.direction.mode:
            return -np.inf, np.inf, self.f['y']-half_lane_width, \
                self.f['y']+half_lane_width
        else:
            return self.f['x']-half_lane_width,self.f['x']+half_lane_width, \
                -np.inf, np.inf

    def closest_agent_forward(self, list_of_agents):
        """
        Return the displacement, agent that is the closest and in the 
        forward direction as self.
        """
        my_direction = self.direction
        ret_agent = None
        min_displacement = np.inf
        for agent in list_of_agents:
            displacement_unit_vector = Direction2D([agent.f['x']-self.f['x'], \
                agent.f['y']-self.f['y']])
            displacement = np.sqrt((agent.f['x']-self.f['x'])**2+\
                (agent.f['y']-self.f['y'])**2)
            if displacement < min_displacement and \
                    displacement_unit_vector.dot(my_direction) > 0:
                min_displacement = displacement
                ret_agent = agent
        if min_displacement == np.inf:
            min_displacement = -1
        return {'how_far': min_displacement, 'obj': ret_agent}
    
    def minimal_stopping_distance(self):
        """
        If we do max deceleration, what is the distance needed to stop?
        """
        curr_v = self.f['v']
        return (0.5 * curr_v ** 2) / self.MAX_ACCELERATION

    def minimal_stopping_distance_from_max_v(self):
        """
        If we do max deceleration, what is the distance needed to stop?
        """
        curr_v = self.SPEED_MAX
        return (0.5 * curr_v ** 2) / self.MAX_ACCELERATION

    def closest_stop_region_forward(self):
        """
        Return the stop region displacement, StopRegion that is closest and in
        the forward direction as self.
    
        Will not work if direction is not exactly vertical or horizontal.
        """
        assert(self.direction.mode in ['+x', '-x', '+y', '-y'])
        rx1, rx2, ry1, ry2 = self.lane_boundaries() # clipping boundaries
        ret_stopregion = None
        min_displacement = np.inf
        my_direction = self.direction
        if 'x' in self.direction.mode: stopregions = self.canvas.stopx
        if 'y' in self.direction.mode: stopregions = self.canvas.stopy
        for stopregion in stopregions:
            stopregion = stopregion.clip(rx1, rx2, ry1, ry2)
            if stopregion.empty(): continue
            centerx, centery = stopregion.center
            displacement_unit_vector = Direction2D([centerx-self.f['x'], \
                centery-self.f['y']])
            displacement = np.sqrt((centerx-self.f['x'])**2+\
                (centery-self.f['y'])**2)
            if displacement < min_displacement and \
                    displacement_unit_vector.dot(my_direction) > 0:
                min_displacement = displacement
                ret_stopregion = stopregion
        return {'how_far': min_displacement, 'obj': ret_stopregion}
    
    def closest_intersection_forward(self):
        """
        Return the intersection displacement, Intersection that is closest and 
        in the forward direction as self.
        
        Will not work if direction is not exactly vertical or horizontal.
        """
        assert(self.direction.mode in ['+x', '-x', '+y', '-y'])
        rx1, rx2, ry1, ry2 = self.lane_boundaries() # clipping boundaries
        ret_intersection = None
        min_displacement = np.inf
        my_direction = self.direction
        for intersection in self.canvas.intersections:
            clipped_intersection = intersection.clip(rx1, rx2, ry1, ry2) 
                # clip to compute displacement,
                # but return the complete intersection
            if intersection.empty(): continue
            centerx, centery = clipped_intersection.center
            displacement_unit_vector = Direction2D([centerx-self.f['x'], \
                centery-self.f['y']])
            displacement = np.sqrt((centerx-self.f['x'])**2+\
                (centery-self.f['y'])**2)
            if displacement < min_displacement and \
                    displacement_unit_vector.dot(my_direction) > 0:
                min_displacement = displacement
                ret_intersection = intersection
        return {'how_far': min_displacement, 'obj': ret_intersection}
    
    def any_agents_in_intersection(self, intersection):
        """
        Are there any agents in the given intersection?
        """
        all_agents = [agent for agent in self.canvas.agents if agent is not self]
        for agent in all_agents:
            if intersection.inside(agent.f['x'], agent.f['y']): return True
        return False

    def in_any_intersection(self):
        """
        Are we inside any intersection?
        """
        for intersection in self.canvas.intersections:
            if intersection.inside(self.f['x'], self.f['y']): return True
        return False

    def step(self, u):
        """
        u[0] is acceleration (-2, +2).
        u[1] is psi_dot (-1, +1).
        """
        if len(u) != 2:
            return

        if self.frame is not None and not self.ego_appeared:
            self.fr += 1
            assert(self.ego)
            if self.fr == self.frame:
                self.f = Features({
                    'x': self.of['x'](), 'y': self.of['y'](), 'v': self.of['v'](),
                    'acc': 0.0, 'psi_dot': 0.0, 'psi': 0.0,
                    'theta': self.of['theta'](),
                })
                if self.a != None: self.f['acc'] = self.a
                ox, oy, scale = self.canvas.ox, self.canvas.oy, self.canvas.scale
                x = ox + float(self.f['x']) * scale
                y = oy + float(self.f['y']) * scale
                # print(self.f['x'], self.f['y'], x, y, 'step', self.fr)
                self.items[0].items[0].update(x = x, y = y)
                self.ego_appeared = True
                # print('ego appeared at', self.fr)
            else:
                self.f['x'] = np.inf
                self.f['y'] = np.inf
                self.items[0].items[0].update(x = 3*self.canvas.ox, y = 3*self.canvas.oy)
        if self.centerpts is not None and self.angles is not None:
            self.fr += 1
            assert(not self.ego)
            if self.t < len(self.frames) and self.fr == self.frames[self.t]:
                ox, oy, scale = self.canvas.ox, self.canvas.oy, self.canvas.scale
                x = ox + float(self.centerpts[self.t][0]) * scale
                self.f['x'] = self.centerpts[self.t][0]
                y = oy + float(self.centerpts[self.t][1]) * scale
                self.f['y'] = self.centerpts[self.t][1]
                a = self.angles[self.t]
                self.f['theta'] = a
                self.f['v'] = np.sqrt(self.speedxs[self.t]**2+self.speedys[self.t]**2)
                self.f['acc'] = np.sqrt(self.accxs[self.t]**2+self.accys[self.t]**2)
                self.items[0].items[0].update(x = x, y = y, rotation = \
                    np.rad2deg(a))
                self.t += 1
            elif self.fr < self.frames[0] or self.fr > self.frames[-1]:
                self.f['x'] = np.inf
                self.f['y'] = np.inf
                self.f['theta'] = np.inf
                self.f['v'] = np.inf
                self.f['acc'] = np.inf
                self.items[0].items[0].update(x = 3*self.canvas.ox, y = 3*self.canvas.oy)
            return

        if not self.ego_appeared:
            return
        
        # input clipping.
        if not np.isnan(u[0]):
            if abs(u[0]) > self.MAX_ACCELERATION:
                self.f['acc'] = fastclip(u[0], -self.MAX_ACCELERATION, \
                    self.MAX_ACCELERATION)
            else:
                self.f['acc'] = u[0]

        if not np.isnan(u[1]):
            if abs(u[1]) > self.MAX_STEERING_ANGLE_RATE:
                self.f['psi_dot'] = fastclip(u[1], -self.MAX_STEERING_ANGLE_RATE,
                    self.MAX_STEERING_ANGLE_RATE)
            else:
                self.f['psi_dot'] = u[1]

        theta = 2*np.pi-self.f['theta']
        if self.method == 'kinematic_bicycle_RK4':
            K1x,K1y,K23x,K23y,K4x,K4y,K1th,K23th,K4th,v_temp,psi_temp = \
                kinematic_bicycle_RK4(self.f['v'],theta,self.f['psi'],
                    self.f['acc'],self.f['psi_dot'],self.VEHICLE_WHEEL_BASE,
                    self.DT_over_2,self.MAX_STEERING_ANGLE,self.DT)
            # print("orig", self.f['x'])
            self.f['x'] += self.DT_over_6 * (K1x + K4x) + self.DT_over_3 * K23x
            # print("new", self.f['x'])
            self.f['y'] += self.DT_over_6 * (K1y + K4y) + self.DT_over_3 * K23y
            theta += self.DT_over_6 * (K1th + K4th) + self.DT_2_over_3 * K23th
            self.f['theta'] = 2*np.pi-theta
            self.f['v'] = v_temp
            self.psi = psi_temp

        elif self.method == 'kinematic_bicycle_Euler':
            self.f['x'] += self.DT * self.f['v'] * np.cos(theta)
            self.f['y'] += self.DT * self.f['v'] * np.sin(theta)
            theta += self.DT * self.f['v'] * \
                np.tan(self.f['psi']) / self.VEHICLE_WHEEL_BASE
            self.f['theta'] = 2*np.pi-theta

            self.f['v'] = max([0.0, self.f['v'] + self.DT * self.f['acc']])
            self.f['psi'] = fastclip(self.f['psi'] + self.DT * self.f['psi_dot'],
                -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        elif self.method == 'point_mass_Euler':
            dv = self.f['acc'] * self.DT
            dx = self.f['v'] * self.DT
            self.f['v'] += dv
            if self.f['v'] < 0: self.f['v'] = 0
            self.f['x'] += self.direction.value[0] * dx
            self.f['y'] += self.direction.value[1] * dx

        else:
            # point_mass_RK4 method is not implemented yet.
            raise ValueError

        if 'kinematic' in self.method:
            theta = 2*np.pi-self.f['theta']
            orig_theta = 2*np.pi-self.direction.angle()
            if theta > orig_theta+self.THETA_DEVIATION_ALLOWED:
                theta = orig_theta+self.THETA_DEVIATION_ALLOWED
            if theta < orig_theta-self.THETA_DEVIATION_ALLOWED:
                theta = orig_theta-self.THETA_DEVIATION_ALLOWED
            self.f['theta'] = 2*np.pi-theta
        
        ox, oy, scale = self.canvas.ox, self.canvas.oy, self.canvas.scale
        x = ox + self.f['x'] * scale
        y = oy + self.f['y'] * scale
        # if self.ego:
        #     print('normal step', self.f['x'], self.f['y'], x, y)
        self.items[0].items[0].update(x = x, y = y, rotation = \
            np.rad2deg(self.f['theta']))

@numba.jit(nopython=True)
def fastclip(val, lo, hi):
    if val > hi:
        return hi
    elif val < lo:
        return lo
    else:
        return val

@numba.jit(nopython=True)
def kinematic_bicycle_RK4(v,theta,psi,acc,psi_dot,VEHICLE_WHEEL_BASE,DT_over_2,
    MAX_STEERING_ANGLE,DT):
    K1x = v * np.cos(theta)
    K1y = v * np.sin(theta)
    K1th = v * np.tan(psi) / VEHICLE_WHEEL_BASE

    theta_temp = theta + DT_over_2 * K1th
    v_temp = max([0.0, v + DT_over_2 * acc])
    psi_temp = fastclip(psi + DT_over_2 * psi_dot,
        -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    K23x = np.cos(theta_temp)
    K23y = np.sin(theta_temp)
    K23th = v_temp * np.tan(psi_temp) / VEHICLE_WHEEL_BASE

    theta_temp = theta + DT_over_2 * K23th

    K23x += np.cos(theta_temp)
    K23y += np.sin(theta_temp)
    K23x *= v_temp
    K23y *= v_temp

    v_temp = max([0.0, v + DT * acc])
    psi_temp = fastclip(psi + DT * psi_dot,
        -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    K4x = v_temp * np.cos(theta_temp)
    K4y = v_temp * np.sin(theta_temp)
    K4th = v_temp * np.tan(psi_temp) / VEHICLE_WHEEL_BASE
    return K1x,K1y,K23x,K23y,K4x,K4y,K1th,K23th,K4th,v_temp,psi_temp