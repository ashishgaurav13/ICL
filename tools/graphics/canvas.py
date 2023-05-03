import pyglet
from pyglet import gl
from tools.graphics import GrassBackground2D, Intersection2D, Lane2D, \
    StopRegion2D, TwoLaneRoad2D, Car2D, Background2D, StretchBackground2D, \
    Rectangle2D, Text2D
from tools.math import Box2D, Direction2D

class Canvas2D(pyglet.window.Window):
    """
    2D Canvas object.
    Taken from github.com/ashishgaurav13/wm2
    """

    ALLOWED_STATIC_ELEMENTS = ['GrassBackground', 'StretchBackground',
        'Rectangle', 'Lane', 'Intersection', 'TwoLaneRoad', 'Text']
    ALLOWED_AGENTS = ['Ego', 'Vehicle', 'VehicleDataInD', 'VehicleDataHighD']

    def __init__(self, w, h, static, agents, ox = 0.0, oy = 0.0, scale = 1.0,
        agentscale = 1.0):
        """
        Initialize Canvas2D.
        """
        super().__init__(w, h, visible = False)
        self.w, self.h = w, h
        self.items = []
        self.on_draw = self.event(self.on_draw)
        self.ox, self.oy, self.scale = ox, oy, scale
        self.agentscale = agentscale
        self.allowed_regions = []
        self.agents = []
        self.static_ids = {'Lane': 0, 'Intersection': 0,
            'TwoLaneRoad': 0, 'StopRegion': 0}
        self.agent_ids = {'Car': 0, 'Ego': 0}
        self.lane_width = 0
        # Only x/y stop regions are supported
        self.stopx = []
        self.stopy = []
        self.intersections = []
        self.minx, self.maxx = self.transform_x_inv(0), self.transform_x_inv(w)
        self.miny, self.maxy = self.transform_y_inv(0), self.transform_y_inv(h)
        self._add_static_elements(*static)
        self._add_agents(*agents)

    def get_static_id_and_increment(self, x):
        """
        Returns id for a static element and autoincrements for next time.
        """
        assert(x in self.static_ids.keys())
        curr_id = self.static_ids[x]
        self.static_ids[x] += 1
        return curr_id, "%s%d" % (x, curr_id)

    def get_agent_id_and_increment(self, x):
        """
        Returns id for an agent element and autoincrements for next time.
        """
        assert(x in self.agent_ids.keys())
        curr_id = self.agent_ids[x]
        self.agent_ids[x] += 1
        return curr_id, "%s%d" % (x, curr_id)

    def set_lane_width(self, x1, x2, y1, y2, factor=1.):
        """
        Sets canvas lane width by seeing which dimension is smaller (x or y).
        This function enforces a consistent lane width throughout the canvas.
        If it detects a different lane width, it will throw an error.
        """
        self.lane_factor = factor
        lane_width = abs(x1-x2) if abs(x1-x2) < abs(y1-y2) else abs(y1-y2)
        if self.lane_width == 0: self.lane_width = lane_width
        else: assert(self.lane_width == lane_width)

    def show_allowed_regions(self):
        """
        Shows elements which are allowed regions, where a vehicle can move.
        """
        for item in self.allowed_regions:
            print(item)

    def _add_static_elements(self, *args):
        """
        Add static elements to the Canvas2D.
        """
        for item in args:
            assert(type(item) == list)

            if item[0] == 'GrassBackground':
                self.items += [GrassBackground2D(self)]
            
            elif item[0] == 'Background':
                self.items += [Background2D(item[1], self)]
            
            elif item[0] == 'StretchBackground':
                self.items += [StretchBackground2D(item[1], self)]
            
            elif item[0] == "Rectangle":
                if len(item) == 5:
                    x1, x2, y1, y2 = item[1:]
                else:
                    x1, x2, y1, y2, color = item[1:]
                x1, x2 = self.transform_x(x1), self.transform_x(x2)
                y1, y2 = self.transform_y(y1), self.transform_y(y2)
                if len(item) == 5:
                    self.items += [Rectangle2D(x1, x2, y1, y2)]
                else:
                    self.items += [Rectangle2D(x1, x2, y1, y2, color)]
            
            elif item[0] == "Text":
                txt, x, y, color = item[1:]
                x, y = self.transform_x(x), self.transform_y(y)
                self.items += [Text2D(txt, x, y, color=color)]
                if txt == "t=0":
                    self.text = self.items[-1]

            elif item[0] == 'Lane':
                x1, x2, y1, y2 = item[1:]
                sid, boxname = self.get_static_id_and_increment('Lane')
                self.allowed_regions += [Box2D(x1, x2, y1, y2, boxname)]
                self.set_lane_width(x1, x2, y1, y2)
                x1, x2 = self.transform_x(x1), self.transform_x(x2)
                y1, y2 = self.transform_y(y1), self.transform_y(y2)
                self.items += [Lane2D(x1, x2, y1, y2)]

            elif item[0] == 'Intersection':
                x1, x2, y1, y2 = item[1:]
                sid, boxname = self.get_static_id_and_increment('Intersection')
                self.allowed_regions += [Box2D(x1, x2, y1, y2, boxname)]
                self.intersections.append(self.allowed_regions[-1])
                x1, x2 = self.transform_x(x1), self.transform_x(x2)
                y1, y2 = self.transform_y(y1), self.transform_y(y2)
                self.items += [Intersection2D(x1, x2, y1, y2)]

            elif item[0] == 'StopRegionX' or item[0] == 'StopRegionY':
                x1, x2, y1, y2 = item[1:]
                sid, boxname = self.get_static_id_and_increment('StopRegion')
                self.allowed_regions += [Box2D(x1, x2, y1, y2, boxname)]
                if 'X' in item[0]: self.stopx.append(self.allowed_regions[-1])
                if 'Y' in item[0]: self.stopy.append(self.allowed_regions[-1])
                x1, x2 = self.transform_x(x1), self.transform_x(x2)
                y1, y2 = self.transform_y(y1), self.transform_y(y2)
                self.items += [StopRegion2D(x1, x2, y1, y2)]

            elif item[0] == 'TwoLaneRoad':
                x1, x2, y1, y2, sep = item[1:]
                sid, boxname = self.get_static_id_and_increment('TwoLaneRoad')
                if abs(x1-x2) < abs(y1-y2):
                    width = abs(x1-x2) / 2
                    sid1, _ = self.get_static_id_and_increment('Lane')
                    laneboxname1 = "%s_Lane%d" % (boxname, sid1)
                    sid2, _ = self.get_static_id_and_increment('Lane')
                    laneboxname2 = "%s_Lane%d" % (boxname, sid2)
                    self.allowed_regions += \
                        [Box2D(x1, x1+width, y1, y2, laneboxname1)]
                    self.allowed_regions += \
                        [Box2D(x1+width, x2, y1, y2, laneboxname2)]
                    self.set_lane_width(x1, x1+width, y1, y2)
                    self.set_lane_width(x1+width, x2, y1, y2)
                else:
                    width = abs(y1-y2) / 2
                    sid1, _ = self.get_static_id_and_increment('Lane')
                    laneboxname1 = "%s_Lane%d" % (boxname, sid1)
                    sid2, _ = self.get_static_id_and_increment('Lane')
                    laneboxname2 = "%s_Lane%d" % (boxname, sid2)
                    self.allowed_regions += \
                        [Box2D(x1, x2, y1, y1+width, laneboxname1)]
                    self.allowed_regions += \
                        [Box2D(x1, x2, y1+width, y2, laneboxname2)]
                    self.set_lane_width(x1, x2, y1, y1+width)
                    self.set_lane_width(x1, x2, y1+width, y2)
                self.allowed_regions += [Box2D(x1, x2, y1, y2, boxname)]
                x1, x2 = self.transform_x(x1), self.transform_x(x2)
                y1, y2 = self.transform_y(y1), self.transform_y(y2)
                sep = sep*self.scale
                self.items += [TwoLaneRoad2D(x1, x2, y1, y2, sep)]

            else:
                print('Unsupported static element: %s' % item[0])
                print('Allowed static elements: %s' % self.ALLOWED_STATIC_ELEMENTS)
                exit(1)

    def _add_agents(self, *args):
        """
        Add agents to Canvas2D.
        """
        ego_taken = False
        for item in args:
            assert(type(item) == list)

            if item[0] == 'Ego':
                assert(not ego_taken)
                ego_taken = True
                frame = None
                a = None
                if len(item) == 5:
                    x, y, v, direction = item[1:]
                elif len(item) == 6:
                    x, y, v, direction, frame = item[1:]
                else:
                    x, y, v, direction, frame, a = item[1:]
                aid, aname = self.get_agent_id_and_increment('Ego')
                self.agents += [Car2D(x, y, v, True, direction, self, name = aname,
                    frame = frame, a=a)]

            elif item[0] == 'Vehicle':
                x, y, v, direction = item[1:]
                aid, aname = self.get_agent_id_and_increment('Car')
                self.agents += [Car2D(x, y, v, False, direction, self, name = aname)]
                # self.agents[-1].method = 'kinematic_bicycle_Euler' # 'point_mass_Euler'
            
            elif item[0].startswith('VehicleData'):
                centerpts, angles, frames, speedxs, speedys, accxs, accys, h, w = item[1:]
                ix, iy = float(centerpts[0][0]), float(centerpts[0][1])
                ia = float(angles[0])
                direction = Direction2D(mode='+x')
                aid, aname = self.get_agent_id_and_increment('Car')
                self.agents += [Car2D(ix, iy, ia, False, direction, self, name = aname,
                    centerpts=centerpts, angles=angles, h=h, w=w, frames=frames,
                    speedxs=speedxs, speedys=speedys, accxs=accxs, accys=accys)]

            else:
                print('Unsupported agent: %s' % item[0])
                print('Allowed agents: %s' % self.ALLOWED_AGENTS)
                exit(1)

    def on_draw(self):
        """
        On draw method of Pyglet window.
        """
        self.clear()
        for item in self.items:
            item.draw()
        drew_agents = 0
        if hasattr(self, 'text'):
            # curr = self.text.items[0].text
            # self.text.items[0].text = "t=%d" % (int(curr.split("=")[-1])+1)
            # print(self.text.items[0].text)
            self.text.draw()
        for agent in self.agents:
            if self.is_agent_in_bounds(agent):
                agent.draw()
                drew_agents += 1
        return drew_agents

    def is_agent_in_bounds(self, agent):
        """
        Is the agent within bounds?
        """
        return self.minx <= agent.f['x'] <= self.maxx and \
            self.miny <= agent.f['y'] <= self.maxy

    def render(self):
        """
        Render this canvas.
        """
        pyglet.app.run()
        
    def transform_x(self, x):
        """
        Transform x through translation and scaling.
        """
        return self.ox+x*self.scale
    
    def transform_y(self, y):
        """
        Transform y through translation and scaling.
        """
        return self.oy+y*self.scale
    
    def transform_x_inv(self, x):
        """
        Get original x through reverse scaling and reverse translation.
        """
        return (x-self.ox) / self.scale
    
    def transform_y_inv(self, y):
        """
        Get original y through reverse scaling and reverse translation.
        """
        return (y-self.oy) / self.scale
    