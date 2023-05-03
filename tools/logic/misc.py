from tools.base import Policy
import time
from .controller import Controller, DefaultController, ComplexController

class RandomPolicy(Policy):
    """
    Choose a random action every timestep.
    """

    def __init__(self, action_space, slow_down_by=None):
        """
        Initialize RandomPolicy.
        """
        self.action_space = action_space
        self.slow_down_by = slow_down_by

    def act(self, s=None):
        """
        Return random action.
        """
        if self.slow_down_by != None:
            time.sleep(self.slow_down_by)
        return self.action_space.sample()


class AggressiveDrivingPolicy(Policy):
    """
    Aggressive driving policy.
    NOTE: Only for IntersectionScenario.
    closest car = CC, closest stop region = CSR,
    closest intersection forward = CIF
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, car):
        """
        Initialize AggressiveDrivingPolicy.
        """
        self.car = car
        self.K1, self.K2 = 31.6228, 7.9527
        self.A1, self.A2 = 100.0, 10.0 # arbitrary constants
        self.complexcontroller = ComplexController(
            predicates = dict(
                # Ego car
                ego = lambda p: self.car,
                # Cars in front or behind
                cars = lambda p: self.car.agents_in_front_behind(),
                # Closest car ahead
                cc = lambda p: p['ego'].closest_agent_forward(p['cars']),
                # Closest stop region ahead
                csr = lambda p: p['ego'].closest_stop_region_forward(),
                # Closest intersection ahead
                cif = lambda p: p['ego'].closest_intersection_forward(),
                # Closest car position from ego's reference, after stop
                # In the predicates if there is a closest car
                cc_after_stop = [
                    lambda p: p['cc']['obj'],
                    lambda p: p['cc']['how_far']+\
                        p['cc']['obj'].minimal_stopping_distance(),
                ],
                # Ego car from ego's reference, after stop
                ego_after_stop = lambda p: \
                    p['ego'].minimal_stopping_distance(),
                # Stop region is sufficiently away if twice the stop distance or more
                sufficiently_away_sr = lambda p: \
                    2 * p['ego'].minimal_stopping_distance(),
                # Is ego allowed to go towards closest car ahead?
                # In the predicates if there is a closest car
                allowed_to_go_towards_cc = [
                    lambda p: p['cc']['obj'],
                    lambda p: p['cc']['how_far'] > p['ego'].SAFETY_GAP,
                ],
                # Is ego allowed to go towards stop region ahead?
                # In the predicates if there is a stop region ahead
                allowed_to_go_towards_sr = [
                    lambda p: p['csr']['obj'],
                    lambda p: p['csr']['how_far'] > p['sufficiently_away_sr'],
                ],
                # Should ego decelerate to stop?
                # In the predicates if there is a stop region ahead
                should_decelerate_to_stop = [
                    lambda p: p['csr']['obj'],
                    lambda p: p['csr']['how_far'] > 0 and \
                        (p['csr']['how_far'] <= \
                            p['ego'].minimal_stopping_distance_from_max_v()) and \
                        (p['csr']['how_far'] <= \
                            p['ego'].minimal_stopping_distance())
                ],
                # Within stop region?
                within_stop_region = lambda p: p['ego'].in_allowed_regions('StopRegion'),
                # Is intersection ahead clear?
                # In the predicates if there is an intersection ahead
                intersection_clear = [
                    lambda p: p['cif']['obj'],
                    lambda p: not \
                        p['ego'].any_agents_in_intersection(p['cif']['obj']),
                ],
                # Is ego in any intersection?
                in_any_intersection = lambda p: p['ego'].in_any_intersection(),
                # Request priority when stopped and intersection is clear
                requested_priority = [
                    lambda p: p['within_stop_region'] and \
                        'intersection_clear' in p and p['intersection_clear'],
                    lambda p: \
                        p['ego'].canvas.priority_manager.request_priority(\
                            p['ego'].name),
                ],
                # Does ego have priority to go into the intersection?
                has_priority = lambda p: \
                    p['ego'].canvas.priority_manager.has_priority(\
                        p['ego'].name),
                # Give up priority when not within stop region
                released_priority = [
                    lambda p: not p['within_stop_region'] and p['has_priority'],
                    lambda p: \
                        p['ego'].canvas.priority_manager.release_priority(\
                            p['ego'].name),
                ],
            ),
            multipliers = dict(
                displacement = [
                    lambda p: p['cc']['obj'],
                    lambda p: p['cc']['how_far']-p['ego'].SAFETY_GAP,
                ],
                sr_displacement = [
                    lambda p: p['csr']['obj'],
                    lambda p: 0-p['csr']['how_far'],
                ],
                speed = [
                    lambda p: p['cc']['obj'],
                    lambda p: p['cc']['obj'].f['v']-p['ego'].f['v'],
                ],
                free_road_speed = lambda p: p['ego'].SPEED_MAX-p['ego'].f['v'],
                decelerate_speed = lambda p: 0-p['ego'].f['v'],
            ),
            controllers = [
                Controller( 
                    # if CC is facing ego => emergency stop
                    lambda p: p['cc']['obj'] and \
                        p['cc']['obj'].direction.dot(p['ego'].direction) <= 0,
                    lambda p, m: (-p['ego'].MAX_ACCELERATION, 0),
                    name = 'EmergencyStop0',
                ),
                Controller( 
                    # if after max deceleration for both ego and CC, 
                    # ego violates SAFETY_GAP => emergency stop
                    lambda p: 'cc_after_stop' in p and \
                        p['cc_after_stop']-p['ego_after_stop'] <= p['ego'].SAFETY_GAP,
                    lambda p, m: (-p['ego'].MAX_ACCELERATION, 0),
                    name = 'EmergencyStop1',
                ),
                Controller(
                    # allowed to go towards CSR and should decelerate
                    # -> decelerate to stop
                    lambda p: 'should_decelerate_to_stop' in p \
                        and p['should_decelerate_to_stop'],
                    lambda p, m: (self.K1*m['sr_displacement']+self.K2*m['decelerate_speed'], 0),
                    name = 'Decelerate0',
                ),
                Controller(
                    # in stop region and has priority and intersection is clear
                    # => free road acceleration
                    lambda p: (p['within_stop_region'] and \
                        'intersection_clear' in p and p['intersection_clear'] and \
                        p['has_priority']) or p['in_any_intersection'],
                    lambda p, m: (self.A1*m['free_road_speed'], 0),
                    name = 'Enter0'
                ),
                Controller(
                    # in stop region and does not have priority or 
                    # intersection is not clear => emergency stop
                    lambda p: p['within_stop_region'],
                    lambda p, m: (-p['ego'].MAX_ACCELERATION, 0),
                    name = 'EmergencyStop2',
                ),
                Controller(
                    # allowed to go towards CC and no stop region coming up
                    # => LQR movement
                    lambda p: not p['within_stop_region'] and \
                        'allowed_to_go_towards_cc' in p and \
                            p['allowed_to_go_towards_cc'],
                    lambda p, m: (p['cc']['obj'].f['acc'] + \
                        self.K1*m['displacement'] + self.K2*m['speed'], 0),
                    name = 'LQR0',
                ),
                DefaultController(
                    # just drive
                    lambda p, m: (self.A2*m['free_road_speed'], 0),
                    name = 'Default0',
                )
            ],
        )

    def act(self, s=None):
        """
        Act in state s.
        """
        ret_acc, ret_psi_dot = self.complexcontroller.act()
        return (ret_acc, ret_psi_dot)
