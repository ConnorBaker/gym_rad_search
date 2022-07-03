import math
from typing import (
    Final,
    Literal,
    NewType,
    Optional,
    TypedDict,
    get_args,
)
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
import numpy.random as npr

import matplotlib.pyplot as plt
import matplotlib.animation as animation


import gym
from gym import spaces

import visilibity as vis  # type: ignore

Action: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7]
ActionSpaceShape: Final[tuple[int]] = (len(get_args(Action)),)
# An array where index relates an action to the corresponding result.
ActionArray: TypeAlias = NewType("ActionArray", npt.NDArray[np.float64])

Point: TypeAlias = NewType("Point", npt.NDArray[np.float64])
PointSpaceShape: Final[tuple[int]] = (2,)

Observation: TypeAlias = NewType("ObsType", npt.NDArray[np.float64])
ObservationSpaceShape: Final[tuple[int]] = (
    1 + PointSpaceShape[0] + ActionSpaceShape[0],
)

RenderFrame: TypeAlias = NewType("RenderFrame", npt.NDArray[np.uint8])

Interval: TypeAlias = NewType("Interval", npt.NDArray[np.float64])
IntervalSpaceShape: Final[tuple[int]] = (2,)

BoundingBox: TypeAlias = NewType("BoundingBox", npt.NDArray[np.float64])
# Bounding box is four points
BoundingBoxSpaceShape: Final[tuple[int, int]] = (4, PointSpaceShape[0])

ObstructionSetting: TypeAlias = Literal[-1, 0, 1]

EPSILON: Final[float] = 0.0000001
DET_STEP: Final[float] = 100.0  # detector step size at each timestep in cm/s
DET_STEP_FRAC: Final[float] = 71.0  # diagonal detector step size in cm/s
DIST_TH: Final[float] = 110.0  # Detector-obstruction range measurement threshold in cm
DIST_TH_FRAC: Final[
    float
] = 78.0  # Diagonal detector-obstruction range measurement threshold in cm

# Scaling factor for noise generated for the detector.
COORD_NOISE_SCALE = 5.0

# 0: (-1)*DET_STEP     *x, ( 0)*DET_STEP     *y
# 1: (-1)*DET_STEP_FRAC*x, (+1)*DET_STEP_FRAC*y
# 2: ( 0)*DET_STEP     *x, (+1)*DET_STEP     *y
# 3: (+1)*DET_STEP_FRAC*x, (+1)*DET_STEP_FRAC*y
# 4: (+1)*DET_STEP     *x, ( 0)*DET_STEP     *y
# 5: (+1)*DET_STEP_FRAC*x, (-1)*DET_STEP_FRAC*y
# 6: ( 0)*DET_STEP     *x, (-1)*DET_STEP     *y
# 7: (-1)*DET_STEP_FRAC*x, (-1)*DET_STEP_FRAC*y


def get_actions() -> list[Action]:
    """
    Return an array of all possible actions.
    """
    return list(get_args(Action))


# If action is odd, then we are moving on the diagonal and so our step size is smaller.
# Otherwise, we're moving solely in a cardinal direction.
def get_step_size(action: Action) -> float:
    """
    Return the step size for the given action.
    """
    return DET_STEP if action % 2 == 0 else DET_STEP_FRAC


# The signs of the y-coeffecients follow the signs of sin(pi * (1 - action/4))
def get_y_step_coeff(action: Action) -> int:
    return round(math.sin(math.pi * (1.0 - action / 4.0)))


# The signs of the x-coefficients follow the signs of cos(pi * (1 - action/4)) = sin(pi * (1 - (action + 6)/4))
def get_x_step_coeff(action: Action) -> int:
    return get_y_step_coeff((action + 6) % 8)


def get_step(action: Action) -> Point:
    """
    Return the step for the given action.
    """
    return Point(
        get_step_size(action)
        * np.array((get_x_step_coeff(action), get_y_step_coeff(action)))
    )


class RadSearchConfig(TypedDict):
    observation_area: Interval
    bounding_box: BoundingBox
    obstruction_setting: ObstructionSetting
    seed: int


def obs_to_component(obs: Observation) -> tuple[float, Point, ActionArray]:
    return obs[0], Point(obs[1:3]), ActionArray(obs[3:])


def mk_rectangle(x: float, y: float) -> BoundingBox:
    return BoundingBox(
        np.array(((0.0, 0.0), (x, 0.0), (x, y), (0.0, y)))  # type: ignore
    )


class RadSearch(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}
    render_mode = "human"  # define render_mode if your environment supports rendering

    # Set these in ALL subclasses
    action_space: spaces.Discrete = spaces.Discrete(ActionSpaceShape[0])
    # The original step returned "state" instead of observation.
    # It consisted of a sensor measurement, the scaled detector coordinates,
    # and a list of detector obstruction range measurements.
    observation_space: spaces.Box

    # Set in __init__
    observation_area: Interval
    bounding_box: BoundingBox
    search_area: BoundingBox
    max_dist: float
    epoch_end: bool

    # Set in reset
    done: bool
    oob: bool
    iter_count: int
    oob_count: int
    np_random: npr.Generator
    num_obstructions: int
    obstruction_setting: ObstructionSetting
    world: vis.Environment
    polygons: list[vis.Polygon]
    graph: vis.Visibility_Graph
    epoch_count: int = 0
    source: vis.Point
    detector: vis.Point
    detector_coords: Point
    source_coords: Point
    rad_intensity: int
    background_rad_intensity: int
    rad_intensity_bounds: Interval = Interval(np.array((1e6, 10e6)))
    background_rad_intensity_bounds: Interval = Interval(np.array((10, 51)))
    prev_detector_dist: float
    detector_history: list[Point]
    measurement_history: list[float]
    obstruction_coords: list[list[Point]]
    line_segments: list[list[vis.Line_Segment]]

    # Set in step
    shortest_path_dist: float
    euclidean_dist: float
    coord_noise: bool = False

    def __init__(self, observation_area: Interval, bounding_box: BoundingBox, obstruction_setting: ObstructionSetting, seed: int) -> None:
        super().__init__()
        self.observation_area = observation_area
        self.bounding_box = bounding_box
        self.obstruction_setting = obstruction_setting

        self.search_area = BoundingBox(
            np.array(
                (
                    tuple(self.bounding_box[0] + self.observation_area),
                    (
                        self.bounding_box[1][0] - self.observation_area[0],
                        self.bounding_box[1][1] + self.observation_area[1],
                    ),
                    tuple(self.bounding_box[2] - self.observation_area),
                    (
                        self.bounding_box[3][0] + self.observation_area[0],
                        self.bounding_box[3][1] - self.observation_area[1],
                    ),
                )
            )
        )

        measurement_min = (0.0,)
        scaled_search_area_min = (-np.inf, -np.inf)
        detector_obstruction_range_min = (0.0,) * ActionSpaceShape[0]
        observation_space_min = measurement_min + scaled_search_area_min + detector_obstruction_range_min

        measurement_max = (np.inf,)
        # Since the scaled coordiantes are the current detector coordinates, plus some noise, then divided by the search area, it's possible for us to slightly overshoot 1.
        scaled_search_area_max = (np.inf,np.inf)
        detector_obstruction_range_max = (1.0,) * ActionSpaceShape[0]
        observation_space_max = measurement_max + scaled_search_area_max + detector_obstruction_range_max

        self.observation_space = spaces.Box(np.array(observation_space_min), np.array(observation_space_max), shape=ObservationSpaceShape, dtype=np.float64)

        self.max_dist: float = np.linalg.norm(self.bounding_box[2] - self.bounding_box[1])  # type: ignore
        self.epoch_end: bool = True
        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: None = None,
    ) -> Observation:
        self.done = False
        self.oob = False
        self.iter_count = 0
        self.oob_count = 0
        self.np_random = npr.default_rng(seed)

        if self.epoch_end:
            if self.obstruction_setting == -1:
                self.num_obstructions = self.np_random.integers(1, 6)  # type: ignore
            else:
                self.num_obstructions = self.obstruction_setting

            self.create_obstructions()
            walls = vis.Polygon([vis.Point(*p) for p in self.bounding_box])
            environment_line_segments = (polygon for polygon in self.polygons)
            self.world = vis.Environment([walls, *environment_line_segments])
            self.graph = vis.Visibility_Graph(self.world, EPSILON)
            self.epoch_count += 1
            self.epoch_end = False

        (
            self.source,
            self.detector,
            self.source_coords,
            self.detector_coords,
        ) = self.sample_source_loc_pos()
        self.rad_intensity = self.np_random.integers(*self.rad_intensity_bounds)  # type: ignore
        self.background_rad_intensity = self.np_random.integers(*self.background_rad_intensity_bounds)  # type: ignore
        shortest_path: vis.Polyline = self.world.shortest_path(  # type: ignore
            self.source, self.detector, self.graph, EPSILON
        )
        self.prev_detector_dist: float = shortest_path.length()
        self.detector_history = []
        self.measurement_history = []

        # Check if the environment is valid
        if not (self.world.is_valid(EPSILON)):
            print("Environment is not valid, retrying!")
            self.epoch_end = True
            self.reset()

        # TODO: The old behavior had it such that with None as an action, it followed the "you're not in an obstruction" path in step.
        return self.step(None)[0]

    def create_obstructions(self) -> None:
        """
        Method that randomly samples obstruction coordinates from 90% of search area dimensions.
        Obstructions are not allowed to intersect.
        """
        current_number_of_obstructions: int = 0
        intersect: bool = False
        self.obstruction_coords = [[] for _ in range(self.num_obstructions)]
        self.polygons = []
        self.line_segments = []
        while current_number_of_obstructions < self.num_obstructions:
            seed_x: float = self.np_random.integers(  # type: ignore
                self.search_area[0][0], self.search_area[2][0] * 0.9
            ).astype(  # type: ignore
                np.float64
            )
            seed_y: float = self.np_random.integers(  # type: ignore
                self.search_area[0][1], self.search_area[2][1] * 0.9
            ).astype(  # type: ignore
                np.float64
            )
            ext_x: float = self.np_random.integers(  # type: ignore
                *self.observation_area
            ).astype(  # type: ignore
                np.float64
            )
            ext_y: float = self.np_random.integers(  # type: ignore
                *self.observation_area
            ).astype(  # type: ignore
                np.float64
            )
            current_obstruction_coords: list[Point] = [
                Point(np.array(t))
                for t in (
                    (seed_x, seed_y),
                    (seed_x, seed_y + ext_y),
                    (seed_x + ext_x, seed_y + ext_y),
                    (seed_x + ext_x, seed_y),
                )
            ]

            if current_number_of_obstructions > 0:
                # See if the current obstruction intersects with any of the previous obstructions
                intersect = any(
                    math.isclose(
                        vis.boundary_distance(  # type: ignore
                            vis.Polygon([vis.Point(*p) for p in obs_coords]),
                            vis.Polygon(
                                [vis.Point(*p) for p in current_obstruction_coords]
                            ),
                        ),
                        0.0,
                        abs_tol=EPSILON,
                    )
                    for obs_coords in self.obstruction_coords
                    if obs_coords != []
                )

                if intersect:
                    current_obstruction_coords = []

            if not intersect:
                self.obstruction_coords[current_number_of_obstructions].extend(
                    current_obstruction_coords
                )
                polygon = vis.Polygon(
                    [vis.Point(*p) for p in current_obstruction_coords]
                )
                self.polygons.append(polygon)
                self.line_segments.append(
                    [
                        vis.Line_Segment(polygon[i], polygon[j])  # type: ignore
                        for i in (0, 2)
                        for j in (1, 3)
                    ]
                )
                current_number_of_obstructions += 1

            intersect = False

    def sample_source_loc_pos(
        self,
    ) -> tuple[vis.Point, vis.Point, Point, Point]:
        """
        Method that randomly generate the detector and source starting locations.
        Locations can not be inside obstructions and must be at least 1000 cm apart
        """

        def rand_point() -> Point:
            """
            Generate a random point within the search area.
            """
            return Point(
                self.np_random.integers(  # type: ignore
                    self.search_area[0], self.search_area[2], size=PointSpaceShape
                ).astype(np.float64)
            )

        detector = rand_point()
        detector_point = vis.Point(*detector)
        detector_clear: bool = False
        resample: bool = False
        while not detector_clear:
            resample = any(
                detector_point._in(polygon, EPSILON)  # type: ignore
                for polygon in self.polygons
            )
            if resample:
                detector = rand_point()
                detector_point = vis.Point(*detector)
                resample = False
            else:
                detector_clear = True

        source = rand_point()
        source_point = vis.Point(*source)
        source_clear: bool = False
        resample = False
        jj: int = 0
        intersect: bool = False
        num_retries: int = 0
        while not source_clear:
            # Require the source and detector to be at least 1000 cm apart
            while np.linalg.norm(source - detector) < 1000:  # type: ignore
                source = rand_point()

            # Only need to make the call to vis.Point after we're satisfied with our source location (avoids a more expensive call inside the while loop)
            source_point = vis.Point(*source)
            line_segment = vis.Line_Segment(source_point, detector_point)
            resample = any(
                source_point._in(polygon, EPSILON)  # type: ignore
                for polygon in self.polygons
            )

            intersect = not resample and any(vis.boundary_distance(line_segment, polygon) < EPSILON for polygon in self.polygons)  # type: ignore

            if (
                self.num_obstructions == 0
                or intersect
                or (num_retries > 100 and not resample)
            ):
                source_clear = True
            elif resample or not intersect:
                source = rand_point()
                source_point = vis.Point(*source)
                resample = False
                intersect = False
                num_retries += 1

        return source_point, detector_point, source, detector

    def step(
        self, action: Optional[Action]
    ) -> tuple[Observation, float, bool, dict[str, None]]:
        """
        Method that takes an action and updates the detector position accordingly.
        Returns an observation, reward, and whether the termination criteria is met.
        """
        # Move detector and make sure it is not in an obstruction
        in_obstruction: bool = False if action is None else self.check_action(action)
        if not in_obstruction:
            if any(self.detector_coords < self.search_area[0]) or any(
                self.search_area[2] < self.detector_coords
            ):
                self.oob = True
                self.oob_count += 1

            # Returns the length of a Polyline, which is a double
            # https://github.com/tsaoyu/PyVisiLibity/blob/80ce1356fa31c003e29467e6f08ffdfbd74db80f/visilibity.cpp#L1398
            self.shortest_path_dist: float = self.world.shortest_path(  # type: ignore
                self.source, self.detector, self.graph, EPSILON
            ).length()

            # TODO: In the original, this value was squared.
            # Make sure to square this value in all the places it was used in the original.
            self.euclidean_dist: float = np.linalg.norm(self.detector_coords - self.source_coords)  # type: ignore

            self.intersect = self.is_intersect()
            measurement: float = self.np_random.poisson(  # type: ignore
                self.background_rad_intensity
                if self.intersect
                else self.rad_intensity / (self.euclidean_dist**2.0)
                + self.background_rad_intensity
            )

            # Reward logic
            if self.shortest_path_dist < 110:
                reward: float = 0.1
                self.done = True
            elif self.shortest_path_dist < self.prev_detector_dist:
                reward: float = 0.1
                self.prev_detector_dist = self.shortest_path_dist
            else:
                reward: float = -0.5 * self.shortest_path_dist / self.max_dist

        else:
            # If detector starts on obs. edge, it won't have the sp_dist calculated
            if self.iter_count > 0:
                measurement: float = self.np_random.poisson(
                    self.background_rad_intensity
                    if self.intersect
                    else self.rad_intensity / (self.euclidean_dist**2.0)
                    + self.background_rad_intensity
                )
            else:
                assert self.shortest_path_dist > self.euclidean_dist
                self.shortest_path_dist = self.prev_detector_dist
                assert self.shortest_path_dist > self.euclidean_dist
                # TODO: In the original, this value was squared.
                # Make sure to square this value in all the places it was used in the original.
                self.euclidean_dist: float = np.linalg.norm(self.detector_coords - self.source_coords)  # type: ignore
                self.intersect = self.is_intersect()
                measurement: float = self.np_random.poisson(  # type: ignore
                    self.background_rad_intensity
                    if self.intersect
                    else self.rad_intensity / (self.euclidean_dist**2.0)
                    + self.background_rad_intensity
                )

            reward: float = -0.5 * self.shortest_path_dist / self.max_dist

        # If detector coordinate noise is desired
        noise: Point = Point(
            self.np_random.normal(scale=COORD_NOISE_SCALE, size=PointSpaceShape)
            if self.coord_noise
            else np.zeros(PointSpaceShape)
        )

        # Scale detector coordinates by search area
        det_coord_x, det_coord_y = self.detector_coords + noise
        search_area_x_max, search_area_y_max = self.search_area[2]
        det_coord_scaled = np.array([det_coord_x / search_area_x_max, det_coord_y / search_area_y_max])

        # TODO: The agent is allowed to wander freely out of bounds (and we set self.oob and increment the out of bounds counter). What does calculation of scaled coordinates relative to the search area mean to us?
        # Since the detector coordinates range from -inf to inf because nothing prevents it from going, and staying, out of bounds, the scaled value won't be in 0 to 1 but rather in -inf to inf.
        # assert np.all(det_coord_scaled <= 1.0)


        # Observation with the radiation meas., detector coords and detector-obstruction range meas.
        # TODO: State should really be better organized. If there are distinct components to it, why not make it
        # a named tuple?
        sensor_meas: ActionArray
        if self.num_obstructions > 0:
            sensor_meas = ActionArray(np.array(self.dist_sensors()))
        else:
            sensor_meas = ActionArray(np.zeros(ActionSpaceShape))

        # State is an 11-tuple ndarray
        state: Observation = Observation(
            np.array([measurement, *det_coord_scaled, *sensor_meas])
        )
        self.oob = False
        self.detector_history.append(self.detector_coords)
        self.measurement_history.append(measurement)
        self.iter_count += 1

        return state, round(reward, 2), self.done if self.oob_count == 0 else True, {}

    # TODO: Name is dishonest. If the action is valid, it actually *takes* the action!
    def check_action(self, action: Action) -> bool:
        """
        Method that checks which direction to move the detector based on the action.
        If the action moves the detector into an obstruction, the detector position
        will be reset to the prior position.
        """
        step: Point = get_step(action)
        self.detector = vis.Point(*(self.detector_coords + step))

        in_obstruction: bool = self.in_obstruction()
        if in_obstruction:
            # If we're in an obsticle, roll back
            self.detector = vis.Point(*self.detector_coords)
        else:
            # If we're not in an obsticle, update the detector coordinates
            self.detector_coords = Point(np.array((self.detector.x(), self.detector.y())))  # type: ignore

        return in_obstruction

    def in_obstruction(self) -> bool:
        """
        Method that checks if the detector position intersects or is inside an obstruction.
        """
        return any(
            self.detector._in(polygon, EPSILON)  # type: ignore
            for polygon in self.polygons
        )

    # TODO: Better name! los_blocked?
    def is_intersect(self, threshold: float = 0.001) -> bool:
        """
        Method that checks if the line of sight is blocked by any obstructions in the environment.
        """

        # Close enough that we're touching the source
        if math.isclose(self.euclidean_dist, self.shortest_path_dist, abs_tol=0.1):
            return True

        line_segment = vis.Line_Segment(self.detector, self.source)
        return any(
            vis.boundary_distance(line_segment, polygon) < threshold  # type: ignore
            for polygon in self.polygons
        )

    # TODO: Return ActionArray
    def dist_sensors(self) -> list[float]:
        """
        Method that generates detector-obstruction range measurements with values between 0-1.
        """
        line_segs: list[vis.Line_Segment] = [
            vis.Line_Segment(
                self.detector, vis.Point(*(self.detector_coords + get_step(action)))
            )
            for action in get_actions()
        ]
        # TODO: Currently there are only eight actions -- what happens if we change that?
        # This annotation would need to change as well.
        dists: list[float] = [0.0] * len(line_segs)
        obstruction_idx_line_segs: list[int] = [0] * len(self.polygons)
        intersections: int = 0
        line_seg_dist: list[float] = [0.0] * 4
        if self.num_obstructions > 0:
            for idx_line_segs, line_seg in enumerate(line_segs):
                # TODO: Declare self.line_segs (in the case we didn't already and named it line_segments).
                # TODO: Make clear the difference between idx_line_segs and line_segs_idx.
                for obstruction_idx, polygon in enumerate(self.line_segments):
                    for line_seg_idx, obstruction_line_seg in enumerate(polygon):
                        if intersections < 2 and vis.intersect(obstruction_line_seg, line_seg, EPSILON):  # type: ignore
                            # check if step direction intersects poly seg
                            line_seg_dist[line_seg_idx] = (  # type: ignore
                                DIST_TH - vis.distance(line_seg.first(), obstruction_line_seg)  # type: ignore
                            ) / DIST_TH
                            intersections += 1
                            obstruction_idx_line_segs[obstruction_idx] += 1
                    if intersections > 0:
                        dists[idx_line_segs] = max(line_seg_dist)
                        line_seg_dist = [0.0] * 4
                intersections = 0
            # If there are more than three dists equal to one, we need to correct the coordinates.
            if sum(filter(lambda x: x == 1.0, dists)) > 3:
                # Take the polygon which corresponds to the index with the maximum number of intersections.
                argmax = max(zip(obstruction_idx_line_segs, self.polygons))[1]
                dists = self.correct_coords(argmax)
        return dists

    # TODO: Return ActionArray
    def correct_coords(self, polygon: vis.Polygon) -> list[float]:
        """
        Method that corrects the detector-obstruction range measurement if more than the correct
        number of directions are being activated due to the Visilibity implementation.
        """
        x_check: list[bool] = [False] * ActionSpaceShape[0]
        dist = 0.1
        length = 1

        qs: list[Point] = [self.detector_coords.copy()] * ActionSpaceShape[0]
        dists: list[float] = [0.0] * ActionSpaceShape[0]
        while not any(x_check):
            for action in get_actions():
                step = Point(get_step(action) / get_step_size(action) * dist * length)
                qs[action] = Point(qs[action] + step)
                if vis.Point(*qs[action])._in(polygon, EPSILON):  # type: ignore
                    x_check[action] = True

        # i.e. if one outside the poly then
        if sum(x_check) >= 4:
            for ii in [0, 2, 4, 6]:
                if x_check[ii - 1] and x_check[ii + 1]:
                    dists[ii - 1 : ii + 2] = [1.0, 1.0, 1.0]

        return dists

    def render(self, mode: Literal["human", "rgb_array"] = "human") -> Optional[list[RenderFrame]]:
        # TODO: Do we need to render everything here? Can we initialize the environment in the constructor?
        # TODO: Does it make sense to render an animation for the human case?
        fig, axes = plt.subplots()
        (line,) = plt.plot([], "bo")

        def init_chart() -> tuple[plt.Line2D]:
            x_max, y_max = self.bounding_box[2]
            observation_area_x, observation_area_y = self.observation_area
            axes.set_xlim(0, x_max)
            axes.set_ylim(0, y_max)

            # Draw the border around the search area
            left = axes.axvspan(0, observation_area_x, facecolor="gray")
            bottom = axes.axhspan(0, observation_area_y, facecolor="gray")
            right = axes.axvspan(x_max - observation_area_x, x_max, facecolor="gray")
            top = axes.axhspan(y_max - observation_area_y, y_max, facecolor="gray")

            # Draw the polygons
            for obstruction in self.obstruction_coords:
                axes.fill(*zip(*obstruction), "orange")

            # Draw the source
            axes.plot(*self.source_coords, "go")

            return (line,)

        # Must return an iterable, so we return a singleton
        def update(frame: npt.NDArray[np.float64]) -> tuple[plt.Line2D]:
            line.set_data(*frame)
            return (line,)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=env.detector_history,
            init_func=init_chart,
            interval=200,
            blit=False,
        )

        if mode == "human":
            plt.show()
            return None
        
        # TODO: Support rendering for the rgb_array mode.
        # elif mode == "rgb_array":
        #     canvas = FigureCanvasAgg(fig)
        #     canvas.draw()
        #     render_in_mem: memoryview = canvas.buffer_rgba()
        #     arr: npt.NDArray[np.uint8] = np.asarray(render_in_mem)
        #     return RenderFrame(arr)

        raise NotImplementedError


if __name__ == "__main__":
    observation_area = Interval(np.array((200.0, 500.0)))
    bounding_box = mk_rectangle(2700.0, 2700.0)
    env_config = RadSearchConfig(
        observation_area=observation_area,
        bounding_box=bounding_box,
        obstruction_setting=-1,
        seed=0,
    )
    env = RadSearch(**env_config)

    for _ in range(10):
        print("Current state:")
        print(f"\tDetector: {env.detector_coords}")
        print(f"\tDetector vis.Point: {env.detector.x(), env.detector.y()}")
        print(f"\tSource: {env.source_coords}")
        print(f"\tSource vis.Point: {env.source.x(), env.source.y()}")
        print(f"\tNum Obstructions: {env.num_obstructions}")
        print(f"\t\tObstructions: {env.obstruction_coords}")
        print(f"\tShortest Path: {env.shortest_path_dist}")
        print(f"\tEuclidean Dist: {env.euclidean_dist}")
        print(f"\tIs OOB: {env.oob}")
        print(f"\tIs Obstructed: {env.in_obstruction()}")
        print(f"\tIs Intersect: {env.is_intersect()}")
        print(f"\tDist Sensors: {env.dist_sensors()}")
        action = env.action_space.sample()
        print(f"Action: {action}")
        print(f"\tDelta: {get_step(action)}\n")
        env.step(action)
    env.render()
    env.close()
