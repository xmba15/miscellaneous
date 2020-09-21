#!/usr/bin/env python
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anm
import numpy as np

matplotlib.use("nbagg")


__all__ = ["Agent", "IdealRobot", "Landmark", "Map", "World"]


class Agent:
    def __init__(self, nu, omega):
        self._nu = nu
        self._omega = omega

    def decision(self, observation=None):
        return self._nu, self._omega


class IdealCamera:
    def __init__(
        self,
        env_map,
        distance_range=(0.5, 6.0),
        direction_range=(-np.pi / 3, np.pi / 3),
    ):
        self._map = env_map
        self._last_data = []

        self._distance_range = distance_range
        self._direction_range = direction_range

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[:2]

        phi = np.arctan2(diff[1], diff[0]) - cam_pose[2]

        while phi >= np.pi:
            phi -= 2 * np.pi
        while phi < -np.pi:
            phi += 2 * np.pi

        return np.array([np.hypot(*diff), phi])

    def _visible(self, polar_pos):
        return (
            self._distance_range[0] <= polar_pos[0] <= self._distance_range[1]
            and self._direction_range[0] <= polar_pos[1] < self._direction_range[1]
        )

    def data(self, cam_pose):
        observed = []
        for lm in self._map.landmarks:
            polar_pos = self.observation_function(cam_pose, lm.pos)
            if self._visible(polar_pos):
                observed.append((polar_pos, lm.id))

        self._last_data = observed

        return observed

    def draw(self, ax, elems, cam_pose):
        for lm in self._last_data:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * np.cos(direction + theta)
            ly = y + distance * np.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color="pink")


class IdealRobot:
    EPSILON = 1e-10

    def __init__(
        self,
        pose,
        agent: Optional[Agent] = None,
        sensor: Optional[IdealCamera] = None,
        color: str = "black",
    ):
        self._poses = [pose]
        self._agent = agent
        self._sensor = sensor
        self._color = color
        self._r = 0.2

    @property
    def pose(self):
        return self._poses[-1]

    def draw(self, ax, elems):
        x, y, theta = self._poses[-1]
        xn = x + self._r * np.cos(theta)
        yn = y + self._r * np.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color=self._color)
        c = patches.Circle(xy=(x, y), radius=self._r, fill=False, color=self._color)
        elems.append(ax.add_patch(c))

        elems += ax.plot(
            [e[0] for e in self._poses],
            [e[1] for e in self._poses],
            linewidth=0.5,
            color="black",
        )

        if self._sensor and len(self._poses) > 0:
            self._sensor.draw(ax, elems, self._poses[-1])

        if self._agent and hasattr(self._agent, "draw"):
            self._agent.draw(ax, elems)

    def one_step(self, time_interval):
        if not self._agent:
            return

        assert isinstance(self._agent, Agent), "agent is of type {}".format(
            type(self._agent)
        )
        observation = self._sensor.data(self._poses[-1]) if self._sensor else None
        nu, omega = self._agent.decision(observation)
        self._poses.append(
            self.state_transition(nu, omega, time_interval, self._poses[-1])
        )

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        theta0 = pose[2]
        if np.abs(omega) < IdealRobot.EPSILON:
            return (
                pose
                + np.array([nu * np.cos(theta0), nu * np.sin(theta0), omega]) * time
            )
        else:
            return pose + np.array(
                [
                    nu / omega * (np.sin(theta0 + omega * time) - np.sin(theta0)),
                    nu / omega * (-np.cos(theta0 + omega * time) + np.cos(theta0)),
                    omega * time,
                ]
            )


class Landmark:
    def __init__(self, x, y):
        self._pos = np.array([x, y])
        self._id = None

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        assert isinstance(val, int)
        self._id = val

    @property
    def pos(self):
        return self._pos

    def draw(self, ax, elems):
        c = ax.scatter(
            self._pos[0],
            self._pos[1],
            s=100,
            marker="*",
            label="landmarks",
            color="orange",
        )
        elems.append(c)
        elems.append(
            ax.text(self._pos[0], self._pos[1], "id: {}".format(self._id), fontsize=10)
        )


class Map:
    def __init__(self):
        self._landmarks = []

    @property
    def landmarks(self):
        return self._landmarks

    def append_landmark(self, landmark):
        landmark.id = len(self._landmarks)
        self._landmarks.append(landmark)

    def draw(self, ax, elems):
        [lm.draw(ax, elems) for lm in self._landmarks]


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self._objects = []
        self._time_span = time_span
        self._time_interval = time_interval
        self._debug = debug

    def append(self, obj):
        self._objects.append(obj)

    def one_step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        time_str = "t = %.2f[s]" % (self._time_interval * i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self._objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"):
                obj.one_step(self._time_interval)

    def draw(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)

        elems = []
        if self._debug:
            for i in range(1000):
                self.one_step(i, elems, ax)
        else:
            self._ani = anm.FuncAnimation(
                fig,
                self.one_step,
                fargs=(elems, ax),
                frames=int(self._time_span / self._time_interval) + 1,
                interval=int(self._time_interval * 1e3),
                repeat=False,
            )

        plt.show()
