#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
from scipy.spatial.transform import Rotation as R
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def setCameraParams(camera):
   sizeX = 1280
   sizeY = 720
   fov = 110
   focal = sizeX /(2 * math.tan(fov * 3.141516 / 360))

   print("##### Focal length: ", focal)

   camera.set_attribute('image_size_x', str(sizeX))
   camera.set_attribute('image_size_y', str(sizeY))
   camera.set_attribute('fov', str(fov))
   if camera.has_attribute('motion_blur_max_distortion'):
       camera.set_attribute('motion_blur_max_distortion', '0')

   return camera


def carla_transform_to_mat(carla_transform):
    """
    Convert a carla transform from a left-handed X-forward system (unreal)
    to a right-handed Z-forward camera pose

    :param carla_transform: the carla transform
    :return: a numpy.array with 4x4 pose matrix
    """
    camToWorld = np.matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    worldToCam = np.matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

    # Identity pose
    mat = np.eye(4)

    # Position
    mat[:3, 3] = worldToCam @ np.array([
        carla_transform.location.x,
        -carla_transform.location.y,
        carla_transform.location.z])

    # Rotation
    roll = carla_transform.rotation.roll
    pitch = -carla_transform.rotation.pitch
    yaw = -carla_transform.rotation.yaw
    worldCS = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

    mat[:3, :3] = worldToCam @ worldCS @ camToWorld

    return mat

def append_camera_pose(frame_id, transform):
    # time,x,y,z,lat,lon,altitude,pitch,yaw,roll
    T = carla_transform_to_mat(transform)

    rot = R.from_matrix(T[:3, :3]).as_quat()
    trans = T[:3, 3]
    with open('data/gt.tum','a+') as f:
        f.write('%s %s %s %s %s %s %s %s\n' % (frame_id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]))

def main():
    # Parameters
    T_vehicle_camera = carla.Transform(carla.Location(x= 1, y = 0.0, z=3), carla.Rotation(pitch=10, roll = 0, yaw=0))

    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (1280, 720),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = m.get_spawn_points()[0]
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            blueprint_library.find('vehicle.audi.tt'),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        camera_rgb = world.spawn_actor(
            setCameraParams(blueprint_library.find('sensor.camera.rgb')),
            T_vehicle_camera,
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_depth = world.spawn_actor(
            setCameraParams(blueprint_library.find('sensor.camera.depth')),
            T_vehicle_camera,
            attach_to=vehicle)
        actor_list.append(camera_depth)
        counter = -1

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_depth, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_depth = sync_mode.tick(timeout=2.0)

                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Save data
                if counter >= 0:
                    image_rgb.save_to_disk(f'data/rgb_{str(counter).zfill(3)}')
                    image_depth.save_to_disk(f'data/depth_{str(counter).zfill(3)}')
                    append_camera_pose(counter, camera_rgb.get_transform())

                # Draw the display.
                draw_image(display, image_rgb)
                draw_image(display, image_depth, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

                counter += 1

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
