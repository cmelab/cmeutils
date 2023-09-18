import fresnel
import gsd.hoomd
import numpy as np


class FresnelGSD:
    def __init__(
        self,
        gsd_file,
        frame=0,
        color_dict=None,
        diameter_scale=0.30,
        solid=0.1,
        roughness=0.3,
        specular=0.5,
        specular_trans=0,
        metal=0,
        view_axis=(1, 0, 0),
        height=10,
        up=(0, 0, 1),
        unwrap_positions=False,
        device=fresnel.Device()
    ):
        self.scene = fresnel.Scene()
        self.gsd_file = gsd_file
        with gsd.hoomd.open(gsd_file) as traj:
            self._n_frames = len(traj)
        self._unwrap_positions = unwrap_positions 
        self._snapshot = None
        self._frame = 0 
        self.frame = frame
        self._height = height 
        self._color_dict = color_dict
        self._diameter_scale = diameter_scale
        self._device = device
        self._solid = solid
        self._roughness = roughness
        self._specular = specular
        self._specular_trans = specular_trans
        self._metal = metal
        self._view_axis = np.asarray(view_axis)
        self._up = np.asarray(up) 

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        if frame > self._n_frames - 1:
            raise ValueError(
                    f"The GSD file only has {self._n_frames} frames."
            )
        self._frame = frame
        with gsd.hoomd.open(self.gsd_file) as f:
            self._snapshot = f[frame]

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def diameter_scale(self):
        return self._diameter_scale

    @diameter_scale.setter
    def diameter_scale(self, value):
        self._diameter_scale = value

    @property
    def color_dict(self):
        return self._color_dict

    @color_dict.setter
    def color_dict(self, value):
        if not isinstance(value, dict):
            raise ValueError(
                    "Pass in a dicitonary with "
                    "keys of particle type, values of color"
            )
        self._color_dict = value

    def set_type_color(self, particle_type, color):
        if not particle_type in set(self.particle_types):
            raise ValueError(
                    f"Particle type of {particle_type} is not in the Snapshot"
            )
        self._color_dict[particle_type] = color

    @property
    def unwrap_positions(self):
        return self._unwrap_positions

    @unwrap_positions.setter
    def unwrap_positions(self, value):
        if not isinstance(value, bool):
            raise ValueError(
                "Set to True or False where "
                "True uses unwrapped particle positions"
        )
        self._unwrap_positions = value

    @property
    def solid(self):
        return self._solid

    @solid.setter
    def solid(self, value):
        self._solid = value

    @property
    def roughness(self):
        return self._roughness

    @roughness.setter
    def roughness(self, value):
        self._roughness = value

    @property
    def specular(self):
        return self._specular

    @specular.setter
    def specular(self, value):
        self._specular = value

    @property
    def specular_trans(self):
        return self._specular_trans

    @specular_trans.setter
    def specular_trans(self, value):
        self._specular_trans = value

    @property
    def metal(self):
        return self._metal

    @metal.setter
    def metal(self, value):
        self._metal = value

    @property
    def view_axis(self):
        return self._view_axis

    @view_axis.setter
    def view_axis(self, value):
        #TODO Assert is 1,3  array
        self._view_axis = np.asarray(value)

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, value):
        self._up = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def camera_position(self):
        return (self.snapshot.configuration.box[:3] * self.view_axis) - 0.5

    @property
    def look_at(self):
        return self.snapshot.configuration.box[:3] * -self.view_axis 

    @property
    def positions(self):
        if self.unwrap_positions:
           pos = self.snapshot.particles.position
           imgs = self.snapshot.particles.image
           box_lengths = self.snapshot.configuration.box[:3]
           return pos + (imgs * box_lengths)
        else:
            return self.snapshot.particles.position

    @property
    def radius(self):
        return self.snapshot.particles.diameter * self.diameter_scale 
    
    @property
    def particle_types(self):
        return np.array(
                [
                    self.snapshot.particles.types[i] for
                    i in self.snapshot.particles.typeid
                ]
        )

    @property
    def colors(self):
        if self.color_dict:
            return np.array(
                    [self.color_dict[i] for i in self.particle_types]
            )
        else:
            return np.array([0.5, 0.25, 0.5])

    def geometry(self):
        geometry = fresnel.geometry.Sphere(
            scene=self.scene,
            position=self.positions,
            radius=self.radius,
            color=fresnel.color.linear(self.colors),
        )
        geometry.material = self.material()
        return geometry

    def material(self):
        material = fresnel.material.Material(
                primitive_color_mix=1,
                solid=self.solid,
                roughness=self.roughness,
                specular=self.specular,
                spec_trans=self.specular_trans,
                metal=self.metal
        )
        return material

    def camera(self):
        camera=fresnel.camera.Orthographic(
                position=self.camera_position,
                look_at=self.look_at,
                up=self.up,
                height=self.height
        )
        return camera


    def view(self, width=300, height=300):
        self.scene.camera = self.camera()
        self.scene.geometry = [self.geometry()]
        return fresnel.preview(scene=self.scene, w=width, h=height)
    
    def path_trace(self, width=300, height=300, samples=64, light_samples=1):
        self.scene.camera = self.camera()
        self.scene.geometry = [self.geometry()]
        return fresnel.pathtrace(
                scene=self.scene,
                w=width,
                h=height,
                samples=samples,
                light_samples=light_samples
        )

    def trace(self, width=300, height=300, n_samples=1):
        self.scene.camera = self.camera()
        self.scene.geometry = [self.geometry()]
        tracer = fresnel.tracer.Preview(device=self.scene.device, w=width, h=height)
        return tracer.render(self.scene)
