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
            device=fresnel.Device()
        ):
        self.scene = fresnel.Scene()
        self._unwrap_positions = False
        self.gsd_file = gsd_file
        self._snapshot = None
        self._frame = 0 
        self.frame = frame
        self._height = height 
        self.color_dict = color_dict
        self._diameter_scale = diameter_scale
        self._device = device
        # Set fresnel.material.Material attrs
        self.solid = solid
        self._roughness = roughness
        self._specular = specular
        self._specular_trans = specular_trans
        self._metal = metal
        # Set fresnel.camera attrs
        self._view_axis = np.asarray(view_axis)
        self._up = np.array([0, 0, 1])

    @property
    def frame(self):
        return self._frame

    @frame.setter
    #TODO: Assert num frames
    def frame(self, frame):
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
        self._color_dict = value

    def set_type_color(self, particle_type, color):
        self._color_dict[particle_type] = color

    @property
    def unwrap_positions(self):
        return self._unwrap_positions

    @unwrap_positions.setter
    def unwrap_positions(self, value):
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
    def geometry(self):
        geometry = fresnel.geometry.Sphere(
            scene=self.scene,
            position=self.positions,
            radius=self.radius,
            color=fresnel.color.linear(self.colors),
        )
        geometry.material = self.material
        return geometry

    @property
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

    @property
    def camera(self):
        camera=fresnel.camera.Orthographic(
                position=self.camera_position,
                look_at=self.look_at,
                up=self.up,
                height=self.height
        )
        return camera

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

    def view(self, width=300, height=300):
        self.scene.camera = self.camera
        self.scene.geometry = [self.geometry]
        return fresnel.preview(scene=self.scene, w=width, h=height)
    
    def path_trace(self, width=300, height=300, samples=64, light_samples=1):
        self.scene.camera = self.camera
        self.scene.geometry = [self.geometry]
        return fresnel.pathtrace(
                scene=self.scene,
                w=width,
                h=height,
                samples=samples,
                light_samples=light_samples
        )

    def trace(self, width=300, height=300, n_samples=1):
        self.scene.camera = self.camera
        self.scene.geometry = [self.geometry]
        tracer = fresnel.tracer.Preview(device=self.scene.device, w=width, h=height)
        return tracer.render(self.scene)
