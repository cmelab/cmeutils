import fresnel
import gsd.hoomd
import numpy as np


class FresnelGSD:
    def __init__(
            self,
            gsd_file,
            frame=0,
            color_dict=None,
            solid=0.1,
            roughness=0.3,
            specular=0.5,
            specular_trans=0,
            metal=0,
            view_axis=(1, 0, 0)
    ):
        self.scene = fresnel.Scene()
        self.gsd_file = gsd_file
        self._snapshot = None
        self._frame = 0 
        self.frame = frame
        self._height = 5
        self.color_dict = color_dict
        # Set fresnel.material.Material attrs
        self.solid = solid
        self.roughness = roughness
        self.specular = specular
        self.specular_trans = specular_trans
        self.metal = metal
        # Set fresnel.camera attrs
        self.view_axis = np.asarray(view_axis)
        self._up = np.array([0, 0, 1])

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
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def up(self):
        return self._up
    
    @up.setter
    def up(self, value):
        self._up = value

    @property
    def snapshot(self):
        return self._snapshot

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
    def positions(self):
        return self.snapshot.particles.position

    @property
    def radius(self):
        return self.snapshot.particles.diameter / 2
    
    @property
    def particle_types(self):
        return np.array(
                [snapshot.particles.types[i] for i in snap.particles.typeid]
        )

    @property
    def colors(self):
        if self.color_dict:
            return np.array(
                    [self.color_dict[i] for i in self.particle_types]
            )
        else:
            return np.array([0.5, 0.25, 0.5])

    def view(self):
        self.scene.camera = self.camera
        self.scene.geometry = [self.geometry]
        fresnel.preview(self.scene)

