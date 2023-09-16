import fresnel
import gsd.hoomd
import numpy as np


class FresnelGSD:
    def __init__(
            self,
            material,
            gsd_file,
            frame=0,
            color_dict=None,
            camera=fresnel.camera.Orthographic,
    ):
        self.gsd_file = gsd_file
        self._scene = fresnel.Scene()
        self.color_dict = color_dict
        self._camera = camera
        self._snapshot = None
        self._frame = None
        self.frame = frame
        self.material = fresnel.material.Material()

    @property
    def geometry(self):
        geometry = fresnel.geometry.Sphere(
            scene=self._scene,
            position=self.positions,
            radius=self.radius,
            color=fresnel.color.linear(self.colors),
        )
        geometry.material = self.material
        return geometry
    
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
        fresnel.preview(self.scene)

