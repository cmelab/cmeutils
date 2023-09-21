import fresnel
import gsd.hoomd
import numpy as np


class FresnelGSD:
    def __init__(
        self,
        gsd_file,
        frame=0,
        view_axis=(1, 0, 0),
        color_dict=None,
        diameter_scale=0.30,
        height=None,
        solid=0,
        roughness=0.3,
        specular=0.5,
        specular_trans=0,
        metal=0,
        up=(0, 0, 1),
        unwrap_positions=False,
        device=fresnel.Device(),
        show_box=True,
        box_radius=0.05,
    ):
        """A wrapper class that automatically creates the Fresnel objects
        needed to view snapshots from a GSD file.

        Parameters
        ----------
        gsd_file : str, required
            Path to a GSD file to load
        frame : int, optional, default 0
            The frame of the GSD file to load the gsd.hoomd.Frame
        view_axis : np.ndarray (3,), optional, default (1, 0, 0)
            Sets the fresnel.camera attributes of camrea position and direction
        color_dict : dict, optional, default None
            Set colors for particle types
        diameter_scale : float, optional default 0.30
            Scale the diameter values stored in gsd.hoomd.Frame
        height : float, optional default 10
            Sets the fresnel.camera height attriubute
            Acts like a zoom where larger values zooms out
        solid : float, optional default 0
            fresnel.material.Material attribute.
            Sets solid colors regardless of light and angle
        roughness : float, optional default 0.3
            fresnel.material.Material attribute.
            Sets the material roughness
        specular : float, optional, default 0.5
            fresnel.material.Material attribute.
            Sets the strength of specular highlights
        specular_trans : float, optional, default 0
            fresnel.material.Material attribute.
            Sets magnitude of specular light transmission
        metal : float, optional, default 0
            fresnel.material.Material attribute.
            Sets the dielectric or metal property value of the particles.
        up : np.ndarray (3,), optional, default (0, 0, 1)
            fresnel.camera attriubute
            Determines which direction is considered up
        upwrap_positions: bool, optional, default False
            If True, the particle positions are unwrapped in the image
            This requires the GSD file snapshot contain accurate values for
            gsd.hoomd.Frame.particles.image
        device, fresnel.Device(), optional
            Set the device to be used by the scene and in rendering.
        show_box: bool, optional, default True
            If True, the box is shown in the visualization.
        box_radius: float, optional, default 0.02
            The radius of the box lines.

        """
        self.scene = fresnel.Scene()
        self.gsd_file = gsd_file
        with gsd.hoomd.open(gsd_file) as traj:
            self._n_frames = len(traj)
        self._height_factor = 1.25
        self._unwrap_positions = unwrap_positions
        self._snapshot = None
        self._view_axis = np.asarray(view_axis)
        self._frame = 0
        self.frame = frame
        self._color_dict = color_dict
        self._colors = np.array([0.5, 0.25, 0.5])
        self._diameter_scale = diameter_scale
        self._height = height
        self._device = device
        self._solid = solid
        self._roughness = roughness
        self._specular = specular
        self._specular_trans = specular_trans
        self._metal = metal
        self._up = np.asarray(up)
        self._show_box = show_box
        self._box_radius = box_radius

    @property
    def frame(self):
        """The frame of the GSD trajectory to use.
        The frame determines the particle positions usedin the image.
        """
        return self._frame

    @frame.setter
    def frame(self, frame):
        if frame > self._n_frames - 1:
            raise ValueError(f"The GSD file only has {self._n_frames} frames.")
        self._frame = frame
        with gsd.hoomd.open(self.gsd_file) as f:
            self._snapshot = f[frame]
        self.height = self._default_height()

    @property
    def snapshot(self):
        """gsd.hoomd.Frame loaded from the GSD file.
        The snapshot loaded depends on FresnelGSD.frame
        """
        return self._snapshot

    @property
    def diameter_scale(self):
        """Scales the snapshot diameter values to set FresnelGSD.radius"""
        return self._diameter_scale

    @diameter_scale.setter
    def diameter_scale(self, value):
        self._diameter_scale = value

    @property
    def color_dict(self):
        """Dictionary of key: particle type value: color"""
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
        """Set colors for particle types one at a time

        Parameters
        ----------
        particle_type : str; required
            The particle type found in FresnelGSD.particle_types
            or FresnelGSD.snapshot.particles.types
        color : np.ndarray, shape=(3,), required
            sRGB color of (red, green, blue) values
        """
        if particle_type not in set(self.particle_types):
            raise ValueError(
                f"Particle type of {particle_type} is not in the Frame"
            )
        self._color_dict[particle_type] = color

    @property
    def unwrap_positions(self):
        """If set to True, then positions of the particles are
        unwrapped before creating the image.
        This requires that the GSD file snapshots contain accurate
        image values (gsd.hoomd.Frame.particles.image

        """
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
        """Set to 1 to use solid colors regardless of light and angle"""
        return self._solid

    @solid.setter
    def solid(self, value):
        self._solid = value

    @property
    def roughness(self):
        """Sets the material roughness (ranges from 0.1 to 1)"""
        return self._roughness

    @roughness.setter
    def roughness(self, value):
        self._roughness = value

    @property
    def specular(self):
        """Determines the strength of the specular highlights
        (ranges from 0 to 1)
        """
        return self._specular

    @specular.setter
    def specular(self, value):
        self._specular = value

    @property
    def specular_trans(self):
        """Determines the magnitude of specular light transmission
        (ranges from 0 to 1)
        """
        return self._specular_trans

    @specular_trans.setter
    def specular_trans(self, value):
        self._specular_trans = value

    @property
    def metal(self):
        """Set to 0 for dielectric material or 1 for metal
        (ranges from 0 to 1)
        """
        return self._metal

    @metal.setter
    def metal(self, value):
        self._metal = value

    @property
    def view_axis(self):
        """Sets the direction and position of the camera"""
        return self._view_axis

    @view_axis.setter
    def view_axis(self, value):
        # TODO Assert is 1,3  array
        new_view_axis = np.asarray(value)
        if new_view_axis.shape != (3,):
            raise ValueError("View axis must be a 3x1 array")
        self._view_axis = np.asarray(new_view_axis)

    @property
    def up(self):
        """Determines which direction is up"""
        return self._up

    @up.setter
    def up(self, value):
        self._up = value

    @property
    def height(self):
        """Acts like a zoom. Larger values zoom out, smaller vaues zoom in"""
        if self._height is None:
            return self._default_height()
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def camera_position(self):
        """The camera position.
        Determined from box dimensions and FresnelGSD.view_axis"""
        return self.snapshot.configuration.box[:3] * self.view_axis

    @property
    def look_at(self):
        """The direction the camera is facing.
        By default, uses position directly opposite of camera position
        """
        return self.snapshot.configuration.box[:3] * -self.view_axis

    @property
    def positions(self):
        """Particle positions used in the image.
        Determined by FresnelGSD.frame
        """
        if self.unwrap_positions:
            pos = self.snapshot.particles.position
            imgs = self.snapshot.particles.image
            box_lengths = self.snapshot.configuration.box[:3]
            return pos + (imgs * box_lengths)
        else:
            return self.snapshot.particles.position

    @property
    def radius(self):
        """Sets the size of the particles.
        Determined by the gsd.hoomd.Frame.particles.diameter
        values and FresnelGSD.diameter_scale
        """
        return self.snapshot.particles.diameter * self.diameter_scale

    @property
    def particle_types(self):
        """Array of particle types of length number of particles"""
        return np.array(
            [
                self.snapshot.particles.types[i]
                for i in self.snapshot.particles.typeid
            ]
        )

    @property
    def colors(self):
        """Generates the colors by particle type,
        or sets a default of (0.5, 0.25, 0.5)
        """
        if self.color_dict:
            return np.array([self.color_dict[i] for i in self.particle_types])
        else:
            return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = value

    @property
    def box_length(self):
        """The box length of the snapshot"""
        return self.snapshot.configuration.box[:]

    def _default_height(self):
        """Set the height based on box dimensions and view axis"""
        height = np.linalg.norm(self.view_axis * self.box_length[:3])
        return height * self._height_factor

    def reset_height(self):
        """Reset the height of the camera to the default."""
        self.height = self._default_height()

    @property
    def show_box(self):
        """If True, the box is shown in the visualization."""
        return self._show_box

    @show_box.setter
    def show_box(self, value):
        self._show_box = value

    @property
    def box_radius(self):
        """The radius of the box lines."""
        return self._box_radius

    @box_radius.setter
    def box_radius(self, value):
        self._box_radius = value

    def box_geometry(self):
        return fresnel.geometry.Box(
            self.scene, self.box_length, box_radius=self.box_radius
        )

    def geometry(self):
        """Creates and returns a fresnel.geometry.Sphere object"""
        geometry = fresnel.geometry.Sphere(
            scene=self.scene,
            position=self.positions,
            radius=self.radius,
            color=fresnel.color.linear(self.colors),
        )
        geometry.material = self.material()
        return geometry

    def material(self):
        """Creates and returns a fresnel.material.Material object"""
        material = fresnel.material.Material(
            primitive_color_mix=1,
            solid=self.solid,
            roughness=self.roughness,
            specular=self.specular,
            spec_trans=self.specular_trans,
            metal=self.metal,
        )
        return material

    def camera(self):
        """Creates and returns a fresnel.camera.Orthographic object"""
        camera = fresnel.camera.Orthographic(
            position=self.camera_position,
            look_at=self.look_at,
            up=self.up,
            height=self.height,
        )
        return camera

    def view(self, width=300, height=300):
        """Creates an image using fresnel.preview

        Parameters
        ----------
        width : int, optional, default 300
            Image width size in pixels
        height : int, optional, default 300
            Image height size in pixels

        """
        self.scene.camera = self.camera()
        if self.show_box:
            self.scene.geometry = [self.geometry(), self.box_geometry()]
        else:
            self.scene.geometry = [self.geometry()]
        return fresnel.preview(scene=self.scene, w=width, h=height)

    def path_trace(self, width=300, height=300, samples=64, light_samples=1):
        """Creates an image using fresnel.pathtrace

        Parameters
        ----------
        width : int, optional, default 300
            Image width size in pixels
        height : int, optional, default 300
            Image height size in pixels
        samples : int, optional, default 64
            The number of times to sample pixels in the scene
        light_samples : int, optional, default=1
            The number of light samples for each pixel sample

        """
        self.scene.camera = self.camera()
        if self.show_box:
            self.scene.geometry = [self.geometry(), self.box_geometry()]
        else:
            self.scene.geometry = [self.geometry()]
        return fresnel.pathtrace(
            scene=self.scene,
            w=width,
            h=height,
            samples=samples,
            light_samples=light_samples,
        )
