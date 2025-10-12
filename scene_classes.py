from pydantic import BaseModel


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class SceneObject(BaseModel):
    name: str
    position: Vector3
    size: Vector3
    rotation_euler_angles_degrees: Vector3


class Scene(BaseModel):
    stable_diffusion_scene_skybox_prompt: str
    objects: list[SceneObject]
    camera: SceneObject


class SceneImageAnalysisResult(BaseModel):
    objects_in_first_image: list[str]
    objects_in_second_image: list[str]
    first_image_better_scene: bool
