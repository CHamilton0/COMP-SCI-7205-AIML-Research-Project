from pydantic import BaseModel, Field

# Utility classes for generation of 3D scenes


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
    scene_skybox_prompt: str
    scene_skybox_negative_prompt: str
    objects: list[SceneObject]
    camera: SceneObject


class SceneImageAnalysisResult(BaseModel):
    objects_in_first_image: list[str]
    objects_in_second_image: list[str]
    first_image_better_scene: bool


class SceneImageRatingResult(BaseModel):
    # Scores from 1 to 10
    background_score: int = Field(..., ge=1, le=10)
    objects_score: int = Field(..., ge=1, le=10)
    layout_score: int = Field(..., ge=1, le=10)
