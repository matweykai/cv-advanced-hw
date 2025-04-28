from pydantic import BaseModel, Field


class Detection(BaseModel):
    cb_id: int
    bounding_box: list[int]
    x: int
    y: int
    track_id: int | None


class FramePrediction(BaseModel):
    frame_id: int
    data: list[Detection]
