from typing import List

from pydantic import BaseModel, validator, Field
from datetime import date


class Offer(BaseModel):
    sdp: str
    type: str
    video_transform: str
