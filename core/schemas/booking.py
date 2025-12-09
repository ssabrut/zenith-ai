from pydantic import BaseModel, Field
from typing import Optional

class BookingSchema(BaseModel):
    """Schema for extracting booking details from conversation."""
    name: Optional[str] = Field(None, description="The patient's full name.")
    phone_number: Optional[str] = Field(None, description="The patient's phone number.")
    service_type: Optional[str] = Field(None, description="The specific treatment or facial type requested.")
    doctor: Optional[str] = Field(None, description="The preferred doctor's name.")
    date: Optional[str] = Field(None, description="The desired date of appointment (YYYY-MM-DD format if possible).")
    time: Optional[str] = Field(None, description="The desired time of appointment (HH:MM format).")