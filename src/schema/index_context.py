from pydantic import BaseModel

class ProgramContext(BaseModel):
    full_program_name: str 
    detailed_overview: str
